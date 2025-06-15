import os
import io
import logging
import tempfile
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy import signal


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AudioProcessor:
    """
    AudioProcessor: clean and normalize speech for TTS.

    Provides silence trimming, comfort-noise insertion, high-pass filtering,
    loudness normalization (LUFS), peak limiting, and safe file I/O.
    """

    def __init__(
        self,
        target_sr: int = 48_000,
        target_lufs: float = -24.0,
        peak_ceiling_db: float = -3,
        hpf_cutoff: int = 80,
        butter_order: int = 2,
        target_silence_ms: int = 200,
        fade_ms: int = 10,
        silence_margin_db: float = 3,
        seed: Optional[int] = None,
        use_memory_io: bool = False,
    ):

        self.target_sr = target_sr
        self.target_lufs = target_lufs
        self.peak_ceiling = peak_ceiling_db
        self.hpf_cutoff = hpf_cutoff
        self.butter_order = butter_order
        self.target_silence_ms = target_silence_ms
        self.fade_ms = fade_ms
        self.silence_margin_db = silence_margin_db

        # reproducible noise RNG
        self.seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

        # in-memory I/O switch
        self.use_memory_io = use_memory_io

        # reuse pyloudnorm meter if needed
        self._meter = pyln.Meter(self.target_sr)

    def update_config(
        self,
        *,
        target_sr: Optional[int] = None,
        target_lufs: Optional[float] = None,
        peak_ceiling_db: Optional[float] = None,
        hpf_cutoff: Optional[int] = None,
        butter_order: Optional[int] = None,
        target_silence_ms: Optional[int] = None,
        fade_ms: Optional[int] = None,
        silence_margin_db: Optional[float] = None,
        seed: Optional[int] = None,
        use_memory_io: Optional[bool] = None,
    ) -> None:
        """
        Update processing parameters in-place.
        Only non-None args override the current settings.
        """
        if target_sr is not None:
            self.target_sr = target_sr
            # rebuild loudness meter at the new sample rate
            self._meter = pyln.Meter(self.target_sr)

        if target_lufs is not None:
            self.target_lufs = target_lufs

        if peak_ceiling_db is not None:
            self.peak_ceiling = peak_ceiling_db

        if hpf_cutoff is not None:
            self.hpf_cutoff = hpf_cutoff

        if butter_order is not None:
            self.butter_order = butter_order

        if target_silence_ms is not None:
            self.target_silence_ms = target_silence_ms

        if fade_ms is not None:
            self.fade_ms = fade_ms

        if silence_margin_db is not None:
            self.silence_margin_db = silence_margin_db

        if seed is not None:
            self.seed = seed
            # re-seed RNG for reproducible noise
            self._rng = np.random.RandomState(seed)

        if use_memory_io is not None:
            self.use_memory_io = use_memory_io

    def _butter_highpass(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Apply zero-phase Butterworth high-pass filter to remove low-frequency rumble."""
        # ensure valid cutoff
        nyq = 0.5 * sr
        if not 0 < self.hpf_cutoff < nyq:
            raise ValueError(f"hpf_cutoff must be between 0 and {nyq}, got {self.hpf_cutoff}")

        # design filter
        b, a = signal.butter(self.butter_order, self.hpf_cutoff / nyq, btype="high", analog=False)

        # zero-phase filter: filtfilt requires at least 3× max(len(a),len(b)) samples
        if len(x) < max(len(a), len(b)) * 3:
            return x
        return signal.filtfilt(b, a, x)

    def _loudness_normalize(self, x: np.ndarray, sr: int) -> np.ndarray:
        """Adjust integrated loudness of `x` to `self.target_lufs` LUFS."""
        # ensure meter matches audio SR
        if sr != self._meter.rate:
            meter = pyln.Meter(sr)
        else:
            meter = self._meter
        loudness = meter.integrated_loudness(x)

        logger.debug("Current LUFS=%0.2f -> Target LUFS=%0.2f", loudness, self.target_lufs)
        return pyln.normalize.loudness(x, loudness, self.target_lufs)

    def _peak_normalize(self, x: np.ndarray, peak_ceiling: float) -> np.ndarray:
        """Limit samples so that max(|x|) < 10^(peak_ceiling/20)."""
        ceiling_lin = 10 ** (peak_ceiling / 20)

        peak = np.max(np.abs(x))
        if peak == 0 or peak <= ceiling_lin:
            return x
        factor = ceiling_lin / peak
        out = x * factor
        logger.debug("Peak=%0.3f -> factor=%0.3f", peak, factor)
        return out

    def _apply_fade(self, x: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
        """Fade in/out over `fade_ms` milliseconds to avoid clicks."""
        fade_samples = int(fade_ms * sr / 1000)
        if fade_samples and len(x) > 2 * fade_samples:
            x = x.copy()
            fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=False)
            fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False)
            x[:fade_samples] *= fade_in
            x[-fade_samples:] *= fade_out
        return x

    def _crossfade(self, a: np.ndarray, b: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
        """Overlap tails of `a` and heads of `b` over `fade_ms` ms for seamless joins."""
        fade_samples = int(fade_ms * sr / 1000)
        if fade_samples <= 0 or len(a) < fade_samples or len(b) < fade_samples:
            return np.concatenate([a, b])

        a_end = a[-fade_samples:]
        b_start = b[:fade_samples]

        # build new overlap region
        fade_in = np.linspace(0.0, 1.0, fade_samples, endpoint=False, dtype=a.dtype)
        fade_out = np.linspace(1.0, 0.0, fade_samples, endpoint=False, dtype=a.dtype)
        overlap = a_end * fade_out + b_start * fade_in

        return np.concatenate([a[:-fade_samples].copy(), overlap, b[fade_samples:].copy()])

    def energy_vad(self, x: np.ndarray, sr: int, frame_ms: float = 20.0, hop_ms: float = 10.0, noise_pct: float = 10.0) -> tuple[float, float]:
        """
        Estimate a dB threshold for silence by computing RMS per frame, taking the `noise_pct`-percentile as noise-floor
        """
        frame_len = int(sr * frame_ms / 1000)
        hop_len = int(sr * hop_ms / 1000)
        # how many full frames fit?
        n_frames = 1 + max(0, (len(x) - frame_len) // hop_len)

        energies_db = np.empty(n_frames, dtype=np.float32)
        for i in range(n_frames):
            start = i * hop_len
            frame = x[start: start + frame_len]
            rms = np.sqrt(np.mean(frame**2)) or 1e-12
            energies_db[i] = 20 * np.log10(rms)

        # pick a low percentile as the noise floor
        noise_floor_db = float(np.percentile(energies_db, noise_pct))
        top_db = max(0, abs(noise_floor_db + self.silence_margin_db))

        return noise_floor_db, top_db

    def _generate_comfort_noise(self, length: int, db_level: float) -> np.ndarray:
        """Generate Gaussian noise of `length` samples at `db_level` dBFS RMS."""
        # For amplitude ratios, dB = 20 * log10(linear), so linear = 10^(dB/20).
        db_level = max(-80.0, min(0.0, db_level))
        rms_lin = 10 ** (db_level / 20)
        noise = self._rng.randn(length).astype(np.float32)

        # Measure the current RMS of the raw noise
        cur_rms = np.sqrt(np.mean(noise**2))
        if cur_rms > 0:
            # Scale noise so its RMS matches the target linear RMS
            noise *= rms_lin / cur_rms

        return noise

    def _trim_speech_region(self, x: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
        """Trim leading/trailing silence based on VAD, keeping a small 2x`fade_ms` margin. No actual fade."""

        _, top_db = self.energy_vad(x, sr)
        pad_ms = 2 * self.fade_ms if fade_ms is None else 2 * fade_ms  # 2x fade default fade ms arbitrary setting

        intervals = librosa.effects.split(x, top_db=top_db)
        if intervals.size == 0:
            logger.debug("trim_speech_region: no voiced intervals; returning full chunk")
            return x

        # Compute margin in samples
        margin = int(pad_ms * sr / 1000)
        start = max(0, intervals[0, 0] - margin)
        end = min(len(x), intervals[-1, 1] + margin)

        logger.debug("trim_speech_region: keeping samples [%d:%d] (margin %d samples, top_db=%.1f)", start, end, margin, top_db)
        return x[start:end]

    def remove_edge_silence(self, x: np.ndarray, sr: int, fade_ms: int) -> np.ndarray:
        """Trim silence from start/end then apply a fade of `fade_ms` ms."""
        _, top_db = self.energy_vad(x, sr)

        # Trim edges
        y, _ = librosa.effects.trim(x, top_db=top_db)
        # Fade
        return self._apply_fade(y, sr, fade_ms)

    def remove_internal_silence(self, x: np.ndarray, sr: int, target_silence_ms: int, fade_ms: int) -> np.ndarray:
        """Replace internal gaps > `target_silence_ms` ms with comfort noise and crossfade all joins with `fade_ms` ms overlaps."""
        noise_floor_db, top_db = self.energy_vad(x, sr)
        silence_margin_db = 2*self.silence_margin_db
        target_silence_samples = int(target_silence_ms * sr / 1000)

        intervals = librosa.effects.split(x, top_db=top_db)
        if len(intervals) < 2:
            logger.debug("remove_internal_silence: %d interval(s), nothing to do", len(intervals))
            return x

        out_chunks: List[np.ndarray] = []
        collapsed = 0

        for idx, (start, end) in enumerate(intervals):
            if idx > 0:
                prev_end = intervals[idx - 1][1]
                gap = start - prev_end

                if gap <= target_silence_samples:
                    # short gap: keep original
                    chunk = x[prev_end:start]
                else:
                    # long gap: replace with comfort noise at (trim_silence_db) dBFS
                    collapsed += 1
                    chunk = self._generate_comfort_noise(target_silence_samples, noise_floor_db - silence_margin_db)

                out_chunks.append(chunk)

            # voiced segment, no fades
            seg = x[start:end]
            out_chunks.append(seg)

        logger.info("remove_internal_silence: collapsed %d gaps > %dms into comfort noise", collapsed, target_silence_ms)

        # Now crossfade every boundary
        result = out_chunks[0]
        for next_chunk in out_chunks[1:]:
            result = self._crossfade(a=result, b=next_chunk, sr=sr, fade_ms=fade_ms)

        return result

    def process_chunk_array(self, audio: np.ndarray, sr: int, target_silence_ms: int | None, fade_ms: int | None) -> np.ndarray:
        """Quick per-chunk cleanup: trim speech region, high-pass filter, and collapse internal silence."""
        target_silence_ms = target_silence_ms if target_silence_ms is not None else self.target_silence_ms
        fade_ms = fade_ms if fade_ms is not None else self.fade_ms

        audio = self._trim_speech_region(x=audio, sr=sr, fade_ms=fade_ms)
        audio = self._butter_highpass(audio, sr)
        audio = self.remove_internal_silence(x=audio, sr=sr, target_silence_ms=target_silence_ms, fade_ms=fade_ms)

        return audio

    def process_array(
        self, audio: np.ndarray, sr: int, target_silence_ms: int | None, fade_ms: int | None, modify_sr: bool = False
    ) -> Tuple[np.ndarray, int]:
        """
        Full-pipeline cleanup:
          1. Downmix to mono
          2. Remove edge silence + fade
          3. High-pass filter
          4. Collapse internal silence
          5. (Optional) Resample to `self.target_sr`
          6. Loudness normalize to `self.target_lufs` LUFS
          7. Peak-limit to `self.peak_ceiling` dB
        Returns (processed_audio, new_sr).
        """
        target_silence_ms = target_silence_ms if target_silence_ms is not None else self.target_silence_ms
        fade_ms = fade_ms if fade_ms is not None else self.fade_ms
        # Mono (check correct axis)
        if audio.ndim > 1:
            audio = audio.mean(axis=1) * np.sqrt(2)

        # Trim silence
        audio = self.remove_edge_silence(x=audio, sr=sr, fade_ms=fade_ms)
        # High-pass (DC/rumble removal)
        audio = self._butter_highpass(audio, sr)
        # Remove long gaps
        audio = self.remove_internal_silence(x=audio, sr=sr, target_silence_ms=target_silence_ms, fade_ms=fade_ms)
        # Resample if desired -> cast to float32
        if (sr != self.target_sr) and modify_sr:
            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr
        # Loudness normalization (LUFS)
        audio = self._loudness_normalize(audio, sr)
        # Peak ceiling
        audio = self._peak_normalize(x=audio, peak_ceiling=self.peak_ceiling)

        return audio.astype(np.float32), sr

    def process_file(self, src: Path, dst: Path, target_silence_ms: int | None, fade_ms: int | None) -> None:
        """Read from `src`, run `process_array`, write WAV to `dst`"""
        target_silence_ms = target_silence_ms if target_silence_ms is not None else self.target_silence_ms
        fade_ms = fade_ms if fade_ms is not None else self.fade_ms
        try:
            audio, sr = sf.read(str(src), always_2d=False)
        except Exception as e:
            logger.error("Could not read %s: %s", src, e)
            raise

        audio, sr = self.process_array(audio=audio, sr=sr, target_silence_ms=target_silence_ms, fade_ms=fade_ms)

        if self.use_memory_io:
            # In-memory buffer is cross-platform
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format='WAV', subtype='PCM_16')
            buf.seek(0)
            with open(dst, 'wb') as f:
                f.write(buf.read())
        else:
            # Use mkstemp for safe temp file creation on Windows, Linux, macOS
            fd, tmp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)  # Close OS-level handle so sf.write can open it
            try:
                sf.write(tmp_path, audio, sr, subtype='PCM_16')
                os.replace(tmp_path, str(dst))
            except Exception:
                # Clean up temp file on failure
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
                raise

        logger.debug(f"Processed {src} -> {dst}")
