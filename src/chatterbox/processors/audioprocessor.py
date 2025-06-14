import os
import logging
import tempfile
from pathlib import Path
from typing import Literal, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln
from scipy import signal
import ffmpeg


# constants
TARGET_SR = 48_000  # Target sample rate in Hz. Typical: 44_100 (CD), 48_000 (video), 96_000 (high-res)
# Integrated loudness target in LUFS. Typical: –23 (broadcast), –16 (podcast), –14 (streaming services)
TARGET_LUFS = -24.0
PEAK_CEILING = -3.0  # True-peak ceiling in dBTP. Typical: –3 to –1 dBTP
HPF_CUTOFF = 80  # High-pass cutoff in Hz. Typical: 20–100 Hz (rumble/DC removal)
SILENCE_DB = 60  # Trim threshold: dB below peak. Typical: 40–80 dB


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class AudioPreprocessor:
    """
    Audio preprocessing pipeline:
      - Mono mix
      - Resample to target_sr
      - Trim leading/trailing silence
      - High-pass filter (rumble/DC removal)
      - LUFS normalization (via pyloudnorm or FFmpeg loudnorm)
      - Peak ceiling
      - File-level read/write with atomic replace
    """

    def __init__(
        self,
        target_sr: int = TARGET_SR,
        target_lufs: float = TARGET_LUFS,
        peak_ceiling_db: float = PEAK_CEILING,
        hpf_cutoff: float = HPF_CUTOFF,
        silence_db: float = SILENCE_DB,
        use_ffmpeg: bool = False,
    ):
        self.target_sr = target_sr
        self.target_lufs = target_lufs
        self.peak_ceiling = peak_ceiling_db
        self.hpf_cutoff = hpf_cutoff
        self.silence_db = silence_db
        self.use_ffmpeg = use_ffmpeg

        # reuse pyloudnorm meter if needed
        self._meter = pyln.Meter(self.target_sr)

    def _butter_highpass(self, x: np.ndarray, sr: int, order: int = 2) -> np.ndarray:
        nyq = 0.5 * sr
        b, a = signal.butter(order, self.hpf_cutoff / nyq, btype="high")
        return signal.lfilter(b, a, x)

    def _loudness_normalize_np(self, x: np.ndarray) -> np.ndarray:
        """
        Normalize to target LUFS via pyloudnorm in NumPy.
        """
        loudness = self._meter.integrated_loudness(x)
        logger.debug("Current LUFS=%.2f → Target LUFS=%.2f", loudness, self.target_lufs)
        return pyln.normalize.loudness(x, loudness, self.target_lufs)

    def _normalize_ffmpeg(self, src: Path, dst: Path) -> None:
        """
        Use ffmpeg to process
        """
        filter_chain = (
            f"silenceremove=start_periods=1:start_threshold=-{self.silence_db}dB:"
            f"stop_periods=1:stop_threshold=-{self.silence_db}dB,"
            f"pan=mono|c0=0.5*FL+0.5*FR,"
            f"highpass=f={self.hpf_cutoff},"
            f"loudnorm=I={self.target_lufs}:TP={self.peak_ceiling}:LRA=7,"
            #  → brick-wall limiter at –0.1 dB
            "alimiter=level=-0.1dB:attack=5:release=50"
        )
        (
            ffmpeg.input(str(src))
            .output(
                str(dst), ar=self.target_sr, ac=1, af=filter_chain, sample_fmt="s16"
            )
            .overwrite_output()
            .run(quiet=False)
        )

    def _peak_normalize(self, x: np.ndarray) -> np.ndarray:
        peak = np.max(np.abs(x))
        if peak == 0:
            return x
        ceiling_lin = 10 ** (self.peak_ceiling / 20)
        factor = min(1.0, ceiling_lin / peak)
        logger.debug("Peak=%.3f → factor=%.3f", peak, factor)
        return x * factor

    def preprocess_array(
        self, audio: np.ndarray, sr: int, modify_sr=False
    ) -> Tuple[np.ndarray, int]:
        """
        Apply all in-memory steps up to loudness/peak.
        (FFmpeg loudnorm and final write is in preprocess_file.)
        """
        # 1. Mono
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        # 2. Resample
        if sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sr)
            sr = self.target_sr

        # 3. Trim silence
        audio, _ = librosa.effects.trim(audio, top_db=self.silence_db)

        # 4. High-pass
        audio = audio - np.mean(audio)
        audio = self._butter_highpass(audio, sr)

        # 5. Loudness
        if not self.use_ffmpeg:
            audio = self._loudness_normalize_np(audio)

        # 6. Peak ceiling
        audio = self._peak_normalize(audio)

        return audio.astype(np.float32), sr

    def preprocess_file(self, src: Path, dst: Path) -> None:
        """
        Read src → process → write to dst atomically.
        """
        if self.use_ffmpeg:
            try:
                self._normalize_ffmpeg(src, dst)
            except Exception as e:
                logger.error("Could not read %s: %s", src, e)
                raise
        else:

            try:
                audio, sr = sf.read(str(src), always_2d=False)
            except Exception as e:
                logger.error("Could not read %s: %s", src, e)
                raise

            audio, sr = self.preprocess_array(audio, sr)

            # write into a temp WAV
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_name = tmp.name
            sf.write(tmp_name, audio, sr, subtype="PCM_16")
            logger.debug("Wrote intermediate file %s", tmp_name)
            # atomic replace into destination
            os.replace(tmp_name, str(dst))

        logger.info("Preprocessed %s → %s", src, dst)
