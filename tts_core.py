"""
Main processing functions and pipeline for TTS.
"""

import os
import subprocess
import tempfile
import random
from pathlib import Path
import logging
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np
import torch
import soundfile as sf
import nltk

from chatterbox.tts import ChatterboxTTS
from chatterbox.processors.textprocessor import TextPreprocessor
from chatterbox.processors.audioprocessor import AudioProcessor
from chatterbox.processors.asrprocessor import ASRVerifier
from chatterbox.common.torch_device import get_default_device

logger = logging.getLogger(__name__)


DEVICE = get_default_device()

print(f"Using device: {DEVICE}")


# One global text preprocessor; ASR get passed around, audio processor initiated elsewhere
TXT_PROC = TextPreprocessor(max_chunk_chars=400)


# Cache most‐recent whisper model
@lru_cache(maxsize=1)
def get_asr_verifier(model_name: str) -> ASRVerifier:
    return ASRVerifier(model_name=model_name, use_faster_whisper=True, device=DEVICE)


def _ensure_nltk():
    """Download NLTK punkt tokenizer if it's missing."""
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)


_ensure_nltk()


def _set_seed(seed: int) -> None:
    """Seed all RNGs for reproducible TTS + noise."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model() -> ChatterboxTTS:
    """Load or return an existing TTS model on DEVICE."""
    return ChatterboxTTS.from_pretrained(DEVICE)


def _preprocess_reference_audio(ap: AudioProcessor, path: Optional[str]) -> Optional[str]:
    """
    If user gave a reference audio, clean & normalize it to WAV.
    Returns path to the cleaned file (or original on failure).

    NOTE: Target silence ms is hard coded for the moment for reference audio.
    """
    if not path:
        return None

    src = Path(path)
    dst = src.with_name(src.stem + "_clean.wav")
    try:
        ap.process_file(src, dst, target_silence_ms=250, fade_ms=None)
        return str(dst)
    except Exception as e:
        logger.warning("Reference audio preprocessing failed: %s", e)
        return path


def _wav_to_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "192k") -> None:
    """
    Convert WAV -> MP3 via ffmpeg, silently fall back to WAV if it errors.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), "-vn", "-b:a", bitrate, str(mp3_path)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as e:
        logger.warning("ffmpeg conversion failed, keeping WAV: %s", e)


def _save_temp_wav(wav: np.ndarray, sr: int, prefix: str) -> Path:
    """Write a temp WAV and return its path."""
    fd, path = tempfile.mkstemp(suffix=".wav", prefix=prefix)
    os.close(fd)
    sf.write(path, wav, sr)
    return Path(path)


def _write_output_audio(ap: AudioProcessor, wav: np.ndarray, sr: int, output_format: str = "wav") -> Tuple[str, str]:
    """
    1. Write `wav` -> raw-temp.wav
    2. Run ap.process_file -> proc-temp.wav
    3. Optionally convert to MP3

    Return (final_path, raw_path)
    """
    raw_path = _save_temp_wav(wav, sr, prefix="cbx_raw_")
    proc_path = _save_temp_wav(wav, sr, prefix="cbx_proc_")
    ap.process_file(raw_path, proc_path, target_silence_ms=None, fade_ms=None)

    final_path = proc_path
    if output_format.lower() == "mp3":
        mp3_path = proc_path.with_suffix(".mp3")
        _wav_to_mp3(proc_path, mp3_path)
        try:
            proc_path.unlink()
        except OSError:
            pass
        final_path = mp3_path

    return str(final_path), str(raw_path)


def _verify_full_audio(verifier: ASRVerifier, wav: np.ndarray, sr: int, reference: str) -> Tuple[str, float]:
    """
    Run ASR QA on the full concatenated audio.
    
    Returns (transcript, score).
    """
    # Create a temp WAV file via mkstemp
    fd, temp_path = tempfile.mkstemp(suffix=".wav", prefix="cbx_asr_")
    os.close(fd)  # Close the OS-level handle immediately

    # Write out the numpy array
    sf.write(temp_path, wav, sr)

    try:
        res = verifier.verify(temp_path, reference)
        return res.transcription, res.score
    finally:
        # Clean up even if verification raises
        try:
            os.remove(temp_path)
        except OSError:
            pass


def _tts_chunks(
    model: ChatterboxTTS,
    ap: AudioProcessor,
    text: str,
    cleaned_prompt: Optional[str],
    asr_verifier: Optional[ASRVerifier],
    max_retry: int,
    *,
    exaggeration: float,
    temperature: float,
    cfg_weight: float,
    min_p: float,
    top_p: float,
    repetition_penalty: float,
) -> Tuple[np.ndarray, int]:
    """
    Chunk the text, synthesize each piece (with optional ASR retries),
    insert target_silence from audioprocessor between them, then concatenate.

    Returns (mono_float32_wav, sample_rate).
    """
    chunks = TXT_PROC.process(text)
    sr = model.sr
    # Not sure if it is worth generating scaled noise here, we can just fill in on final pass?
    silence_db_offset = 2*ap.silence_margin_db
    # Generate at 2x target silence to guaranteee processor will trim at final run
    target_silence_samples = 2*int(ap.target_silence_ms * sr / 1000)

    out_wavs = []
    for idx, chunk in enumerate(chunks, start=1):
        logger.info("Synthesizing chunk %d/%d", idx, len(chunks))
        logger.info("Chunk: %s", chunk)
        best_wav, best_score = None, -1.0

        for attempt in range(max_retry + 1):
            wav = (
                model.generate(
                    chunk,
                    audio_prompt_path=cleaned_prompt,
                    temperature=temperature,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    min_p=min_p,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

            if not asr_verifier:
                best_wav = wav
                break

            # ASR‐verify this chunk
            _, score = _verify_full_audio(asr_verifier, wav, sr, chunk)
            logger.info("Chunk %d attempt %d -> ASR score %.3f", idx, attempt, score)
            if score > best_score:
                best_score, best_wav = score, wav
            if asr_verifier.is_acceptable(score):
                break

        # noise_floor_db, _ = ap.energy_vad(best_wav, sr)
        # silence = ap._generate_comfort_noise(length=target_silence_samples, db_level=noise_floor_db - silence_db_offset)
        silence = np.zeros(target_silence_samples, dtype=np.float32)

        out_wavs.append(best_wav)
        if idx < len(chunks):
            out_wavs.append(silence)

    return np.concatenate(out_wavs), sr


def generate(
    model: Optional[ChatterboxTTS],
    ap: AudioProcessor,
    text: str,
    audio_prompt_path: Optional[str],
    *,
    exaggeration: float,
    temperature: float,
    seed_num: int,
    cfg_weight: float,
    min_p: float,
    top_p: float,
    repetition_penalty: float,
    whisper_model: str,
    max_retry: int,
    output_format: str,
) -> Tuple[Tuple[int, np.ndarray], Optional[str], str, str, str, float]:
    """
    Full TTS -> [optional ASR QA] -> normalization pipeline.

    Returns:
      - (sr, wav)            : raw concatenated audio (for in-browser preview)
      - cleaned_prompt_path  : cleaned reference-voice path (if any)
      - raw_output_path      : temp WAV before final normalization
      - final_path           : broadcast-normalized WAV/MP3 for download
      - transcript           : ASR transcript (if enabled)
      - score                : ASR WER-based score (1.0 = perfect)
    """
    # 1) load & seed
    if model is None:
        model = load_model()
    if seed_num:
        _set_seed(seed_num)

    # 2) preprocess reference audio
    cleaned_prompt = _preprocess_reference_audio(ap, audio_prompt_path)
    if cleaned_prompt:
        logger.info("Using cleaned prompt: %s", cleaned_prompt)

    # 3) optional per-chunk ASR QA
    asr_verifier = None if whisper_model == "none" else get_asr_verifier(whisper_model)

    # 4) synthesize all chunks + inter-chunk silence
    wav, sr = _tts_chunks(
        model=model,
        ap=ap,
        text=text,
        cleaned_prompt=cleaned_prompt,
        asr_verifier=asr_verifier,
        max_retry=max_retry,
        exaggeration=exaggeration,
        temperature=temperature,
        cfg_weight=cfg_weight,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    # 5) optional full-pass ASR QA
    transcript, score = ("", 0.0)
    if asr_verifier:
        transcript, score = _verify_full_audio(asr_verifier, wav, sr, text)

    # 6) final normalization & format conversion
    final_path, raw_path = _write_output_audio(ap, wav, sr, output_format)

    return (sr, wav), cleaned_prompt, raw_path, final_path, transcript, score
