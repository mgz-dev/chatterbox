"""
Enhanced gradio frontend for tts: text/audio preprocessing + optional ASR QA.

Text is cleaned, sentence-split and chunked before being fed to the TTS model.
User-supplied reference audio is normalised before cloning.
The generated audio is normalised to broadcast level before streaming.
If "Verify output with ASR" is ticked, we transcribe the result and display WER + full transcript.
"""

import os
import subprocess
import tempfile
import random
import uuid
from pathlib import Path
import logging
import numpy as np
import torch
import gradio as gr
import soundfile as sf
import nltk
from functools import lru_cache

from chatterbox.tts import ChatterboxTTS
from chatterbox.processors.textprocessor import TextPreprocessor
from chatterbox.processors.audioprocessor import AudioPreprocessor
from chatterbox.processors.asrprocessor import ASRVerifier

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

# Download NLTK 'punkt' tokenizer if needed
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
logger.setLevel(logging.DEBUG)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# One global instance per process
TXT_PROC = TextPreprocessor(max_chunk_chars=400)
AUD_PROC = AudioPreprocessor(target_sr=48_000, target_lufs=-16.0, peak_ceiling_db=-2.0, use_ffmpeg=False)


# Cache most recent whisper model
@lru_cache(maxsize=1)
def get_asr_verifier(model_name: str):
    return ASRVerifier(model_name=model_name, use_faster_whisper=True, device=DEVICE)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model() -> ChatterboxTTS:
    return ChatterboxTTS.from_pretrained(DEVICE)


def _preprocess_reference_audio(path: str | None) -> str | None:
    if not path:
        return None
    src = Path(path)
    dst = src.with_name(src.stem + "_clean.wav")
    try:
        AUD_PROC.preprocess_file(src, dst)
        return str(dst)
    except Exception as e:
        # Fall back silently - clone with raw file
        logger.warning(f"reference audio preprocessing failed: {e}\n")
        return path


def _wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> None:
    """
    Convert a WAV file to MP3 using ffmpeg.
    Falls back silently to the WAV if ffmpeg is missing/failed.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-vn", "-b:a", bitrate, mp3_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except Exception as e:
        logger.warning("ffmpeg conversion failed → keeping WAV: %s", e)


def _write_output_audio(wav: np.ndarray, sr: int, output_format: str = "wav") -> str:
    """
    • Writes a temporary WAV (always - needed for ASR and as a ffmpeg source)
    • Optionally transcodes to MP3
    • Returns *the* path that should be sent back to Gradio
    """
    base = Path(tempfile.gettempdir()) / f"cbx_{uuid.uuid4().hex}"
    wav_path = str(base.with_suffix(".wav"))
    sf.write(wav_path, wav, sr)

    if output_format.lower() == "mp3":
        mp3_path = str(base.with_suffix(".mp3"))
        _wav_to_mp3(wav_path, mp3_path)
        return mp3_path

    return wav_path


def _tts_chunks(
    model: ChatterboxTTS,
    text: str,
    ref_path: str | None,
    asr_verifier: ASRVerifier | None,
    max_retry: int,
    **params,
) -> tuple[np.ndarray, int]:
    """
    Run the full TTS pipeline, chunk-by-chunk, and concatenate.
    Returns (mono_float32_wav, sample_rate)

    If `asr_verifier` is supplied, each chunk is re-synthesised up to
    `max_retry` times until the WER score is > verifier.threshold.
    The best attempt (highest score) is kept.  # TODO: pass in threshold to UI
    Returns the concatenated waveform and its sample-rate.
    """
    chunks = TXT_PROC.process(text)
    sr = model.sr
    silence = np.zeros(int(0.2 * sr))  # 200ms  # TODO: pass in as setting. Maybe add optional syntax into text for longer pauses.
    out_wavs: list[np.ndarray] = []

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"chunk {i}:\n{chunk}\n")
        best_wav, best_score = None, -1.0

        for attempt in range(max_retry + 1):
            wav = (
                model.generate(
                    chunk,
                    audio_prompt_path=ref_path,
                    temperature=params["temperature"],
                    exaggeration=params["exaggeration"],
                    cfg_weight=params["cfgw"],
                    min_p=params["min_p"],
                    top_p=params["top_p"],
                    repetition_penalty=params["repetition_penalty"],
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )

            if asr_verifier is None:  # QA disabled
                best_wav = wav
                break

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                temp_wav_path = tf.name  # get name before closing
            sf.write(temp_wav_path, wav, sr)  # now file is closed, safe to write

            try:
                res = asr_verifier.verify(temp_wav_path, chunk)
            finally:
                os.remove(temp_wav_path)  # clean up temp file

            logger.debug("chunk %d try %d - score %.3f", i, attempt, res.score)

            # keep best so far
            if res.score > best_score:
                best_score, best_wav = res.score, wav
            if asr_verifier.is_acceptable(res.score):
                break

        out_wavs.append(best_wav)
        if i != len(chunks):
            out_wavs.append(silence)

    merged = np.concatenate(out_wavs)
    return merged, sr


def generate(
    model: ChatterboxTTS | None,
    text: str,
    audio_prompt_path: str | None,
    exaggeration: float,
    temperature: float,
    seed_num: int,
    cfgw: float,
    min_p: float,
    top_p: float,
    repetition_penalty: float,
    whisper_model: str,
    max_retry: int,
    output_format: str,
) -> tuple[tuple[int, np.ndarray], str, str, float]:
    if model is None:
        model = load_model()
    if seed_num:
        _set_seed(int(seed_num))

    ref_path = _preprocess_reference_audio(audio_prompt_path)
    logger.debug(f"Cleaned ref audio path: {ref_path}\n")

    asr_verifier = None if whisper_model == "none" else get_asr_verifier(whisper_model)

    wav, sr = _tts_chunks(
        model,
        text,
        ref_path,
        asr_verifier,
        max_retry=max_retry,
        exaggeration=exaggeration,
        temperature=temperature,
        cfgw=cfgw,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
    )

    tmp_wav_path = Path(tempfile.gettempdir()) / f"cbx_full_{uuid.uuid4().hex}.wav"
    logger.debug(f"temp wave audio path: {tmp_wav_path}\n")
    sf.write(tmp_wav_path, wav, sr)

    # full-pass ASR QA (optional) on the merged result
    transcript, score = "", 0.0
    if asr_verifier:
        res = asr_verifier.verify(str(tmp_wav_path), text)
        transcript, score = res.transcription, res.score

    # finally write the user-visible file (wav or  mp3)
    final_path = _write_output_audio(wav, sr, output_format)
    return (sr, wav), final_path, transcript, score


# GRADIO UI
with gr.Blocks(title="ChatterboxTTS Modified") as demo:
    model_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            text_in = gr.Textbox(
                label="Text (any length - will be auto-chunked)",
                value="""All that is gold does not glitter,
Not all those who wander are lost;
The old that is strong does not wither,
Deep roots are not reached by the frost.

From the ashes a fire shall be woken,
A light from the shadows shall spring;
Renewed shall be blade that was broken,
The crownless again shall be king.""",
                lines=6,
            )

            ref_wav = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="Reference voice (optional)",
            )

            exaggeration = gr.Slider(0.25, 2, step=0.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=0.5)
            cfg_weight = gr.Slider(0.0, 1, step=0.05, label="CFG / pace", value=0.5)
            temperature = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)
            whisper_model = gr.Dropdown(
                choices=["none", "tiny", "base", "small", "medium", "large"], value="none", label="Whisper model (for ASR QA)"
            )
            output_format = gr.Radio(["wav", "mp3"], value="wav", label="Output format")

            with gr.Accordion("More options", open=False):
                max_retry = gr.Slider(0, 5, step=1, value=2, label="Max retries per chunk for ASR")
                seed_num = gr.Number(value=0, label="Random seed (0 = random)")
                min_p = gr.Slider(
                    0.00,
                    1.00,
                    step=0.01,
                    label="min_p || Newer Sampler. Recommend 0.02 > 0.1. Handles Higher Temperatures better. 0.00 Disables",
                    value=0.05,
                )
                top_p = gr.Slider(0.00, 1.00, step=0.01, label="top_p || Original Sampler. 1.0 Disables(recommended). Original 0.8", value=1.00)
                rep_penalty = gr.Slider(1.0, 2.0, step=0.1, label="Repetition penalty", value=1.2)

            run_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=1):
            # audio_out = gr.Audio(label="Synthesised speech")
            audio_out = gr.State()  # placeholder in case we want to revert to this again
            path_out = gr.Audio(label="Synthesised speech - Converted File", type="filepath")
            transcript_out = gr.Textbox(label="ASR transcript (auto-evaluated)", lines=4)
            score_out = gr.Number(label="Score (1 = perfect)", precision=3)

    demo.load(load_model, inputs=[], outputs=model_state)

    run_btn.click(
        fn=generate,
        inputs=[
            model_state,
            text_in,
            ref_wav,
            exaggeration,
            temperature,
            seed_num,
            cfg_weight,
            min_p,
            top_p,
            rep_penalty,
            whisper_model,
            max_retry,
            output_format,
        ],
        outputs=[audio_out, path_out, transcript_out, score_out],
    )

if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=1).launch()
