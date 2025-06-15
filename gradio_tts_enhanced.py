"""
Enhanced gradio frontend for tts: text/audio preprocessing + optional ASR QA.

Text is cleaned, sentence-split and chunked before being fed to the TTS model.
User-supplied reference audio is normalised before cloning.
The generated audio is normalised to broadcast level before streaming.
If "Verify output with ASR" is ticked, we transcribe the result and display WER + full transcript.
"""

import os
import gradio as gr
import logging
from typing import Optional

from tts_core import load_model, generate
from chatterbox.processors.audioprocessor import AudioProcessor


os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
# os.environ["HF_HUB_OFFLINE"] = "1"  # remove comment to not request HEAD from HF

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)

AUD_PROC = AudioProcessor()


# GENERATE WRAPPER FOR AUDIO PROCESSOR
def generate_with_audio_processor(
    model,
    text: str,
    reference_audio_path: Optional[str],
    exaggeration: float,
    temperature: float,
    seed: int,
    cfg_weight: float,
    min_p: float,
    top_p: float,
    repetition_penalty: float,
    whisper_model: str,
    max_retry: int,
    output_format: str,
    # AudioProcessor settings:
    target_sr: int,
    target_lufs: float,
    peak_ceiling_db: float,
    hpf_cutoff: int,
    butter_order: int,
    target_silence_ms: int,
    fade_ms: int,
    silence_margin_db: float,
    use_memory_io: bool,
):
    """
    Update the shared AudioProcessor instance and return to gradio
    """
    # 1) re-configure audio processor
    ap = AUD_PROC

    ap.update_config(
        target_sr=target_sr,
        target_lufs=target_lufs,
        peak_ceiling_db=peak_ceiling_db,
        hpf_cutoff=hpf_cutoff,
        butter_order=butter_order,
        target_silence_ms=target_silence_ms,
        fade_ms=fade_ms,
        silence_margin_db=silence_margin_db,
        seed=seed,
        use_memory_io=use_memory_io,
    )

    # 2) invoke core generate; it returns 6 items
    full_outputs = generate(
        model,
        ap,
        text,
        reference_audio_path,
        exaggeration=exaggeration,
        temperature=temperature,
        seed_num=seed,
        cfg_weight=cfg_weight,
        min_p=min_p,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        whisper_model=whisper_model,
        max_retry=max_retry,
        output_format=output_format,
    )

    _, cleaned_prompt, raw_path, final_path, transcript, score = full_outputs
    return cleaned_prompt, raw_path, final_path, transcript, score


# GRADIO INTERFACE
with gr.Blocks(title="ChatterboxTTS Modified") as demo:
    model_state = gr.State(None)

    with gr.Row():
        with gr.Column(scale=1):
            # ---- CORE TTS INPUTS ----
            text_in = gr.Textbox(
                label="Text (any length - will be auto-chunked)",
                lines=6,
                value=(
                    "All that is gold does not glitter,\n"
                    "Not all those who wander are lost;\n"
                    "The old that is strong does not wither,\n"
                    "Deep roots are not reached by the frost.\n\n"
                    "From the ashes a fire shall be woken,\n"
                    "A light from the shadows shall spring;\n"
                    "Renewed shall be blade that was broken,\n"
                    "The crownless again shall be king."
                ),
            )
            ref_wav = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference voice (optional)", value=None)
            exaggeration = gr.Slider(0.25, 2, step=0.05, label="Exaggeration (Neutral = 0.5, extreme values can be unstable)", value=0.5)
            cfg_weight = gr.Slider(0.0, 1, step=0.05, label="CFG / pace", value=0.5)
            temperature = gr.Slider(0.05, 5, step=0.05, label="Temperature", value=0.8)
            whisper_model = gr.Dropdown(choices=["none", "tiny", "base", "small", "medium", "large"], value="none", label="Whisper model")
            output_format = gr.Radio(["wav", "mp3"], value="wav", label="Output format")

            with gr.Accordion("More options", open=False):
                max_retries = gr.Slider(0, 6, step=1, value=2, label="Max retries per chunk (ASR QA)")
                seed = gr.Number(value=0, label="Global random seed (0 = random)")
                min_p = gr.Slider(0.00, 1.00, step=0.01, value=0.05, label="min_p (n-sampler) | Recommend 0.02 > 0.1")
                top_p = gr.Slider(0.00, 1.00, step=0.01, value=1.00, label="top_p (original sampler) | 1 disables")
                rep_penalty = gr.Slider(1.0, 2.0, step=0.1, value=1.2, label="Repetition penalty")

            # ---- AudioProcessor Settings ----
            with gr.Accordion("Audio Processor Settings", open=False):
                sr = gr.Slider(8_000, 48_000, step=1_000, value=AUD_PROC.target_sr, label="Target sample rate")
                lufs = gr.Slider(-40.0, -10, step=0.5, value=AUD_PROC.target_lufs, label="Target LUFS")
                peak_ceiling = gr.Slider(-14.0, -1, step=1.0, value=AUD_PROC.peak_ceiling, label="Peak ceiling (dB)")
                hpf_cutoff = gr.Slider(0, 100, step=10, value=AUD_PROC.hpf_cutoff, label="HPF cutoff (Hz)")
                butter_order = gr.Slider(1, 10, step=1, value=AUD_PROC.butter_order, label="Butterworth order")
                target_silence_ms = gr.Slider(0, 2_000, step=50, value=AUD_PROC.target_silence_ms, label="Target silence (ms)")
                fade_ms = gr.Slider(0, 100, step=5, value=AUD_PROC.fade_ms, label="Fade duration (ms)")
                silence_margin = gr.Slider(0.0, 10.0, step=0.5, value=AUD_PROC.silence_margin_db, label="Silence margin (dB)")
                use_memory_io_cb = gr.Checkbox(
                    value=AUD_PROC.use_memory_io,
                    label="Use in-memory I/O",
                )

            # ---- Run button ----
            run_btn = gr.Button("Generate", variant="primary")

        # ---- OUTPUTS ----
        with gr.Column(scale=1):
            cleaned_audio_prompt_path = gr.Audio(label="Cleaned audio prompt", type="filepath")
            raw_out = gr.Audio(label="Raw TTS output (pre-normalization)", type="filepath")
            final_out = gr.Audio(label="Cleaned Output (Converted File)", type="filepath")
            transcript_out = gr.Textbox(label="ASR transcript (auto-evaluated)", lines=6)
            score_out = gr.Number(label="Score (1 = perfect)", precision=3)

    # Load the TTS model
    demo.load(load_model, inputs=[], outputs=model_state)

    # Wire up the Generate button
    run_btn.click(
        fn=generate_with_audio_processor,
        inputs=[
            model_state,
            text_in,
            ref_wav,
            exaggeration,
            temperature,
            seed,
            cfg_weight,
            min_p,
            top_p,
            rep_penalty,
            whisper_model,
            max_retries,
            output_format,
            # AudioProcessor settings:
            sr,
            lufs,
            peak_ceiling,
            hpf_cutoff,
            butter_order,
            target_silence_ms,
            fade_ms,
            silence_margin,
            use_memory_io_cb,
        ],
        outputs=[
            cleaned_audio_prompt_path,
            raw_out,
            final_out,
            transcript_out,
            score_out,
        ],
    )

if __name__ == "__main__":
    demo.queue(max_size=50, default_concurrency_limit=1).launch()
