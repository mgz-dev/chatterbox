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

from tts_core import load_model, generate


os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
)


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
