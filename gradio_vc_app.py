import gradio as gr
from chatterbox.vc import ChatterboxVC
from chatterbox.common.torch_device import get_default_device


DEVICE = get_default_device()

print(f"Using device: {DEVICE}")

model = ChatterboxVC.from_pretrained(DEVICE)


def generate(audio, target_voice_path):
    wav = model.generate(
        audio, target_voice_path=target_voice_path,
    )
    return model.sr, wav.squeeze(0).numpy()


demo = gr.Interface(
    generate,
    [
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Input audio file"),
        gr.Audio(sources=["upload", "microphone"], type="filepath", label="Target voice audio file (if none, the default voice is used)", value=None),
    ],
    "audio",
)

if __name__ == "__main__":
    demo.launch()
