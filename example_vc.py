import torchaudio as ta
from chatterbox.vc import ChatterboxVC
from chatterbox.common.torch_device import get_default_device


DEVICE = get_default_device()

print(f"Using device: {DEVICE}")

AUDIO_PATH = "YOUR_FILE.wav"
TARGET_VOICE_PATH = "YOUR_FILE.wav"

model = ChatterboxVC.from_pretrained(DEVICE)
wav = model.generate(
    audio=AUDIO_PATH,
    target_voice_path=TARGET_VOICE_PATH,
)
ta.save("testvc.wav", wav, model.sr)
