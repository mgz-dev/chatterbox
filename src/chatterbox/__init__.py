try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-tts")

from .common.sage_patch import apply_sage_patch
from .tts import ChatterboxTTS
from .vc import ChatterboxVC
