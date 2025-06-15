import torch
from functools import lru_cache


@lru_cache(maxsize=1)
def get_default_device() -> torch.device:
    """
    Return the best available torch.device in the order:
        cuda -> mps -> cpu
    The result is cached so the hardware probes run only once.
    """
    if torch.cuda.is_available():
        # Use the first visible GPU; override via CUDA_VISIBLE_DEVICES
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Monkey-patch torch.load


def patch_torch_load(device: torch.device | None = None) -> None:
    """
    Replace torch.load with a version that defaults to map_location=<device>.
    The original torch.load is kept as _torch_load_original.
    """
    orig_load = torch.load
    target = device or get_default_device()

    def _patched_torch_load(*args, **kwargs):
        # Respect an explicit map_location from the caller.
        kwargs.setdefault("map_location", target)
        return orig_load(*args, **kwargs)

    # Expose the original just in case.
    torch._torch_load_original = orig_load  # type: ignore[attr-defined]
    torch.load = _patched_torch_load


# Activate immediately
patch_torch_load()
