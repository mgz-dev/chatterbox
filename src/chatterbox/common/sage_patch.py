"""
Drop-in SageAttention patcher.

* Set the variant with SAGE_ATTENTION=auto|int8-fp16-triton|int8-fp16-cuda|int8-fp8-cuda|disabled
Must be imported **before** any `transformers` model is created.
"""
import logging
import os
import inspect
from collections import Counter
import torch
import torch.nn.functional as F

__all__ = ["apply_sage_patch", "restore_sdpa", "_sdpa_cast_safe", "_sage_patched", "report_sage_fallbacks"]

_VARIANT = os.getenv("SAGE_ATTENTION", "auto").lower()


_sage_patched = False
_orig_sdpa = F.scaled_dot_product_attention


logger = logging.getLogger(__name__)
_fallback_counter = Counter()


def _select_kernel(variant: str):
    from sageattention import (
        sageattn,
        sageattn_qk_int8_pv_fp16_cuda,
        sageattn_qk_int8_pv_fp16_triton,
        sageattn_qk_int8_pv_fp8_cuda,
    )

    def _kernel_int8_fp16_cuda(q, k, v, **kw):
        return sageattn_qk_int8_pv_fp16_cuda(q, k, v, pv_accum_dtype="fp32", **kw)

    def _kernel_int8_fp8_cuda(q, k, v, **kw):
        return sageattn_qk_int8_pv_fp8_cuda(q, k, v, pv_accum_dtype="fp32+fp32", **kw)

    if variant in ("", "auto", "sageattn"):
        return sageattn
    if variant == "int8-fp16-cuda":
        return _kernel_int8_fp16_cuda
    if variant == "int8-fp16-triton":
        return sageattn_qk_int8_pv_fp16_triton
    if variant == "int8-fp8-cuda":
        return _kernel_int8_fp8_cuda
    raise ValueError(f"Unknown SageAttention variant '{variant}'")


def _sdpa_cast_safe(q, k, v, *args, **kwargs):
    """
    Wrapper around the real Sage kernel that:

    1. Down-casts fp32 -> fp16/bf16 when needed.
    2. Falls back to Torch SDPA if Sage kernel doesn't support the given head dimension or otherwise errors.

    This keeps the whole pipeline robust while still accelerating every compatible call.
    """
    orig_dtype = q.dtype
    need_cast = orig_dtype not in (torch.float16, torch.bfloat16)
    if need_cast:
        q, k, v = q.half(), k.half(), v.half()

    try:
        out = F._sage_kernel(q, k, v, *args, **kwargs)

    except (ValueError, AssertionError) as e:

        # Figure out who called us:

        frame = inspect.currentframe().f_back
        location = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        del frame
        head_dim = q.shape[-1]
        _fallback_counter[(head_dim, location)] += 1

        fallback_count = _fallback_counter[(head_dim, location)]
        if fallback_count <= 10:
            shape = q.shape  # e.g. torch.Size([B, H, S, D])
            # log full shape instead of unpacking
            logger.warning(
                f"[sage_patch] fallback @ {location} dtype {orig_dtype} "
                f"(shape={tuple(shape)}, head_dim={head_dim}) -> {e}"
                )
        # cast back & call true SDPA
        q0, k0, v0 = q.to(orig_dtype), k.to(orig_dtype), v.to(orig_dtype)
        return _orig_sdpa(q0, k0, v0, *args, **kwargs)

    if need_cast:
        out = out.to(orig_dtype)
    return out


def apply_sage_patch():
    """
    Monkey-patch torch.nn.functional.scaled_dot_product_attention with the chosen SageAttention kernel.
    * Variant comes from env var SAGE_ATTENTION (auto | int8-fp16-cuda | … | disabled)
    Call exactly once BEFORE any transformers models are built.
    """
    global _sage_patched
    if _sage_patched or _VARIANT == "disabled":
        return

    try:
        # 1. choose the real Sage kernel and stash it as F._sage_kernel
        F._sage_kernel = _select_kernel(_VARIANT)

        # 2. wrap it so fp32 callers are automatically served via fp16 cast
        F.scaled_dot_product_attention = _sdpa_cast_safe
        _sage_patched = True
        logger.info(f"SageAttention '{_VARIANT}' activated (fp32 callers auto-cast).")

    except ImportError as e:
        logger.warning(f"SageAttention import failed: {e}")
        return
    except ValueError as e:
        logger.error(f"Bad SAGE_ATTENTION='{_VARIANT}': {e}")
        return
    except Exception as exc:
        logger.warning(f"SageAttention fallback: {exc}")
        return  # keep default SDPA


def report_sage_fallbacks(top_k: int = 10):
    """
    Print the top_k most common SageAttention fallback sites,
    along with how often they occurred.
    """
    for (head_dim, location), count in _fallback_counter.most_common(top_k):
        logger.info(f"{count:5d} x fallback at {location} (head_dim={head_dim})")


def restore_sdpa():
    """Undo the patch (mainly for debugging)."""
    global _sage_patched
    if _sage_patched:
        F.scaled_dot_product_attention = _orig_sdpa
        _sage_patched = False
        delattr(F, "_sage_kernel")
        logger.info("[sage_patch] SDPA restored.")


# Automatically patch on import – *before* transformers are imported elsewhere.
apply_sage_patch()
