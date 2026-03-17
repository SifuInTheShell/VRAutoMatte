"""GPU and device detection utilities."""

import os

import torch
from loguru import logger


def configure_cuda_performance() -> None:
    """Apply CUDA performance flags for inference on NVIDIA GPUs.

    Safe to call before model loading. No-op on non-CUDA systems.

    Enables:
    - TF32 matmul (RTX 30xx+): full FP32 range, ~10-bit mantissa.
      Up to 3× faster matrix multiplications with negligible quality loss.
    - cuDNN benchmark: auto-selects the fastest conv kernel for fixed
      input shapes (pays off after the first frame).
    - cuDNN TF32: same TF32 optimisation for convolution ops.
    - Expandable VRAM segments: reduces fragmentation that causes OOM
      even when total allocation would fit within VRAM budget.
    """
    if not torch.cuda.is_available():
        return

    # TF32: faster matmul on Ampere / Ada / Blackwell (RTX 30xx/40xx/50xx)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Auto-tune cuDNN kernel selection for each unique input shape.
    # Fixed-resolution video = same shape every frame = consistent speedup.
    torch.backends.cudnn.benchmark = True

    # Reduce VRAM fragmentation that can cause OOM even with headroom.
    # Equivalent to PYTORCH_ALLOC_CONF=expandable_segments:True.
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )

    logger.debug(
        "CUDA performance flags set: TF32=on, cuDNN benchmark=on, "
        "expandable_segments=on"
    )


def get_device() -> torch.device:
    """Detect the best available compute device.

    Returns:
        torch.device for CUDA, MPS, or CPU.
    """
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"Using CUDA device: {name}")
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Using Apple MPS device")
        return torch.device("mps")
    logger.info("Using CPU (no GPU detected)")
    return torch.device("cpu")


def get_device_info() -> dict:
    """Return a dict with device details for display in the UI."""
    device = get_device()
    info = {"device": str(device), "name": "CPU"}
    if device.type == "cuda":
        info["name"] = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory
        info["vram_gb"] = round(mem / (1024 ** 3), 1)
    elif device.type == "mps":
        info["name"] = "Apple Silicon (MPS)"
    return info
