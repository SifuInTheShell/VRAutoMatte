"""GPU and device detection utilities."""

import torch
from loguru import logger


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
