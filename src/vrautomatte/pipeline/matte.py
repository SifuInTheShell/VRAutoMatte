"""AI matte generation using Robust Video Matting (RVM).

RVM is a recurrent neural network that processes videos with temporal
memory, producing temporally consistent alpha mattes without any
auxiliary input (no trimap, no background image, no first-frame mask).
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image
from torchvision.transforms.functional import to_tensor

from vrautomatte.utils.gpu import get_device

# Model download URLs
RVM_MODELS = {
    "mobilenetv3": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/"
        "download/v1.0.0/rvm_mobilenetv3.pth"
    ),
    "resnet50": (
        "https://github.com/PeterL1n/RobustVideoMatting/releases/"
        "download/v1.0.0/rvm_resnet50.pth"
    ),
}


def _model_dir() -> Path:
    """Return the directory where models are cached."""
    d = Path.home() / ".cache" / "vrautomatte" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_model(variant: str = "mobilenetv3") -> Path:
    """Download RVM model weights if not already cached.

    Args:
        variant: 'mobilenetv3' (fast, recommended) or 'resnet50' (higher quality).

    Returns:
        Path to the downloaded model file.
    """
    if variant not in RVM_MODELS:
        raise ValueError(f"Unknown variant '{variant}'. Use: {list(RVM_MODELS)}")

    filename = f"rvm_{variant}.pth"
    model_path = _model_dir() / filename

    if model_path.exists():
        logger.debug(f"RVM model already cached: {model_path}")
        return model_path

    url = RVM_MODELS[variant]
    logger.info(f"Downloading RVM {variant} model from {url}...")

    def _progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        print(f"\r  Downloading: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(model_path), reporthook=_progress)
    print()  # newline after progress
    logger.info(f"Model saved to {model_path}")
    return model_path


def load_model(variant: str = "mobilenetv3",
               device: torch.device | None = None):
    """Load a pretrained RVM model.

    Args:
        variant: 'mobilenetv3' or 'resnet50'.
        device: Target device. Auto-detected if None.

    Returns:
        Loaded TorchScript model on the target device.
    """
    if device is None:
        device = get_device()

    model_path = download_model(variant)
    logger.info(f"Loading RVM {variant} on {device}...")
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()
    return model, device


class RVMProcessor:
    """Process video frames through RVM to generate alpha mattes.

    This class manages the recurrent state across frames so that
    temporal consistency is maintained.

    Args:
        variant: 'mobilenetv3' or 'resnet50'.
        downsample_ratio: Processing downscale (0.25 = 1/4 res for speed).
    """

    def __init__(self, variant: str = "mobilenetv3",
                 downsample_ratio: float = 0.25):
        self.model, self.device = load_model(variant)
        self.downsample_ratio = downsample_ratio
        self.rec = [None] * 4  # recurrent hidden states
        self.variant = variant

    def reset(self) -> None:
        """Reset recurrent state (call between videos)."""
        self.rec = [None] * 4

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return the alpha matte.

        Args:
            frame: RGB frame as numpy array (H, W, 3), uint8.

        Returns:
            Alpha matte as numpy array (H, W), uint8 (0-255).
        """
        img = Image.fromarray(frame).convert("RGB")
        src = to_tensor(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            fgr, pha, *self.rec = self.model(
                src, *self.rec, self.downsample_ratio
            )

        # pha is (1, 1, H, W) float in [0, 1]
        matte = (pha[0, 0] * 255).byte().cpu().numpy()
        return matte

    def process_frame_pil(self, frame: Image.Image) -> Image.Image:
        """Process a PIL Image and return the matte as PIL Image.

        Args:
            frame: RGB PIL Image.

        Returns:
            Grayscale PIL Image of the alpha matte.
        """
        arr = np.array(frame.convert("RGB"))
        matte = self.process_frame(arr)
        return Image.fromarray(matte, mode="L")
