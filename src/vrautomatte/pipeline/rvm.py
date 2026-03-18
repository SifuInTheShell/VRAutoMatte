"""RVM (Robust Video Matting) processor.

Handles model downloading, loading, and per-frame inference
with recurrent state for temporal consistency.

Performance notes
-----------------
- ``use_fp16=True`` downloads and runs the official FP16 TorchScript
  weights instead of FP32. RTX 3090 benchmark: 172 FPS vs 104 FPS at
  HD; ~50% VRAM reduction. Automatically disabled on non-CUDA devices.
- ``torch.jit.freeze()`` is applied after load — officially recommended
  by the RVM authors. Triggers graph-level BatchNorm fusion and other
  TorchScript optimisations at no quality cost.
- ``configure_cuda_performance()`` sets TF32 + cuDNN benchmark before
  any tensor work runs.
- The per-frame tensor conversion uses ``torch.from_numpy`` directly
  instead of routing through PIL, eliminating a redundant image
  allocation on every frame.
- ``downsample_ratio``: the RVM authors document 0.25 for HD (1080p)
  and 0.125 for 4K. For 8K SBS content (4K per eye) the default here
  is 0.125. Adjust lower (e.g. 0.1) if you see OOM or slowdown.
"""

import urllib.request
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from vrautomatte.utils.gpu import configure_cuda_performance, get_device

# Official RVM release URLs — both FP32 and FP16 TorchScript variants.
_BASE = (
    "https://github.com/PeterL1n/RobustVideoMatting/"
    "releases/download/v1.0.0/"
)
RVM_MODELS: dict[str, dict[str, str]] = {
    "mobilenetv3": {
        "fp32": _BASE + "rvm_mobilenetv3_fp32.torchscript",
        "fp16": _BASE + "rvm_mobilenetv3_fp16.torchscript",
    },
    "resnet50": {
        "fp32": _BASE + "rvm_resnet50_fp32.torchscript",
        "fp16": _BASE + "rvm_resnet50_fp16.torchscript",
    },
}


def _model_dir() -> Path:
    """Return the directory where models are cached."""
    d = Path.home() / ".cache" / "vrautomatte" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def download_model(
    variant: str = "mobilenetv3",
    use_fp16: bool = True,
) -> Path:
    """Download RVM model weights if not already cached.

    Args:
        variant: ``'mobilenetv3'`` or ``'resnet50'``.
        use_fp16: Download the FP16 TorchScript weights. Gives ~50%
            VRAM reduction and higher throughput on Turing/Ampere/Ada/
            Blackwell GPUs. Default True.

    Returns:
        Path to the downloaded model file.
    """
    if variant not in RVM_MODELS:
        raise ValueError(
            f"Unknown RVM variant '{variant}'. "
            f"Use: {list(RVM_MODELS)}"
        )

    precision = "fp16" if use_fp16 else "fp32"
    filename = f"rvm_{variant}_{precision}.torchscript"
    model_path = _model_dir() / filename

    # Clean up old .pth files (wrong format from earlier versions).
    old_pth = _model_dir() / f"rvm_{variant}.pth"
    if old_pth.exists():
        logger.info(f"Removing incompatible .pth file: {old_pth}")
        old_pth.unlink()

    if model_path.exists():
        logger.debug(f"RVM model cached: {model_path}")
        return model_path

    url = RVM_MODELS[variant][precision]
    logger.info(f"Downloading RVM {variant} ({precision}) from {url}...")

    def _progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        print(f"\r  Downloading: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(url, str(model_path), reporthook=_progress)
    print()
    logger.info(f"Model saved to {model_path}")
    return model_path


class RVMProcessor:
    """Process video frames through RVM for alpha mattes.

    Manages recurrent state for temporal consistency across frames.

    Args:
        variant: ``'mobilenetv3'`` or ``'resnet50'``.
        downsample_ratio: Processing downscale factor. RVM authors
            recommend 0.25 for HD (1080p) and 0.125 for 4K. Default
            0.125, appropriate for 4K-per-eye 8K SBS content.
        device: Target device. Auto-detected if None.
        use_fp16: Use the official FP16 TorchScript weights. Halves
            VRAM, increases throughput on RTX GPUs. Automatically
            falls back to FP32 on non-CUDA devices. Default True.
    """

    def __init__(
        self,
        variant: str = "mobilenetv3",
        downsample_ratio: float = 0.125,
        device: torch.device | None = None,
        *,
        use_fp16: bool = True,
    ):
        if device is None:
            device = get_device()

        # FP16 on CUDA and MPS. CPU requires FP32.
        self._use_fp16 = use_fp16 and device.type in ("cuda", "mps")

        # Apply TF32 + cuDNN benchmark flags before model load.
        configure_cuda_performance()

        model_path = download_model(variant, use_fp16=self._use_fp16)
        logger.info(
            f"Loading RVM {variant} "
            f"({'fp16' if self._use_fp16 else 'fp32'}) on {device}..."
        )

        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()

        # torch.jit.freeze() triggers graph-level optimisations:
        # BatchNorm fusion, constant folding, dead-code elimination.
        # Officially recommended by RVM authors for TorchScript inference.
        # Must be called after .eval() and before any forward pass.
        if device.type == "cuda":
            model = torch.jit.freeze(model)
            logger.debug("RVM: TorchScript graph frozen (BN fusion applied)")

        self.model = model
        self.device = device
        self.downsample_ratio = downsample_ratio
        self.rec = [None] * 4
        self.variant = variant

    def reset(self) -> None:
        """Reset recurrent state (call between videos)."""
        self.rec = [None] * 4

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return alpha matte.

        Args:
            frame: RGB (H, W, 3), uint8.

        Returns:
            Alpha matte (H, W), uint8 (0-255).
        """
        # Direct numpy → tensor conversion, bypassing the PIL round-trip
        # (Image.fromarray → to_tensor) that was in the previous version.
        # frame is already uint8 RGB; we normalise to [0, 1] float in one op.
        src = (
            torch.from_numpy(frame)
            .permute(2, 0, 1)          # HWC → CHW
            .unsqueeze(0)              # add batch dim → [1, C, H, W]
            .to(dtype=torch.float16 if self._use_fp16 else torch.float32,
                device=self.device)
            .div(255.0)
        )

        with torch.no_grad():
            fgr, pha, *self.rec = self.model(
                src, *self.rec, self.downsample_ratio
            )

        return (pha[0, 0] * 255).byte().cpu().numpy()

    def cleanup(self) -> None:
        """Release model and GPU memory."""
        del self.model
        self.rec = [None] * 4
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("RVM processor cleaned up")

    def process_frame_pil(self, frame):
        """Process a PIL Image and return the matte.

        Args:
            frame: RGB PIL Image.

        Returns:
            Grayscale PIL Image of the alpha matte.
        """
        from PIL import Image
        arr = np.array(frame.convert("RGB"))
        matte = self.process_frame(arr)
        return Image.fromarray(matte, mode="L")
