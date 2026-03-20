"""RVM ONNX Runtime processor — hardware-agnostic GPU inference.

Uses ONNX Runtime with DirectML (Windows), CoreML (macOS), CUDA,
or CPU execution providers. No PyTorch dependency at inference time.

Key advantages over the TorchScript backend:
- DirectML: works on any GPU vendor (NVIDIA, AMD, Intel) on Windows
  without vendor-specific drivers/SDKs beyond the standard GPU driver.
- Lighter runtime: onnxruntime-directml is ~50 MB vs ~2 GB for PyTorch.
- INT8 quantization path available for further speedup.

The official RVM release ships FP32 ONNX models for both mobilenetv3
and resnet50. The ONNX graph accepts recurrent hidden state as
explicit inputs/outputs (r1i–r4i / r1o–r4o) plus downsample_ratio.
"""

import urllib.request
from pathlib import Path

import numpy as np
from loguru import logger


# Official RVM ONNX model URLs from the v1.0.0 release.
_BASE = (
    "https://github.com/PeterL1n/RobustVideoMatting/"
    "releases/download/v1.0.0/"
)
RVM_ONNX_MODELS: dict[str, str] = {
    "mobilenetv3": _BASE + "rvm_mobilenetv3_fp32.onnx",
    "resnet50": _BASE + "rvm_resnet50_fp32.onnx",
}


def _model_dir() -> Path:
    """Return the directory where models are cached."""
    d = Path.home() / ".cache" / "vrautomatte" / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download_onnx_model(variant: str) -> Path:
    """Download RVM ONNX model if not already cached.

    Args:
        variant: 'mobilenetv3' or 'resnet50'.

    Returns:
        Path to the downloaded .onnx file.
    """
    if variant not in RVM_ONNX_MODELS:
        raise ValueError(
            f"Unknown RVM variant '{variant}'. "
            f"Use: {list(RVM_ONNX_MODELS)}"
        )

    filename = f"rvm_{variant}_fp32.onnx"
    model_path = _model_dir() / filename

    if model_path.exists():
        logger.debug(f"RVM ONNX model cached: {model_path}")
        return model_path

    url = RVM_ONNX_MODELS[variant]
    logger.info(f"Downloading RVM ONNX {variant} from {url}...")

    def _progress(count, block_size, total_size):
        pct = int(count * block_size * 100 / total_size)
        print(f"\r  Downloading: {pct}%", end="", flush=True)

    urllib.request.urlretrieve(
        url, str(model_path), reporthook=_progress
    )
    print()
    logger.info(f"ONNX model saved to {model_path}")
    return model_path


def _select_providers() -> list[str]:
    """Select the best available ONNX Runtime execution providers.

    Priority: DirectML > CUDA > CoreML > CPU.
    Returns a list that onnxruntime will try in order.
    Each GPU provider includes CPU as fallback so that session
    creation succeeds even if the GPU provider fails to initialise
    (e.g. no compatible hardware present despite the EP being compiled in).
    """
    import onnxruntime as ort

    available = ort.get_available_providers()
    logger.debug(f"ONNX Runtime providers available: {available}")

    selected: list[str] = []

    if "DmlExecutionProvider" in available:
        selected.append("DmlExecutionProvider")
    if "CUDAExecutionProvider" in available:
        selected.append("CUDAExecutionProvider")
    if "CoreMLExecutionProvider" in available:
        selected.append("CoreMLExecutionProvider")

    # Always include CPU as final fallback.
    selected.append("CPUExecutionProvider")

    return selected


def _empty_rec() -> list[np.ndarray]:
    """Create fresh recurrent state arrays (4 independent zeros)."""
    return [
        np.zeros([1, 1, 1, 1], dtype=np.float32)
        for _ in range(4)
    ]


class RVMOnnxProcessor:
    """Process video frames through RVM using ONNX Runtime.

    Manages recurrent state (r1–r4) for temporal consistency.
    Uses DirectML/CUDA/CoreML/CPU depending on available providers.

    Args:
        variant: 'mobilenetv3' or 'resnet50'.
        downsample_ratio: RVM downscale factor. 0.25 for HD,
            0.125 for 4K. Default 0.25.
    """

    def __init__(
        self,
        variant: str = "mobilenetv3",
        downsample_ratio: float = 0.25,
    ):
        import onnxruntime as ort

        model_path = _download_onnx_model(variant)
        providers = _select_providers()

        logger.info(
            f"Loading RVM ONNX {variant} with providers: "
            f"{providers}"
        )

        opts = ort.SessionOptions()
        opts.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        try:
            self._session = ort.InferenceSession(
                str(model_path),
                sess_options=opts,
                providers=providers,
            )
        except Exception as exc:
            # GPU provider may fail to initialise (no compatible
            # hardware, driver too old, etc.). Fall back to CPU.
            if providers != ["CPUExecutionProvider"]:
                logger.warning(
                    f"GPU provider failed ({exc}), "
                    f"falling back to CPU"
                )
                self._session = ort.InferenceSession(
                    str(model_path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
            else:
                raise

        active = self._session.get_providers()
        logger.info(f"ONNX Runtime active providers: {active}")

        self._downsample_ratio = np.array(
            [downsample_ratio], dtype=np.float32
        )
        self._rec = _empty_rec()
        self.variant = variant

    def reset(self) -> None:
        """Reset recurrent state (call between videos)."""
        self._rec = _empty_rec()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single RGB frame and return alpha matte.

        Args:
            frame: RGB (H, W, 3), uint8.

        Returns:
            Alpha matte (H, W), uint8 (0-255).
        """
        # HWC uint8 → CHW float32 [0, 1] with batch dim.
        src = (
            frame.astype(np.float32) / 255.0
        )
        src = np.transpose(src, (2, 0, 1))  # HWC → CHW
        src = np.expand_dims(src, 0)        # [1, 3, H, W]

        inputs = {
            "src": src,
            "r1i": self._rec[0],
            "r2i": self._rec[1],
            "r3i": self._rec[2],
            "r4i": self._rec[3],
            "downsample_ratio": self._downsample_ratio,
        }

        outputs = self._session.run(
            ["fgr", "pha", "r1o", "r2o", "r3o", "r4o"],
            inputs,
        )

        _fgr, pha, r1o, r2o, r3o, r4o = outputs
        self._rec = [r1o, r2o, r3o, r4o]

        # pha shape: [1, 1, H, W], float32 [0, 1].
        matte = (pha[0, 0] * 255).clip(0, 255).astype(np.uint8)
        return matte

    def cleanup(self) -> None:
        """Release ONNX session."""
        del self._session
        self._rec = _empty_rec()
        logger.debug("RVM ONNX processor cleaned up")
