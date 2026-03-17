"""MatAnyone 2 matting processor and SAM2 mask generation.

MatAnyone 2 uses a memory-based approach that requires a first-frame
segmentation mask. SAM2 auto-generates this mask from the first frame.

GPU memory strategy: SAM2 loads, generates a mask from frame 1,
then unloads before MatAnyone 2 loads (D003).
"""

from pathlib import Path

import numpy as np
import torch
from loguru import logger
from PIL import Image

from vrautomatte.utils.gpu import get_device

# SAM2 model variants by device capability
_SAM2_VARIANTS = {
    "cpu": "facebook/sam2-hiera-tiny",
    "cuda": "facebook/sam2-hiera-small",
    "mps": "facebook/sam2-hiera-small",
}


def generate_first_frame_mask(
    frame: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Auto-generate a person segmentation mask from frame 1.

    Uses SAM2's automatic mask generator to find the largest
    person-shaped region in the frame.

    Args:
        frame: RGB array (H, W, 3), uint8.
        device: Target device. Auto-detected if None.

    Returns:
        Binary mask (H, W), uint8 (0 or 255).

    Raises:
        ImportError: If sam2 package is not installed.
        RuntimeError: If no suitable mask found.
    """
    try:
        from sam2.sam2_image_predictor import (
            SAM2ImagePredictor,
        )
        from sam2.automatic_mask_generator import (
            SAM2AutomaticMaskGenerator,
        )
    except ImportError:
        raise ImportError(
            "sam2 is required for MatAnyone 2. "
            "Install with: uv sync --extra matanyone2"
        )

    if device is None:
        device = get_device()

    variant = _SAM2_VARIANTS.get(
        device.type, _SAM2_VARIANTS["cpu"]
    )
    logger.info(f"Loading SAM2 ({variant}) on {device}...")

    predictor = SAM2ImagePredictor.from_pretrained(
        variant, device=str(device)
    )
    mask_gen = SAM2AutomaticMaskGenerator(predictor.model)

    logger.info("Generating first-frame mask...")
    masks = mask_gen.generate(frame)

    if not masks:
        raise RuntimeError(
            "SAM2 found no objects in the first frame. "
            "The video may be too dark or featureless."
        )

    # Pick the largest mask (most likely the person)
    masks.sort(key=lambda m: m["area"], reverse=True)
    best = masks[0]["segmentation"].astype(np.uint8) * 255

    logger.info(
        f"Mask generated: {best.shape}, "
        f"coverage={masks[0]['area']}/{frame.shape[0]*frame.shape[1]}"
    )

    # Cleanup SAM2 to free GPU memory (D003)
    del mask_gen, predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("SAM2 unloaded, GPU memory freed")

    return best


class MatAnyone2Processor:
    """Process video frames through MatAnyone 2.

    Requires a first-frame mask (from SAM2 or manual input).
    Uses InferenceCore.step() for per-frame matting with
    recurrent memory propagation.

    Args:
        first_frame_mask: Binary mask (H, W), uint8.
        device: Target device. Auto-detected if None.
    """

    def __init__(
        self,
        first_frame_mask: np.ndarray,
        device: torch.device | None = None,
    ):
        try:
            from matanyone2 import MatAnyone2, InferenceCore
        except ImportError:
            raise ImportError(
                "matanyone2 is required. "
                "Install with: uv sync --extra matanyone2"
            )

        if device is None:
            device = get_device()

        self.device = device
        self._mask = first_frame_mask
        self._is_first_frame = True

        logger.info(f"Loading MatAnyone 2 on {device}...")
        model = MatAnyone2.from_pretrained(
            "PeiqingYang/MatAnyone2"
        )
        self._processor = InferenceCore(
            model, device=str(device)
        )
        logger.info("MatAnyone 2 loaded")

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame and return the alpha matte.

        Args:
            frame: RGB array (H, W, 3), uint8.

        Returns:
            Alpha matte (H, W), uint8 (0-255).
        """
        img_tensor = (
            torch.from_numpy(frame).float().permute(2, 0, 1)
            / 255.0
        )

        with torch.no_grad():
            if self._is_first_frame:
                mask_tensor = torch.from_numpy(
                    self._mask
                ).float() / 255.0
                output = self._processor.step(
                    img_tensor,
                    mask_tensor,
                    first_frame_pred=True,
                )
                self._is_first_frame = False
            else:
                output = self._processor.step(img_tensor)

        # output is alpha map in [0, 1]
        matte = (output * 255).byte().cpu().numpy()
        if matte.ndim == 3:
            matte = matte[0]
        return matte

    def reset(self) -> None:
        """Reset state for a new video."""
        self._is_first_frame = True

    def cleanup(self) -> None:
        """Release model and GPU memory."""
        if hasattr(self, "_processor"):
            del self._processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("MatAnyone 2 processor cleaned up")
