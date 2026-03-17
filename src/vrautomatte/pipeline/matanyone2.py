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


def _mask_center_of_mass(seg: np.ndarray) -> tuple:
    """Return (cy, cx) center of mass for a binary mask."""
    ys, xs = np.where(seg)
    if len(ys) == 0:
        return (0, 0)
    return (ys.mean(), xs.mean())


def _select_non_pov_mask(
    masks: list, frame_shape: tuple
) -> np.ndarray:
    """Select the non-POV person mask for POV content.

    POV body heuristic: in first-person content, the POV
    body tends to occupy the lower portion and edges of
    the frame. The other person is typically centered
    vertically and horizontally (facing the camera).

    Scoring: prefer masks whose center of mass is in the
    upper-center region. Penalize masks that touch the
    bottom edge heavily or are extremely large (full-frame
    POV body).

    Args:
        masks: SAM2 mask list with 'segmentation' and 'area'.
        frame_shape: (H, W, C) of the original frame.

    Returns:
        Binary mask (H, W), uint8 (0 or 255).
    """
    h, w = frame_shape[:2]
    total_px = h * w
    center_y, center_x = h / 2, w / 2

    # Filter to meaningful masks (>1% of frame)
    candidates = [
        m for m in masks if m["area"] > total_px * 0.01
    ]
    if not candidates:
        candidates = masks

    scored = []
    for m in candidates:
        seg = m["segmentation"]
        area_frac = m["area"] / total_px
        cy, cx = _mask_center_of_mass(seg)

        # Distance from center (normalized 0-1)
        dist_y = abs(cy - center_y) / center_y
        dist_x = abs(cx - center_x) / center_x

        # Bottom-edge contact: fraction of bottom row
        bottom_rows = seg[int(h * 0.85):, :]
        bottom_frac = (
            bottom_rows.sum() / max(bottom_rows.size, 1)
        )

        # Score: prefer centered, penalize bottom-heavy
        # and extremely large (POV body covers >60%)
        score = 1.0
        score -= dist_y * 0.3      # penalize far from v-center
        score -= dist_x * 0.2      # penalize far from h-center
        score -= bottom_frac * 0.4  # penalize bottom contact
        if area_frac > 0.6:
            score -= 0.3           # penalize very large masks
        # Slight preference for larger subjects
        score += min(area_frac, 0.4) * 0.2

        scored.append((score, m))
        logger.debug(
            f"POV mask score={score:.2f} area={area_frac:.2%} "
            f"center=({cy:.0f},{cx:.0f}) bottom={bottom_frac:.2%}"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    best_mask = scored[0][1]
    best = best_mask["segmentation"].astype(np.uint8) * 255

    logger.info(
        f"POV mask selected: score={scored[0][0]:.2f}, "
        f"coverage={best_mask['area']}/{total_px}"
    )
    return best


def generate_first_frame_mask(
    frame: np.ndarray,
    device: torch.device | None = None,
    pov_mode: bool = False,
) -> np.ndarray:
    """Auto-generate a segmentation mask from frame 1.

    In default mode, picks the largest mask (the main subject).
    In POV mode, picks the non-POV person — the subject facing
    the camera, excluding the POV body at the frame edges/bottom.

    Args:
        frame: RGB array (H, W, 3), uint8.
        device: Target device. Auto-detected if None.
        pov_mode: If True, exclude the POV body from the mask.

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

    if pov_mode:
        best = _select_non_pov_mask(masks, frame.shape)
    else:
        # Pick the largest mask (most likely the person)
        masks.sort(key=lambda m: m["area"], reverse=True)
        best = masks[0]["segmentation"].astype(
            np.uint8
        ) * 255
        logger.info(
            f"Mask generated: {best.shape}, "
            f"coverage={masks[0]['area']}/"
            f"{frame.shape[0]*frame.shape[1]}"
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
        ).to(self.device)

        with torch.no_grad():
            if self._is_first_frame:
                # Step 1: feed mask into memory
                mask_tensor = torch.from_numpy(
                    self._mask
                ).float().to(self.device)
                self._processor.step(
                    img_tensor,
                    mask_tensor,
                    objects=[1],
                )
                # Step 2: predict using that memory
                output = self._processor.step(
                    img_tensor,
                    first_frame_pred=True,
                )
                self._is_first_frame = False
            else:
                output = self._processor.step(img_tensor)

        # Extract alpha from output probabilities
        matte = self._processor.output_prob_to_mask(output)
        matte = (matte * 255).byte().cpu().numpy()
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
