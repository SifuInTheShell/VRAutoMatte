"""SAM2 mask generation and POV heuristics.

Generates segmentation masks from video frames using SAM2.
Supports POV mode — selects the non-POV person or identifies
the POV body for exclusion.

GPU memory strategy: SAM2 loads, generates masks from frame 1,
then unloads before the matting model loads (D003).
"""

import numpy as np
import torch
from loguru import logger

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


def _score_pov_masks(
    masks: list, frame_shape: tuple
) -> list:
    """Score masks for POV body likelihood (low = POV body).

    Higher score = more likely the non-POV subject
    (centered, not bottom-heavy).

    Args:
        masks: SAM2 mask list with 'segmentation'/'area'.
        frame_shape: (H, W, C) of the original frame.

    Returns:
        List of (score, mask_dict) sorted highest-first.
    """
    h, w = frame_shape[:2]
    total_px = h * w
    center_y, center_x = h / 2, w / 2

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

        dist_y = abs(cy - center_y) / center_y
        dist_x = abs(cx - center_x) / center_x

        bottom_rows = seg[int(h * 0.85):, :]
        bottom_frac = (
            bottom_rows.sum() / max(bottom_rows.size, 1)
        )

        score = 1.0
        score -= dist_y * 0.3
        score -= dist_x * 0.2
        score -= bottom_frac * 0.4
        if area_frac > 0.6:
            score -= 0.3
        score += min(area_frac, 0.4) * 0.2

        scored.append((score, m))
        logger.debug(
            f"POV mask score={score:.2f} "
            f"area={area_frac:.2%} "
            f"center=({cy:.0f},{cx:.0f}) "
            f"bottom={bottom_frac:.2%}"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _select_non_pov_mask(
    masks: list, frame_shape: tuple
) -> np.ndarray:
    """Select the non-POV person mask (subject facing camera).

    Args:
        masks: SAM2 mask list.
        frame_shape: (H, W, C).

    Returns:
        Binary mask (H, W), uint8 (0 or 255).
    """
    scored = _score_pov_masks(masks, frame_shape)
    best_mask = scored[0][1]
    best = best_mask["segmentation"].astype(np.uint8) * 255
    total_px = frame_shape[0] * frame_shape[1]

    logger.info(
        f"POV subject mask: score={scored[0][0]:.2f}, "
        f"coverage={best_mask['area']}/{total_px}"
    )
    return best


def _select_pov_body_mask(
    masks: list, frame_shape: tuple
) -> np.ndarray:
    """Select the POV body mask for exclusion.

    Picks lowest-scoring mask (bottom-heavy, edge-touching).
    Dilates by 2% of frame height for movement tolerance.

    Args:
        masks: SAM2 mask list.
        frame_shape: (H, W, C).

    Returns:
        Binary mask (H, W), uint8 (0 or 255).
    """
    scored = _score_pov_masks(masks, frame_shape)
    if len(scored) < 2:
        pov = scored[0][1]
    else:
        pov = scored[-1][1]

    body = pov["segmentation"].astype(np.uint8) * 255
    total_px = frame_shape[0] * frame_shape[1]

    h = frame_shape[0]
    dilate_px = max(int(h * 0.02), 3)
    try:
        import cv2
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)
        )
        body = cv2.dilate(body, kernel, iterations=1)
    except ImportError:
        logger.warning(
            "cv2 not available for mask dilation"
        )

    logger.info(
        f"POV body mask: score={scored[-1][0]:.2f}, "
        f"coverage={pov['area']}/{total_px}, "
        f"dilated by {dilate_px}px"
    )
    return body


def _run_sam2_masks(
    frame: np.ndarray,
    device: torch.device | None = None,
) -> list:
    """Run SAM2 automatic mask generator on a frame.

    Loads SAM2, generates masks, unloads to free GPU (D003).

    Args:
        frame: RGB array (H, W, 3), uint8.
        device: Target device. Auto-detected if None.

    Returns:
        List of SAM2 mask dicts.
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
            "sam2 is required for POV mode / MatAnyone 2. "
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

    logger.info("Generating masks from first frame...")
    masks = mask_gen.generate(frame)

    del mask_gen, predictor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.debug("SAM2 unloaded, GPU memory freed")

    if not masks:
        raise RuntimeError(
            "SAM2 found no objects in the first frame. "
            "The video may be too dark or featureless."
        )

    return masks


def generate_first_frame_mask(
    frame: np.ndarray,
    device: torch.device | None = None,
    pov_mode: bool = False,
) -> np.ndarray:
    """Auto-generate a segmentation mask from frame 1.

    Default: largest mask. POV mode: non-POV person.

    Args:
        frame: RGB array (H, W, 3), uint8.
        device: Target device. Auto-detected if None.
        pov_mode: If True, select non-POV person mask.

    Returns:
        Binary mask (H, W), uint8 (0 or 255).
    """
    masks = _run_sam2_masks(frame, device)

    if pov_mode:
        return _select_non_pov_mask(masks, frame.shape)

    masks.sort(key=lambda m: m["area"], reverse=True)
    best = masks[0]["segmentation"].astype(np.uint8) * 255
    logger.info(
        f"Mask generated: {best.shape}, "
        f"coverage={masks[0]['area']}/"
        f"{frame.shape[0]*frame.shape[1]}"
    )
    return best


def generate_pov_body_mask(
    frame: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray:
    """Generate a mask of the POV body for exclusion.

    Uses SAM2 to find the bottom-heavy, edge-touching mask
    most likely to be the POV body, then dilates slightly.

    Args:
        frame: RGB array (H, W, 3), uint8.
        device: Target device. Auto-detected if None.

    Returns:
        Binary mask (H, W), uint8 (0 or 255).
    """
    masks = _run_sam2_masks(frame, device)
    return _select_pov_body_mask(masks, frame.shape)
