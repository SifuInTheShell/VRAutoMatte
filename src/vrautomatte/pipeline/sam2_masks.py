"""SAM2 mask generation and POV heuristics.

Generates segmentation masks from video frames using SAM2.
Supports POV mode — selects the non-POV person or identifies
the POV body for exclusion.

GPU memory strategy: SAM2 loads, generates masks from frame 1,
then unloads before the matting model loads (D003).
"""

import gc

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

    # Person-sized candidates only (1%-50% of the frame). A
    # fisheye VR eye yields a giant "entire visible area"
    # mask that sits dead-center and outscores the actual
    # people — no real subject ever covers half the frame.
    candidates = [
        m for m in masks
        if 0.01 * total_px < m["area"] <= 0.50 * total_px
    ]
    rejected = len(masks) - len(candidates)
    if rejected:
        logger.debug(
            f"POV scoring: rejected {rejected} mask(s) "
            "outside the 1%-50% subject size range"
        )
    if not candidates:
        logger.warning(
            "POV scoring: no person-sized masks found — "
            "falling back to all masks; subject selection "
            "may be degenerate"
        )
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


def _load_stock_amg():
    """Load SAM2AutomaticMaskGenerator from the stock install.

    The SAM2Matting fork shadows the stock sam2 package on
    sys.path but does not ship automatic_mask_generator. The
    stock module file runs fine against the fork — every
    sam2.* name it imports (modeling, utils.amg,
    sam2_image_predictor) exists there — so locate it in the
    stock install and register it under the active package.
    """
    import importlib.util
    import sys
    from pathlib import Path

    import sam2

    active_dir = Path(sam2.__file__).parent
    for entry in sys.path:
        candidate = (
            Path(entry) / "sam2"
            / "automatic_mask_generator.py"
        )
        if (
            candidate.is_file()
            and candidate.parent != active_dir
        ):
            spec = importlib.util.spec_from_file_location(
                "sam2.automatic_mask_generator", candidate
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            logger.debug(f"Stock AMG loaded from {candidate}")
            return module.SAM2AutomaticMaskGenerator
    raise ImportError(
        "sam2.automatic_mask_generator not found — the active "
        f"sam2 package ({active_dir}) does not ship it and no "
        "stock sam2 install is on sys.path. "
        "Install with: uv sync --extra matanyone2"
    )


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

    Runs inside stock_sam2(): once the SAM2Matting fork is
    active in a session (any earlier sam2matting processor),
    ``import sam2`` would resolve to the fork, which lacks
    both the automatic mask generator and the standard model
    configs. The context sidelines the fork for the duration
    and restores it after.
    """
    from vrautomatte.pipeline.sam2matting import stock_sam2

    with stock_sam2():
        try:
            from sam2.sam2_image_predictor import (
                SAM2ImagePredictor,
            )
        except ImportError:
            raise ImportError(
                "sam2 is required for POV mode / "
                "MatAnyone 2. Install with: "
                "uv sync --extra matanyone2"
            )
        try:
            from sam2.automatic_mask_generator import (
                SAM2AutomaticMaskGenerator,
            )
        except ImportError:
            # Safety net — with the fork sidelined this
            # should not trigger, but pull the module from
            # the stock install if it somehow does.
            SAM2AutomaticMaskGenerator = _load_stock_amg()

        if device is None:
            device = get_device()

        variant = _SAM2_VARIANTS.get(
            device.type, _SAM2_VARIANTS["cpu"]
        )
        logger.info(
            f"Loading SAM2 ({variant}) on {device}..."
        )

        predictor = SAM2ImagePredictor.from_pretrained(
            variant, device=str(device)
        )
        mask_gen = SAM2AutomaticMaskGenerator(
            predictor.model
        )

        logger.info("Generating masks from first frame...")
        masks = mask_gen.generate(frame)

        del mask_gen, predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("SAM2 unloaded, GPU memory freed")

    if not masks:
        raise RuntimeError(
            "SAM2 found no objects in the first frame. "
            "The video may be too dark or featureless."
        )

    return masks


def _score_person_masks(
    masks: list, frame_shape: tuple,
) -> list[tuple[float, dict]]:
    """Score masks by person-likeness.

    Returns:
        List of (score, mask_dict) sorted highest-first.
    """
    h, w = frame_shape[:2]
    total_px = h * w
    center_y, center_x = h / 2, w / 2

    # Filter to reasonable person-sized masks (1%-60%)
    candidates = [
        m for m in masks
        if 0.01 * total_px < m["area"] < 0.60 * total_px
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

        ys, xs = np.where(seg)
        if len(ys) > 0:
            bbox_h = ys.max() - ys.min() + 1
            bbox_w = xs.max() - xs.min() + 1
            aspect = bbox_h / max(bbox_w, 1)
        else:
            aspect = 1.0

        score = 1.0
        score -= dist_y * 0.25
        score -= dist_x * 0.2
        if 0.03 < area_frac < 0.40:
            score += 0.2
        elif area_frac >= 0.60:
            score -= 0.4
        if aspect > 1.5:
            score += 0.15
        score += min(area_frac, 0.3) * 0.15

        scored.append((score, m))
        logger.debug(
            f"Person mask score={score:.2f} "
            f"area={area_frac:.2%} "
            f"aspect={aspect:.1f} "
            f"center=({cy:.0f},{cx:.0f})"
        )

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _select_person_mask(
    masks: list, frame_shape: tuple
) -> np.ndarray:
    """Select the most likely person mask from SAM2 output.

    Returns the single highest-scoring person-shaped mask.

    Args:
        masks: SAM2 mask list with 'segmentation'/'area'.
        frame_shape: (H, W, C).

    Returns:
        Binary mask (H, W), uint8 (0 or 255).
    """
    scored = _score_person_masks(masks, frame_shape)
    best = scored[0]
    total_px = frame_shape[0] * frame_shape[1]
    mask = best[1]["segmentation"].astype(np.uint8) * 255
    logger.info(
        f"Selected person mask: score={best[0]:.2f}, "
        f"coverage={best[1]['area']}/{total_px}"
    )
    return mask


def _select_all_person_masks(
    masks: list, frame_shape: tuple,
    min_score: float = 0.4,
) -> np.ndarray:
    """Select all person-like masks and combine into one.

    Union of all masks scoring above ``min_score``. Falls back
    to the single best mask if none meet the threshold.

    Args:
        masks: SAM2 mask list with 'segmentation'/'area'.
        frame_shape: (H, W, C).
        min_score: Minimum person-likeness score to include.

    Returns:
        Combined binary mask (H, W), uint8 (0 or 255).
    """
    scored = _score_person_masks(masks, frame_shape)
    total_px = frame_shape[0] * frame_shape[1]

    selected = [
        m for score, m in scored if score >= min_score
    ]
    if not selected:
        selected = [scored[0][1]]

    combined = np.zeros(
        frame_shape[:2], dtype=np.uint8
    )
    for m in selected:
        combined[m["segmentation"]] = 255

    coverage = int(combined.sum() / 255)
    logger.info(
        f"Combined {len(selected)} person masks "
        f"(min_score={min_score}): "
        f"coverage={coverage}/{total_px}"
    )
    return combined


def _select_person_masks_multi(
    masks: list, frame_shape: tuple,
    max_people: int = 4,
    min_score: float = 0.4,
    overlap_thresh: float = 0.5,
) -> list[np.ndarray]:
    """Select up to N distinct person masks, best-first.

    Greedy pick by person-likeness score with overlap
    de-duplication: SAM2's automatic masks often nest
    (torso-inside-person), so a candidate covering more than
    ``overlap_thresh`` of an already-picked mask (or vice
    versa, relative to the smaller area) is skipped.

    Args:
        masks: SAM2 mask list with 'segmentation'/'area'.
        frame_shape: (H, W, C).
        max_people: Maximum number of masks to return.
        min_score: Minimum person-likeness for extra subjects
            (the first/best subject is always returned).

    Returns:
        List of 1..max_people binary masks (H, W), uint8
        (0 or 255), ordered best-first.
    """
    scored = _score_person_masks(masks, frame_shape)
    picked: list[np.ndarray] = []
    picked_bool: list[np.ndarray] = []

    for score, m in scored:
        if len(picked) >= max_people:
            break
        if picked and score < min_score:
            break
        seg = m["segmentation"]
        seg_area = seg.sum()
        if seg_area == 0:
            continue
        duplicate = False
        for prev in picked_bool:
            inter = np.logical_and(seg, prev).sum()
            smaller = min(seg_area, prev.sum())
            if smaller > 0 and inter / smaller > overlap_thresh:
                duplicate = True
                break
        if duplicate:
            continue
        picked_bool.append(seg)
        picked.append(seg.astype(np.uint8) * 255)
        logger.info(
            f"Subject {len(picked)}: score={score:.2f}, "
            f"area={seg_area}"
        )

    if not picked:
        picked = [
            scored[0][1]["segmentation"].astype(np.uint8)
            * 255
        ]
    return picked


def generate_person_masks(
    frame: np.ndarray,
    device: torch.device | None = None,
    max_people: int = 2,
    pov_mode: bool = False,
) -> list[np.ndarray]:
    """Auto-generate separate per-person masks from frame 1.

    Used for multi-subject tracking (SAM2Matting): each mask
    becomes its own tracked object. In POV mode, candidates
    overlapping the detected POV body are excluded first.

    Args:
        frame: RGB array (H, W, 3), uint8.
        device: Target device. Auto-detected if None.
        max_people: Maximum subjects to track (1-4 typical).
        pov_mode: Exclude the POV body from the candidates.

    Returns:
        List of binary masks (H, W), uint8 (0 or 255).
    """
    masks = _run_sam2_masks(frame, device)

    if pov_mode and len(masks) > 1:
        pov_scored = _score_pov_masks(masks, frame.shape)
        pov_seg = pov_scored[-1][1]["segmentation"]
        pov_area = max(pov_seg.sum(), 1)

        def is_pov(m):
            inter = np.logical_and(
                m["segmentation"], pov_seg
            ).sum()
            smaller = min(
                m["segmentation"].sum(), pov_area
            )
            return smaller > 0 and inter / smaller > 0.5

        filtered = [m for m in masks if not is_pov(m)]
        if filtered:
            masks = filtered

    return _select_person_masks_multi(
        masks, frame.shape, max_people=max_people
    )


def _run_sam2_prompted(
    frame: np.ndarray,
    device: torch.device | None = None,
) -> tuple:
    """Segment the centered subject with a point prompt.

    Prompted segmentation is far more reliable than the
    automatic mask generator for close-up subjects on fisheye
    VR frames, where AMG often fragments the person into
    clothing/limb pieces or misses them entirely. The prompt
    sits slightly above frame center — where the subject is
    by construction in VR POV content.

    Returns:
        (masks, scores) from SAM2ImagePredictor, or (None,
        None) if sam2 is unavailable.
    """
    from vrautomatte.pipeline.sam2matting import stock_sam2

    with stock_sam2():
        try:
            from sam2.sam2_image_predictor import (
                SAM2ImagePredictor,
            )
        except ImportError:
            return None, None

        if device is None:
            device = get_device()
        variant = _SAM2_VARIANTS.get(
            device.type, _SAM2_VARIANTS["cpu"]
        )
        logger.info(
            f"Loading SAM2 ({variant}) for prompted "
            f"subject segmentation..."
        )
        predictor = SAM2ImagePredictor.from_pretrained(
            variant, device=str(device)
        )
        h, w = frame.shape[:2]
        points = np.array([[w // 2, int(h * 0.45)]])
        predictor.set_image(frame)
        masks, scores, _ = predictor.predict(
            point_coords=points,
            point_labels=np.array([1]),
            multimask_output=True,
        )

        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("SAM2 unloaded, GPU memory freed")

    return masks, scores


def _select_prompted_subject(
    frame: np.ndarray,
    device: torch.device | None = None,
) -> np.ndarray | None:
    """Prompted subject mask, or None if implausible.

    Picks the highest-scoring prompted mask whose area is
    person-plausible (2%-60% of the frame). None means the
    caller should fall back to AMG-based selection.
    """
    masks, scores = _run_sam2_prompted(frame, device)
    if masks is None:
        return None

    h, w = frame.shape[:2]
    total_px = h * w
    best = None
    for m, s in sorted(
        zip(masks, scores), key=lambda x: -x[1]
    ):
        area_frac = m.sum() / total_px
        if 0.02 <= area_frac <= 0.60 and s >= 0.5:
            best = (m, s, area_frac)
            break
    if best is None:
        logger.info(
            "Prompted subject mask implausible — falling "
            "back to automatic mask selection"
        )
        return None

    m, s, area_frac = best
    logger.info(
        f"Prompted subject mask: score={s:.2f}, "
        f"coverage={int(m.sum())}/{total_px}"
    )
    return m.astype(np.uint8) * 255


def generate_first_frame_mask(
    frame: np.ndarray,
    device: torch.device | None = None,
    pov_mode: bool = False,
) -> np.ndarray:
    """Auto-generate a segmentation mask from frame 1.

    Primary: point-prompted segmentation of the centered
    subject. Fallback: AMG mask scoring (union of person
    masks, or the non-POV person in POV mode).

    Args:
        frame: RGB array (H, W, 3), uint8.
        device: Target device. Auto-detected if None.
        pov_mode: If True, select non-POV person mask.

    Returns:
        Binary mask (H, W), uint8 (0 or 255).
    """
    prompted = _select_prompted_subject(frame, device)
    if prompted is not None:
        return prompted

    masks = _run_sam2_masks(frame, device)

    if pov_mode:
        return _select_non_pov_mask(masks, frame.shape)

    return _select_all_person_masks(masks, frame.shape)


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
