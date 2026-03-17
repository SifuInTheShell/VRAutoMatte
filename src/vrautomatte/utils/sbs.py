"""SBS (side-by-side) stereo utilities.

Handles detection, splitting, and recombination of SBS stereo
frames at the numpy array level for per-eye matting.
"""

import numpy as np
from loguru import logger


def detect_sbs(width: int, height: int) -> bool:
    """Detect if a video is likely side-by-side stereo.

    Uses aspect ratio heuristic: width/height >= 1.9 suggests
    SBS content (each eye is roughly square or 16:9).

    Args:
        width: Video width in pixels.
        height: Video height in pixels.

    Returns:
        True if the video appears to be SBS stereo.
    """
    if height <= 0:
        return False
    ratio = width / height
    is_sbs = ratio >= 1.9
    if is_sbs:
        logger.info(
            f"SBS detected: {width}x{height} "
            f"(ratio={ratio:.2f})"
        )
    return is_sbs


def split_frame(frame: np.ndarray) -> tuple:
    """Split an SBS frame into left and right eyes.

    Args:
        frame: RGB array (H, W, 3) — full SBS width.

    Returns:
        Tuple of (left_eye, right_eye), each (H, W/2, 3).
    """
    h, w, c = frame.shape
    mid = w // 2
    return frame[:, :mid, :].copy(), frame[:, mid:, :].copy()


def merge_frames(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Merge left and right eye frames back to SBS.

    Args:
        left: Left eye array (H, W_half, ...).
        right: Right eye array (H, W_half, ...).

    Returns:
        Merged SBS array (H, W, ...).
    """
    return np.concatenate([left, right], axis=1)


def split_matte(matte: np.ndarray) -> tuple:
    """Split an SBS matte into left and right eyes.

    Args:
        matte: Grayscale array (H, W) — full SBS width.

    Returns:
        Tuple of (left_matte, right_matte), each (H, W/2).
    """
    h, w = matte.shape
    mid = w // 2
    return matte[:, :mid].copy(), matte[:, mid:].copy()


def merge_mattes(
    left: np.ndarray, right: np.ndarray
) -> np.ndarray:
    """Merge left and right eye mattes back to SBS.

    Args:
        left: Left eye matte (H, W_half).
        right: Right eye matte (H, W_half).

    Returns:
        Merged SBS matte (H, W).
    """
    return np.concatenate([left, right], axis=1)
