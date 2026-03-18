"""Frame scaling for GPU VRAM-constrained matting.

When input frames exceed the GPU's VRAM budget, frames are
downscaled before matting and the resulting matte is upscaled
back to original resolution. Uses LANCZOS interpolation.
No-op if frames already fit within max_pixels.
"""

import numpy as np
from loguru import logger
from PIL import Image


class FrameScaler:
    """Downscale frames before matting, upscale mattes after.

    Args:
        max_pixels: Maximum pixel count (width * height) for
            matting. Frames exceeding this are downscaled.
            0 or negative = no limit (no-op).
        original_size: (width, height) of the input frames.
    """

    def __init__(
        self,
        max_pixels: int,
        original_size: tuple[int, int],
    ):
        self._original_w, self._original_h = original_size
        self._active = False
        self._target_w = self._original_w
        self._target_h = self._original_h

        orig_pixels = self._original_w * self._original_h
        if max_pixels > 0 and orig_pixels > max_pixels:
            scale = (max_pixels / orig_pixels) ** 0.5
            # Round to even dimensions (required by codecs).
            self._target_w = int(self._original_w * scale) & ~1
            self._target_h = int(self._original_h * scale) & ~1
            self._active = True
            logger.info(
                f"FrameScaler: {self._original_w}x"
                f"{self._original_h} -> "
                f"{self._target_w}x{self._target_h} "
                f"({self._target_w * self._target_h:,} px, "
                f"scale={scale:.3f})"
            )

    @property
    def active(self) -> bool:
        """True if scaling will be applied."""
        return self._active

    @property
    def target_size(self) -> tuple[int, int]:
        """(width, height) after downscaling."""
        return (self._target_w, self._target_h)

    def downscale(self, frame: np.ndarray) -> np.ndarray:
        """Downscale an RGB frame if it exceeds max_pixels.

        Args:
            frame: RGB array (H, W, 3), uint8.

        Returns:
            Downscaled frame, or original if no scaling needed.
        """
        if not self._active:
            return frame
        img = Image.fromarray(frame)
        img = img.resize(
            (self._target_w, self._target_h),
            Image.LANCZOS,
        )
        return np.array(img)

    def upscale_matte(self, matte: np.ndarray) -> np.ndarray:
        """Upscale a grayscale matte to original resolution.

        Args:
            matte: Grayscale matte (H, W), uint8.

        Returns:
            Upscaled matte, or original if no scaling needed.
        """
        if not self._active:
            return matte
        img = Image.fromarray(matte, mode="L")
        img = img.resize(
            (self._original_w, self._original_h),
            Image.LANCZOS,
        )
        return np.array(img)
