"""ROI-restricted matting — matte only where the subject is.

For VR passthrough the person typically fills a fraction of the
frame; matting the full frame wastes most of the GPU work. This
wrapper tracks the subject's bounding box from each output matte,
crops subsequent frames to a padded window around it, runs the
inner processor on the crop, and pastes the result into a
full-size matte.

Model-agnostic: wraps any per-frame MatteProcessor. Recurrent /
memory models are handled at window moves:

- Models exposing ``reseed(frame, mask)`` (MatAnyone2) are
  re-initialized with the previous matte as the mask for the new
  window, preserving subject identity.
- Other recurrent models (RVM) get ``reset()``; their state
  re-converges within a few frames.

The window is *sticky*: it only moves when the subject approaches
its edge or the size becomes wrong, and its dimensions are
quantized to multiples of 64 px so fixed-shape optimizations
(cuDNN benchmark, CUDA graphs) keep paying off.

Safety valves:
- No subject found -> falls back to full-frame processing until
  the subject reappears.
- Subject covering most of the frame -> ROI disables itself (no
  benefit, avoids churn).
"""

import numpy as np
from loguru import logger

# Matte values above this count as "subject" for bbox tracking.
_BBOX_THRESHOLD = 16
# Window edge padding (px at matte scale): if the subject bbox
# gets closer than this to the window edge, re-anchor.
_EDGE_PAD = 24
# Window dims are rounded up to multiples of this.
_QUANTUM = 64
# If the padded subject bbox covers more than this fraction of
# the frame, process full-frame (ROI gives no benefit).
_MAX_COVERAGE = 0.72


def _matte_bbox(matte: np.ndarray) -> tuple | None:
    """Approximate bbox (x0, y0, x1, y1) of the subject.

    Scans a 4x-downsampled view for speed; the margin added by
    the caller absorbs the quantization error.
    """
    small = matte[::4, ::4]
    ys, xs = np.where(small > _BBOX_THRESHOLD)
    if len(ys) == 0:
        return None
    return (
        int(xs.min()) * 4, int(ys.min()) * 4,
        min((int(xs.max()) + 1) * 4, matte.shape[1]),
        min((int(ys.max()) + 1) * 4, matte.shape[0]),
    )


class ROICropper:
    """Crop-to-subject wrapper for per-frame matting processors.

    Args:
        inner: The wrapped MatteProcessor.
        margin: Fractional padding around the subject bbox when
            (re)anchoring the window.
    """

    def __init__(self, inner, margin: float = 0.35):
        self._inner = inner
        self._margin = margin
        self._window: tuple | None = None
        self._prev_matte: np.ndarray | None = None
        self._needs_reseed = False
        self._full_w = 0
        self._full_h = 0

    @property
    def supports_pair(self) -> bool:
        """Pair mode available iff the inner processor has it."""
        return getattr(self._inner, "supports_pair", False)

    # ── window management ───────────────────────────────────

    def _compute_window(
        self, bbox: tuple, h: int, w: int,
    ) -> tuple | None:
        """Padded, quantized, clamped window around a bbox.

        Returns None when the window would cover most of the
        frame (full-frame processing is then cheaper).
        """
        x0, y0, x1, y1 = bbox
        bw, bh = x1 - x0, y1 - y0
        pad_x = int(bw * self._margin)
        pad_y = int(bh * self._margin)

        # Quantized target dims (multiples of _QUANTUM, even).
        win_w = min(
            w,
            (bw + 2 * pad_x + _QUANTUM - 1)
            // _QUANTUM * _QUANTUM,
        )
        win_h = min(
            h,
            (bh + 2 * pad_y + _QUANTUM - 1)
            // _QUANTUM * _QUANTUM,
        )
        if win_w * win_h > _MAX_COVERAGE * w * h:
            return None

        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        wx0 = max(0, min(cx - win_w // 2, w - win_w))
        wy0 = max(0, min(cy - win_h // 2, h - win_h))
        return (wx0, wy0, wx0 + win_w, wy0 + win_h)

    def _window_ok(self, bbox: tuple) -> bool:
        """True while the subject sits comfortably inside.

        A bbox edge is allowed to approach a window edge only
        where the window is already clamped at the frame border
        (nowhere left to grow).
        """
        if self._window is None:
            return False
        x0, y0, x1, y1 = bbox
        wx0, wy0, wx1, wy1 = self._window
        ok_left = x0 >= wx0 + _EDGE_PAD or wx0 == 0
        ok_top = y0 >= wy0 + _EDGE_PAD or wy0 == 0
        ok_right = (
            x1 <= wx1 - _EDGE_PAD or wx1 >= self._full_w
        )
        ok_bottom = (
            y1 <= wy1 - _EDGE_PAD or wy1 >= self._full_h
        )
        return ok_left and ok_top and ok_right and ok_bottom

    def _track(self, matte: np.ndarray) -> None:
        """Update window state from the latest full-size matte."""
        h, w = matte.shape[:2]
        self._full_h, self._full_w = h, w
        self._prev_matte = matte
        bbox = _matte_bbox(matte)

        if bbox is None:
            # Subject lost — go full-frame until re-found.
            if self._window is not None:
                logger.debug("ROI: subject lost, full-frame")
                self._window = None
                self._needs_reseed = True
            return

        if self._window is not None and self._window_ok(bbox):
            return

        new_win = self._compute_window(bbox, h, w)
        if new_win != self._window:
            self._window = new_win
            self._needs_reseed = True
            if new_win:
                wx0, wy0, wx1, wy1 = new_win
                logger.debug(
                    f"ROI window -> {wx1 - wx0}x{wy1 - wy0} "
                    f"at ({wx0},{wy0})"
                )
            else:
                logger.debug("ROI: subject too large, full-frame")

    def _run_inner(self, crop, mask_crop):
        """Step the inner processor, reseeding after a window move."""
        if self._needs_reseed:
            self._needs_reseed = False
            if (
                hasattr(self._inner, "reseed")
                and mask_crop is not None
            ):
                return self._inner.reseed(
                    crop, (mask_crop > 127).astype(np.uint8)
                )
            self._inner.reset()
        return self._inner.process_frame(crop)

    # ── MatteProcessor protocol ─────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        win = self._window

        if win is None or self._prev_matte is None:
            matte = self._run_inner(
                frame,
                self._prev_matte
                if self._prev_matte is not None
                else None,
            )
            self._track(matte)
            return matte

        x0, y0, x1, y1 = win
        crop = frame[y0:y1, x0:x1]
        mask_crop = self._prev_matte[y0:y1, x0:x1]
        m = self._run_inner(crop, mask_crop)

        matte = np.zeros((h, w), dtype=np.uint8)
        matte[y0:y1, x0:x1] = m
        self._track(matte)
        return matte

    def process_frame_pair(self, left, right):
        """Batched two-eye processing with a shared window.

        SBS eyes see nearly the same scene, so one window (the
        union of both subjects) serves both — required anyway
        for batch processing, which needs equal shapes.
        """
        h, w = left.shape[:2]
        win = self._window

        if win is None or self._prev_matte is None:
            if self._needs_reseed:
                self._needs_reseed = False
                self._inner.reset()
            lm, rm = self._inner.process_frame_pair(
                left, right
            )
            self._track(np.maximum(lm, rm))
            return lm, rm

        x0, y0, x1, y1 = win
        if self._needs_reseed:
            self._needs_reseed = False
            self._inner.reset()
        lm_c, rm_c = self._inner.process_frame_pair(
            left[y0:y1, x0:x1], right[y0:y1, x0:x1]
        )
        lm = np.zeros((h, w), dtype=np.uint8)
        rm = np.zeros((h, w), dtype=np.uint8)
        lm[y0:y1, x0:x1] = lm_c
        rm[y0:y1, x0:x1] = rm_c
        self._track(np.maximum(lm, rm))
        return lm, rm

    def reset(self) -> None:
        self._inner.reset()
        self._window = None
        self._prev_matte = None
        self._needs_reseed = False

    def cleanup(self) -> None:
        self._inner.cleanup()
        self._prev_matte = None
