"""Scene change detection for POV mask refresh.

Detects drastic scene changes (cuts, major position shifts)
by comparing frame histograms. Lightweight — runs per-frame
with negligible overhead.
"""

import numpy as np
from loguru import logger


class SceneChangeDetector:
    """Detect scene changes via histogram correlation.

    Compares each frame's color histogram against a reference.
    When correlation drops below threshold, signals a scene
    change and updates the reference.

    A cooldown period prevents rapid re-triggers during
    transition sequences.

    Args:
        threshold: Correlation below this triggers change.
            Range 0-1, lower = more sensitive. Default 0.4.
        cooldown_frames: Minimum frames between detections.
    """

    def __init__(
        self,
        threshold: float = 0.4,
        cooldown_frames: int = 30,
    ):
        self._threshold = threshold
        self._cooldown = cooldown_frames
        self._reference_hist = None
        self._frames_since_change = 0

    def check(self, frame: np.ndarray) -> bool:
        """Check if the current frame is a scene change.

        Args:
            frame: RGB array (H, W, 3), uint8.

        Returns:
            True if scene change detected.
        """
        hist = self._compute_hist(frame)

        if self._reference_hist is None:
            self._reference_hist = hist
            self._frames_since_change = 0
            return False

        self._frames_since_change += 1

        if self._frames_since_change < self._cooldown:
            return False

        corr = self._correlate(
            self._reference_hist, hist
        )

        if corr < self._threshold:
            logger.info(
                f"Scene change detected: "
                f"correlation={corr:.3f} "
                f"(threshold={self._threshold})"
            )
            self._reference_hist = hist
            self._frames_since_change = 0
            return True

        return False

    def update_reference(self, frame: np.ndarray) -> None:
        """Manually update the reference frame.

        Call after mask regeneration to anchor the
        detector to the new scene.

        Args:
            frame: RGB array (H, W, 3), uint8.
        """
        self._reference_hist = self._compute_hist(frame)
        self._frames_since_change = 0

    def reset(self) -> None:
        """Reset detector state."""
        self._reference_hist = None
        self._frames_since_change = 0

    @staticmethod
    def _compute_hist(
        frame: np.ndarray, bins: int = 64
    ) -> np.ndarray:
        """Compute normalized color histogram.

        Uses a downsampled grayscale conversion for speed.
        """
        # Convert to grayscale via luminance
        gray = (
            0.299 * frame[:, :, 0]
            + 0.587 * frame[:, :, 1]
            + 0.114 * frame[:, :, 2]
        )
        hist, _ = np.histogram(
            gray, bins=bins, range=(0, 256)
        )
        # Normalize
        total = hist.sum()
        if total > 0:
            hist = hist.astype(np.float64) / total
        return hist

    @staticmethod
    def _correlate(
        hist_a: np.ndarray, hist_b: np.ndarray
    ) -> float:
        """Compute Pearson correlation between histograms.

        Returns:
            Correlation coefficient in [-1, 1].
            Values near 1 = similar, near 0 = different.
        """
        a = hist_a - hist_a.mean()
        b = hist_b - hist_b.mean()
        denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
        if denom < 1e-10:
            return 0.0
        return float((a * b).sum() / denom)
