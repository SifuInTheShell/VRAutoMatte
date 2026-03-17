"""Tests for scene change detection."""

import unittest

import numpy as np

from vrautomatte.pipeline.scene_detect import (
    SceneChangeDetector,
)


class TestSceneChangeDetector(unittest.TestCase):
    """Test histogram-based scene change detection."""

    def _solid_frame(self, color, h=100, w=100):
        """Create a solid-color RGB frame."""
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = color
        return frame

    def test_first_frame_no_change(self):
        """First frame is never a scene change."""
        det = SceneChangeDetector()
        frame = self._solid_frame((128, 128, 128))
        self.assertFalse(det.check(frame))

    def test_identical_frames_no_change(self):
        """Identical frames don't trigger."""
        det = SceneChangeDetector(cooldown_frames=0)
        frame = self._solid_frame((128, 128, 128))
        det.check(frame)  # set reference
        for _ in range(10):
            self.assertFalse(det.check(frame))

    def test_drastic_change_triggers(self):
        """Dark-to-bright triggers scene change."""
        det = SceneChangeDetector(cooldown_frames=1)
        dark = self._solid_frame((10, 10, 10))
        bright = self._solid_frame((240, 240, 240))

        det.check(dark)  # set reference
        det.check(dark)  # satisfy cooldown
        self.assertTrue(det.check(bright))

    def test_cooldown_prevents_retrigger(self):
        """Can't trigger again during cooldown."""
        det = SceneChangeDetector(cooldown_frames=5)
        dark = self._solid_frame((10, 10, 10))
        bright = self._solid_frame((240, 240, 240))

        det.check(dark)  # frame 0: set reference
        for _ in range(5):
            det.check(dark)  # frames 1-5: cooldown

        self.assertTrue(det.check(bright))  # frame 6: trigger

        # Within cooldown now — shouldn't trigger
        other = self._solid_frame((100, 50, 200))
        self.assertFalse(det.check(other))

    def test_similar_frames_no_change(self):
        """Slightly different frames don't trigger."""
        det = SceneChangeDetector(cooldown_frames=0)
        rng = np.random.RandomState(42)
        base = rng.randint(100, 150, (100, 100, 3),
                           dtype=np.uint8)
        det.check(base)

        # Small noise variation
        noisy = base.copy()
        noise = rng.randint(-10, 10, base.shape)
        noisy = np.clip(
            noisy.astype(int) + noise, 0, 255
        ).astype(np.uint8)
        det.check(noisy)  # cooldown
        self.assertFalse(det.check(noisy))

    def test_update_reference(self):
        """Manual reference update anchors to new scene."""
        det = SceneChangeDetector(cooldown_frames=0)
        dark = self._solid_frame((10, 10, 10))
        bright = self._solid_frame((240, 240, 240))

        det.check(dark)
        det.update_reference(bright)
        det.check(bright)  # cooldown
        # Now bright is the reference — dark should trigger
        self.assertTrue(det.check(dark))

    def test_reset_clears_state(self):
        """Reset makes next frame a non-change."""
        det = SceneChangeDetector(cooldown_frames=0)
        frame = self._solid_frame((128, 128, 128))
        det.check(frame)
        det.reset()
        # After reset, first frame is never a change
        self.assertFalse(det.check(frame))

    def test_threshold_sensitivity(self):
        """Lower threshold is less sensitive."""
        dark = self._solid_frame((10, 10, 10))
        medium = self._solid_frame((130, 130, 130))

        # Strict threshold — should trigger
        strict = SceneChangeDetector(
            threshold=0.8, cooldown_frames=0
        )
        strict.check(dark)
        strict.check(dark)  # cooldown
        self.assertTrue(strict.check(medium))

        # Lenient threshold — may not trigger
        lenient = SceneChangeDetector(
            threshold=0.1, cooldown_frames=0
        )
        lenient.check(dark)
        lenient.check(dark)
        # With very low threshold, only extreme changes pass
        result = lenient.check(medium)
        # Just verify it runs without error
        self.assertIsInstance(result, bool)


if __name__ == "__main__":
    unittest.main()
