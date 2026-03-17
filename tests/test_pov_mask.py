"""Tests for POV mask selection heuristic."""

import unittest

import numpy as np

from vrautomatte.pipeline.matanyone2 import (
    _mask_center_of_mass,
    _select_non_pov_mask,
)


class TestMaskCenterOfMass(unittest.TestCase):
    """Test center of mass calculation."""

    def test_full_mask(self):
        """Full mask center is at frame center."""
        seg = np.ones((100, 200), dtype=bool)
        cy, cx = _mask_center_of_mass(seg)
        self.assertAlmostEqual(cy, 49.5)
        self.assertAlmostEqual(cx, 99.5)

    def test_top_left_mask(self):
        """Top-left mask has low center coords."""
        seg = np.zeros((100, 200), dtype=bool)
        seg[:20, :40] = True
        cy, cx = _mask_center_of_mass(seg)
        self.assertLess(cy, 20)
        self.assertLess(cx, 40)

    def test_empty_mask(self):
        """Empty mask returns (0, 0)."""
        seg = np.zeros((100, 200), dtype=bool)
        cy, cx = _mask_center_of_mass(seg)
        self.assertEqual(cy, 0)
        self.assertEqual(cx, 0)


class TestSelectNonPOVMask(unittest.TestCase):
    """Test POV mask selection heuristic."""

    def _make_mask(self, h, w, y0, y1, x0, x1):
        """Create a mask dict with the given region."""
        seg = np.zeros((h, w), dtype=bool)
        seg[y0:y1, x0:x1] = True
        return {
            "segmentation": seg,
            "area": int(seg.sum()),
        }

    def test_prefers_centered_over_bottom(self):
        """Centered mask beats bottom-edge mask."""
        h, w = 100, 100
        # Centered person
        centered = self._make_mask(h, w, 20, 60, 30, 70)
        # Bottom POV body
        bottom = self._make_mask(h, w, 70, 100, 10, 90)

        result = _select_non_pov_mask(
            [centered, bottom], (h, w, 3)
        )
        # Should select centered, not bottom
        np.testing.assert_array_equal(
            result,
            centered["segmentation"].astype(np.uint8) * 255,
        )

    def test_penalizes_very_large_masks(self):
        """Very large mask (>60% frame) is penalized."""
        h, w = 100, 100
        # Smaller centered person
        person = self._make_mask(h, w, 20, 70, 30, 70)
        # Huge POV body covering most of frame
        huge = self._make_mask(h, w, 0, 100, 0, 100)

        result = _select_non_pov_mask(
            [person, huge], (h, w, 3)
        )
        np.testing.assert_array_equal(
            result,
            person["segmentation"].astype(np.uint8) * 255,
        )

    def test_single_mask_returns_it(self):
        """With only one mask, returns it regardless."""
        h, w = 100, 100
        only = self._make_mask(h, w, 80, 100, 0, 100)
        result = _select_non_pov_mask(
            [only], (h, w, 3)
        )
        self.assertEqual(result.shape, (h, w))
        self.assertTrue(result.max() > 0)


if __name__ == "__main__":
    unittest.main()
