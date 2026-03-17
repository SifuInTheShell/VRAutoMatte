"""Tests for POV mask selection and exclusion wrapper."""

import unittest

import numpy as np

from vrautomatte.pipeline.sam2_masks import (
    _mask_center_of_mass,
    _select_non_pov_mask,
    _select_person_mask,
    _select_pov_body_mask,
)
from vrautomatte.pipeline.matte import POVExclusionProcessor


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
    """Test non-POV person selection (subject facing camera)."""

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
        centered = self._make_mask(h, w, 20, 60, 30, 70)
        bottom = self._make_mask(h, w, 70, 100, 10, 90)

        result = _select_non_pov_mask(
            [centered, bottom], (h, w, 3)
        )
        np.testing.assert_array_equal(
            result,
            centered["segmentation"].astype(np.uint8) * 255,
        )

    def test_penalizes_very_large_masks(self):
        """Very large mask (>60% frame) is penalized."""
        h, w = 100, 100
        person = self._make_mask(h, w, 20, 70, 30, 70)
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


class TestSelectPOVBodyMask(unittest.TestCase):
    """Test POV body selection (inverse of non-POV)."""

    def _make_mask(self, h, w, y0, y1, x0, x1):
        seg = np.zeros((h, w), dtype=bool)
        seg[y0:y1, x0:x1] = True
        return {
            "segmentation": seg,
            "area": int(seg.sum()),
        }

    def test_picks_bottom_over_centered(self):
        """POV body selector picks the bottom-heavy mask."""
        h, w = 100, 100
        centered = self._make_mask(h, w, 20, 60, 30, 70)
        bottom = self._make_mask(h, w, 70, 100, 10, 90)

        result = _select_pov_body_mask(
            [centered, bottom], (h, w, 3)
        )
        # Bottom mask should be selected (as POV body)
        # Result may be dilated, so check overlap
        bottom_seg = bottom["segmentation"]
        overlap = (result > 0) & bottom_seg
        self.assertTrue(
            overlap.sum() > bottom_seg.sum() * 0.9,
            "POV body mask should overlap with bottom mask",
        )

    def test_single_mask_returns_it(self):
        """Single mask is used as POV body."""
        h, w = 100, 100
        only = self._make_mask(h, w, 30, 70, 30, 70)
        result = _select_pov_body_mask(
            [only], (h, w, 3)
        )
        self.assertEqual(result.shape, (h, w))
        self.assertTrue(result.max() > 0)


class _FakeProcessor:
    """Fake processor for testing POVExclusionProcessor."""

    def __init__(self, matte_value: int = 200):
        self._val = matte_value
        self._reset_count = 0
        self._cleanup_count = 0

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        return np.full((h, w), self._val, dtype=np.uint8)

    def reset(self):
        self._reset_count += 1

    def cleanup(self):
        self._cleanup_count += 1


class TestPOVExclusionProcessor(unittest.TestCase):
    """Test the POV exclusion wrapper."""

    def test_subtracts_mask_region(self):
        """Alpha is zeroed in the POV body region."""
        h, w = 100, 100
        body_mask = np.zeros((h, w), dtype=np.uint8)
        body_mask[80:100, :] = 255  # bottom 20 rows

        inner = _FakeProcessor(matte_value=200)
        proc = POVExclusionProcessor(inner, body_mask)

        frame = np.zeros((h, w, 3), dtype=np.uint8)
        result = proc.process_frame(frame)

        # Non-masked region should keep value
        self.assertEqual(result[50, 50], 200)
        # Masked region should be zero
        self.assertEqual(result[90, 50], 0)

    def test_delegates_reset(self):
        """Reset is forwarded to inner processor."""
        inner = _FakeProcessor()
        proc = POVExclusionProcessor(
            inner,
            np.zeros((10, 10), dtype=np.uint8),
        )
        proc.reset()
        self.assertEqual(inner._reset_count, 1)

    def test_delegates_cleanup(self):
        """Cleanup is forwarded to inner processor."""
        inner = _FakeProcessor()
        proc = POVExclusionProcessor(
            inner,
            np.zeros((10, 10), dtype=np.uint8),
        )
        proc.cleanup()
        self.assertEqual(inner._cleanup_count, 1)


class TestSelectPersonMask(unittest.TestCase):
    """Test person mask selection heuristics."""

    def _make_mask(self, seg):
        return {
            "segmentation": seg,
            "area": int(seg.sum()),
        }

    def test_prefers_centered_over_largest(self):
        """Centered person-sized mask beats large background."""
        h, w = 100, 200
        frame_shape = (h, w, 3)

        # Large background mask (70% of frame)
        bg = np.ones((h, w), dtype=bool)
        bg[20:80, 60:140] = False  # hole where person is

        # Small centered person mask (~15% of frame)
        person = np.zeros((h, w), dtype=bool)
        person[15:85, 70:130] = True

        masks = [self._make_mask(bg), self._make_mask(person)]
        result = _select_person_mask(masks, frame_shape)

        # Should pick the person, not the background
        person_coverage = person.sum()
        result_coverage = (result > 0).sum()
        self.assertAlmostEqual(
            result_coverage, person_coverage, delta=50
        )

    def test_prefers_tall_aspect_ratio(self):
        """Tall mask (person-shaped) beats wide mask."""
        h, w = 200, 200
        frame_shape = (h, w, 3)

        # Wide mask (landscape shape)
        wide = np.zeros((h, w), dtype=bool)
        wide[80:120, 20:180] = True  # 40x160

        # Tall mask (person shape)
        tall = np.zeros((h, w), dtype=bool)
        tall[20:180, 80:120] = True  # 160x40

        masks = [self._make_mask(wide), self._make_mask(tall)]
        result = _select_person_mask(masks, frame_shape)

        # Should prefer the tall mask
        tall_coverage = tall.sum()
        result_coverage = (result > 0).sum()
        self.assertAlmostEqual(
            result_coverage, tall_coverage, delta=50
        )

    def test_rejects_huge_background(self):
        """Mask covering >60% of frame is penalized."""
        h, w = 100, 200
        frame_shape = (h, w, 3)

        # Giant mask (90% of frame) — background
        giant = np.ones((h, w), dtype=bool)

        # Small person mask (10% of frame)
        person = np.zeros((h, w), dtype=bool)
        person[30:70, 80:120] = True

        masks = [self._make_mask(giant), self._make_mask(person)]
        result = _select_person_mask(masks, frame_shape)

        person_coverage = person.sum()
        result_coverage = (result > 0).sum()
        self.assertAlmostEqual(
            result_coverage, person_coverage, delta=50
        )


if __name__ == "__main__":
    unittest.main()
