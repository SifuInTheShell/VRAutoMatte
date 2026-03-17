"""Tests for SBS stereo utilities."""

import unittest

import numpy as np

from vrautomatte.utils.sbs import (
    detect_sbs,
    merge_frames,
    merge_mattes,
    split_frame,
    split_matte,
)


class TestDetectSBS(unittest.TestCase):
    """Test SBS auto-detection heuristic."""

    def test_standard_sbs_3840x1920(self):
        """4K SBS (2:1 ratio) is detected."""
        self.assertTrue(detect_sbs(3840, 1920))

    def test_standard_sbs_5760x2880(self):
        """5.7K SBS is detected."""
        self.assertTrue(detect_sbs(5760, 2880))

    def test_wide_sbs_4096x1800(self):
        """Wide SBS (>2:1) is detected."""
        self.assertTrue(detect_sbs(4096, 1800))

    def test_non_sbs_1920x1080(self):
        """Standard 16:9 is NOT SBS."""
        self.assertFalse(detect_sbs(1920, 1080))

    def test_non_sbs_square(self):
        """Square video is NOT SBS."""
        self.assertFalse(detect_sbs(1024, 1024))

    def test_zero_height(self):
        """Zero height returns False."""
        self.assertFalse(detect_sbs(1920, 0))

    def test_borderline_1_9(self):
        """Ratio exactly 1.9 is SBS."""
        self.assertTrue(detect_sbs(1900, 1000))

    def test_just_below_1_9(self):
        """Ratio 1.89 is NOT SBS."""
        self.assertFalse(detect_sbs(1890, 1000))


class TestSplitMergeFrames(unittest.TestCase):
    """Test frame splitting and merging."""

    def test_split_frame_shapes(self):
        """Split produces two half-width arrays."""
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        left, right = split_frame(frame)
        self.assertEqual(left.shape, (100, 100, 3))
        self.assertEqual(right.shape, (100, 100, 3))

    def test_split_preserves_content(self):
        """Left gets left half, right gets right half."""
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        frame[:, :10, 0] = 255  # left half red
        frame[:, 10:, 1] = 255  # right half green
        left, right = split_frame(frame)
        self.assertEqual(left[:, :, 0].mean(), 255)
        self.assertEqual(right[:, :, 1].mean(), 255)

    def test_merge_round_trip(self):
        """Split then merge recovers original."""
        frame = np.random.randint(
            0, 255, (50, 100, 3), dtype=np.uint8
        )
        left, right = split_frame(frame)
        merged = merge_frames(left, right)
        np.testing.assert_array_equal(merged, frame)

    def test_split_is_copy(self):
        """Modifying split doesn't affect original."""
        frame = np.zeros((10, 20, 3), dtype=np.uint8)
        left, right = split_frame(frame)
        left[:] = 255
        self.assertEqual(frame.max(), 0)


class TestSplitMergeMattes(unittest.TestCase):
    """Test matte (grayscale) splitting and merging."""

    def test_split_matte_shapes(self):
        """Split produces two half-width arrays."""
        matte = np.zeros((100, 200), dtype=np.uint8)
        left, right = split_matte(matte)
        self.assertEqual(left.shape, (100, 100))
        self.assertEqual(right.shape, (100, 100))

    def test_merge_matte_round_trip(self):
        """Split then merge recovers original."""
        matte = np.random.randint(
            0, 255, (50, 100), dtype=np.uint8
        )
        left, right = split_matte(matte)
        merged = merge_mattes(left, right)
        np.testing.assert_array_equal(merged, matte)


if __name__ == "__main__":
    unittest.main()
