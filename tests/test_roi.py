"""Tests for ROI-restricted matting (pipeline/roi.py)."""

import unittest

import numpy as np

from vrautomatte.pipeline.roi import ROICropper, _matte_bbox


def _frame_with_subject(h, w, y0, y1, x0, x1):
    """Gray frame with a bright rectangle 'subject'."""
    f = np.zeros((h, w, 3), dtype=np.uint8)
    f[y0:y1, x0:x1] = 200
    return f


class BrightnessProcessor:
    """Dummy per-frame processor: matte = bright pixels.

    Records the shape of every frame it receives so tests can
    assert whether it saw a crop or the full frame.
    """

    def __init__(self):
        self.seen = []
        self.resets = 0

    def process_frame(self, frame):
        self.seen.append(frame.shape[:2])
        return (
            frame.mean(axis=2) > 100
        ).astype(np.uint8) * 255

    def reset(self):
        self.resets += 1

    def cleanup(self):
        pass


class PairBrightnessProcessor(BrightnessProcessor):
    supports_pair = True

    def process_frame_pair(self, left, right):
        return (
            self.process_frame(left),
            self.process_frame(right),
        )


class TestMatteBbox(unittest.TestCase):
    def test_empty(self):
        self.assertIsNone(
            _matte_bbox(np.zeros((64, 64), np.uint8))
        )

    def test_covers_subject(self):
        m = np.zeros((128, 128), np.uint8)
        m[40:80, 32:96] = 255
        x0, y0, x1, y1 = _matte_bbox(m)
        self.assertLessEqual(x0, 32)
        self.assertLessEqual(y0, 40)
        self.assertGreaterEqual(x1, 92)
        self.assertGreaterEqual(y1, 76)


class TestROICropper(unittest.TestCase):
    H, W = 512, 512

    def test_first_frame_full_then_crops(self):
        inner = BrightnessProcessor()
        roi = ROICropper(inner)
        frame = _frame_with_subject(
            self.H, self.W, 200, 300, 200, 300
        )
        m1 = roi.process_frame(frame)
        self.assertEqual(m1.shape, (self.H, self.W))
        self.assertEqual(inner.seen[0], (self.H, self.W))

        m2 = roi.process_frame(frame)
        self.assertEqual(m2.shape, (self.H, self.W))
        # Second frame must have been a crop
        ch, cw = inner.seen[1]
        self.assertLess(ch * cw, self.H * self.W * 0.5)
        # Matte content identical to full-frame result
        np.testing.assert_array_equal(m1, m2)

    def test_reanchors_when_subject_moves(self):
        inner = BrightnessProcessor()
        roi = ROICropper(inner)
        f1 = _frame_with_subject(
            self.H, self.W, 100, 200, 100, 200
        )
        roi.process_frame(f1)
        roi.process_frame(f1)
        win_before = roi._window
        # Subject jumps to the far corner
        f2 = _frame_with_subject(
            self.H, self.W, 350, 450, 350, 450
        )
        # Frame after jump: window still old, matte empty in
        # window -> subject lost -> full frame on next call
        roi.process_frame(f2)
        m = roi.process_frame(f2)
        self.assertEqual(
            m[400, 400], 255,
            "subject must be re-found after moving",
        )
        self.assertNotEqual(roi._window, win_before)

    def test_full_frame_for_large_subject(self):
        inner = BrightnessProcessor()
        roi = ROICropper(inner)
        big = _frame_with_subject(
            self.H, self.W, 10, 500, 10, 500
        )
        roi.process_frame(big)
        roi.process_frame(big)
        # Subject covers most of frame -> ROI disabled
        self.assertIsNone(roi._window)
        self.assertEqual(
            inner.seen[1], (self.H, self.W)
        )

    def test_pair_mode(self):
        inner = PairBrightnessProcessor()
        roi = ROICropper(inner)
        self.assertTrue(roi.supports_pair)
        left = _frame_with_subject(
            self.H, self.W, 200, 300, 200, 300
        )
        right = _frame_with_subject(
            self.H, self.W, 200, 300, 210, 310
        )
        lm1, rm1 = roi.process_frame_pair(left, right)
        lm2, rm2 = roi.process_frame_pair(left, right)
        self.assertEqual(lm2.shape, (self.H, self.W))
        # Second call went through a shared crop window
        ch, cw = inner.seen[-1]
        self.assertLess(ch * cw, self.H * self.W * 0.5)
        self.assertEqual(rm2[250, 305], 255)
        self.assertEqual(lm2[250, 205], 255)

    def test_reseed_preferred_over_reset(self):
        class ReseedProcessor(BrightnessProcessor):
            def __init__(self):
                super().__init__()
                self.reseeds = 0

            def reseed(self, frame, mask):
                self.reseeds += 1
                return self.process_frame(frame)

        inner = ReseedProcessor()
        roi = ROICropper(inner)
        f1 = _frame_with_subject(
            self.H, self.W, 100, 200, 100, 200
        )
        roi.process_frame(f1)   # full frame, anchors window
        roi.process_frame(f1)   # window move pending -> reseed
        self.assertGreaterEqual(inner.reseeds, 1)
        self.assertEqual(inner.resets, 0)


if __name__ == "__main__":
    unittest.main()
