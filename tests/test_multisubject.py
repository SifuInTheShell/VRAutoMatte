"""Tests for multi-subject tracking (SAM2Matting extension)."""

import unittest

import numpy as np

from vrautomatte.pipeline.sam2_masks import (
    _select_person_masks_multi,
)
from vrautomatte.pipeline.sam2matting import alpha_to_planes


def _mask_dict(h, w, y0, y1, x0, x1):
    seg = np.zeros((h, w), dtype=bool)
    seg[y0:y1, x0:x1] = True
    return {"segmentation": seg, "area": int(seg.sum())}


class TestSelectPersonMasksMulti(unittest.TestCase):
    H, W = 480, 640
    SHAPE = (480, 640, 3)

    def test_two_distinct_people(self):
        masks = [
            # person-shaped, centered-ish, tall
            _mask_dict(self.H, self.W, 100, 400, 150, 250),
            _mask_dict(self.H, self.W, 100, 400, 400, 500),
        ]
        picked = _select_person_masks_multi(
            masks, self.SHAPE, max_people=4
        )
        self.assertEqual(len(picked), 2)

    def test_nested_masks_deduped(self):
        person = _mask_dict(
            self.H, self.W, 100, 400, 150, 250
        )
        torso = _mask_dict(
            self.H, self.W, 150, 300, 160, 240
        )  # inside person
        other = _mask_dict(
            self.H, self.W, 100, 400, 400, 500
        )
        picked = _select_person_masks_multi(
            [person, torso, other], self.SHAPE,
            max_people=4,
        )
        # torso overlaps person -> deduped
        self.assertEqual(len(picked), 2)

    def test_max_people_cap(self):
        masks = [
            _mask_dict(
                self.H, self.W, 100, 400,
                50 + i * 140, 140 + i * 140,
            )
            for i in range(4)
        ]
        picked = _select_person_masks_multi(
            masks, self.SHAPE, max_people=2
        )
        self.assertEqual(len(picked), 2)

    def test_always_returns_at_least_one(self):
        # Single low-scoring blob
        masks = [_mask_dict(self.H, self.W, 0, 20, 0, 20)]
        picked = _select_person_masks_multi(
            masks, self.SHAPE, max_people=3
        )
        self.assertEqual(len(picked), 1)

    def test_masks_are_uint8_255(self):
        masks = [
            _mask_dict(self.H, self.W, 100, 400, 150, 250)
        ]
        picked = _select_person_masks_multi(
            masks, self.SHAPE
        )
        self.assertEqual(picked[0].dtype, np.uint8)
        self.assertEqual(picked[0].max(), 255)


class TestAlphaToPlanes(unittest.TestCase):
    def test_hw_input(self):
        a = np.random.rand(32, 48).astype(np.float32)
        p = alpha_to_planes(a)
        self.assertEqual(p.shape, (1, 32, 48))
        self.assertEqual(p.dtype, np.uint8)

    def test_n1hw_input(self):
        a = np.random.rand(3, 1, 32, 48).astype(np.float32)
        p = alpha_to_planes(a)
        self.assertEqual(p.shape, (3, 32, 48))

    def test_nhw_input(self):
        a = np.random.rand(2, 32, 48).astype(np.float32)
        p = alpha_to_planes(a)
        self.assertEqual(p.shape, (2, 32, 48))

    def test_0_255_range_passthrough(self):
        a = np.full((32, 48), 200.0, dtype=np.float32)
        p = alpha_to_planes(a)
        self.assertEqual(p[0, 0, 0], 200)

    def test_0_1_range_scaled(self):
        a = np.full((32, 48), 0.5, dtype=np.float32)
        p = alpha_to_planes(a)
        self.assertIn(p[0, 0, 0], (127, 128))


class TestHandoff(unittest.TestCase):
    """_update_handoff logic via a bare processor instance."""

    def _bare(self, masks):
        from vrautomatte.pipeline.sam2matting import (
            SAM2MattingProcessor,
        )
        proc = SAM2MattingProcessor.__new__(
            SAM2MattingProcessor
        )
        proc._next_masks = masks
        return proc

    def test_per_object_handoff(self):
        m0 = np.zeros((16, 16), np.uint8)
        m1 = np.zeros((16, 16), np.uint8)
        proc = self._bare([m0, m1])
        planes = np.zeros((2, 16, 16), np.uint8)
        planes[0, 2:6, 2:6] = 255
        planes[1, 10:14, 10:14] = 255
        proc._update_handoff(planes)
        self.assertEqual(proc._next_masks[0][3, 3], 255)
        self.assertEqual(proc._next_masks[1][12, 12], 255)
        self.assertEqual(proc._next_masks[0][12, 12], 0)

    def test_lost_subject_keeps_previous_mask(self):
        m0 = np.zeros((16, 16), np.uint8)
        m1 = np.full((16, 16), 255, np.uint8)
        proc = self._bare([m0, m1])
        planes = np.zeros((2, 16, 16), np.uint8)
        planes[0, 2:6, 2:6] = 255
        # object 2 empty -> keeps previous full mask
        proc._update_handoff(planes)
        self.assertEqual(proc._next_masks[1][8, 8], 255)

    def test_combined_alpha_fallback(self):
        m = np.zeros((16, 16), np.uint8)
        proc = self._bare([m, m.copy()])
        planes = np.zeros((1, 16, 16), np.uint8)
        planes[0, 4:8, 4:8] = 255
        proc._update_handoff(planes)
        self.assertEqual(len(proc._next_masks), 1)
        self.assertEqual(proc._next_masks[0][5, 5], 255)


if __name__ == "__main__":
    unittest.main()
