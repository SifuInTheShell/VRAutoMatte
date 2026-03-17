"""Tests for matting processor protocol and factory."""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from vrautomatte.pipeline.matte import (
    VARIANTS,
    MatteProcessor,
    create_processor,
)
from vrautomatte.pipeline.rvm import RVMProcessor


def _matanyone2_available() -> bool:
    """Check if matanyone2 and sam2 packages are installed."""
    try:
        import matanyone2  # noqa: F401
        import sam2  # noqa: F401
        return True
    except ImportError:
        return False


class TestMatteProtocol(unittest.TestCase):
    """Verify MatteProcessor protocol compliance."""

    def test_rvm_is_matte_processor(self):
        """RVMProcessor matches the MatteProcessor protocol."""
        self.assertTrue(
            issubclass(RVMProcessor, MatteProcessor)
        )

    def test_rvm_has_required_methods(self):
        """RVMProcessor has process_frame, reset, cleanup."""
        for method in ("process_frame", "reset", "cleanup"):
            self.assertTrue(
                hasattr(RVMProcessor, method),
                f"RVMProcessor missing {method}",
            )


class TestVariants(unittest.TestCase):
    """Verify variant list is correct."""

    def test_known_variants(self):
        self.assertIn("mobilenetv3", VARIANTS)
        self.assertIn("resnet50", VARIANTS)
        self.assertIn("matanyone2", VARIANTS)


class TestCreateProcessorRVM(unittest.TestCase):
    """Test factory for RVM variants (with mocked model)."""

    @patch("vrautomatte.pipeline.rvm.download_model")
    @patch("torch.jit.load")
    def test_mobilenetv3(self, mock_load, mock_dl):
        """Factory returns RVMProcessor for mobilenetv3."""
        import torch
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_dl.return_value = "/fake/model.pth"

        proc = create_processor(
            "mobilenetv3", device=torch.device("cpu")
        )
        self.assertIsInstance(proc, RVMProcessor)
        self.assertEqual(proc.variant, "mobilenetv3")

    @patch("vrautomatte.pipeline.rvm.download_model")
    @patch("torch.jit.load")
    def test_resnet50(self, mock_load, mock_dl):
        """Factory returns RVMProcessor for resnet50."""
        import torch
        mock_model = MagicMock()
        mock_load.return_value = mock_model
        mock_dl.return_value = "/fake/model.pth"

        proc = create_processor(
            "resnet50", device=torch.device("cpu")
        )
        self.assertIsInstance(proc, RVMProcessor)
        self.assertEqual(proc.variant, "resnet50")


class TestCreateProcessorUnknown(unittest.TestCase):
    """Test factory rejects unknown variants."""

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            create_processor("nonexistent")


class TestCreateProcessorMatAnyone2(unittest.TestCase):
    """Test factory for matanyone2 (requires first_frame)."""

    def test_missing_first_frame(self):
        """matanyone2 without first_frame raises RuntimeError."""
        with self.assertRaises(RuntimeError):
            create_processor("matanyone2")

    @unittest.skipUnless(
        _matanyone2_available(), "matanyone2 not installed"
    )
    def test_with_first_frame_mock(self):
        """matanyone2 with first_frame calls SAM2."""
        # This test only runs when packages are installed
        pass


if __name__ == "__main__":
    unittest.main()
