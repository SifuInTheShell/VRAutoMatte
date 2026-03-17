"""Integration tests for the MatAnyone 2 pipeline.

These tests verify the factory, runner wiring, and protocol
conformance. Tests that require actual model weights are skipped
when models aren't downloaded.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from vrautomatte.pipeline.matte import (
    MatteProcessor,
    create_processor,
)


def _matanyone2_available() -> bool:
    """Check if matanyone2 and sam2 are installed."""
    try:
        import matanyone2  # noqa: F401
        import sam2  # noqa: F401
        return True
    except ImportError:
        return False


class TestRVMRegressionFactory(unittest.TestCase):
    """Verify RVM still works through the new factory."""

    @patch("vrautomatte.pipeline.rvm.download_model")
    @patch("torch.jit.load")
    def test_mobilenetv3_via_factory(self, mock_load, mock_dl):
        """Factory-created mobilenetv3 is a valid processor."""
        mock_load.return_value = MagicMock()
        mock_dl.return_value = "/fake/model.pth"

        proc = create_processor(
            "mobilenetv3", device=torch.device("cpu")
        )
        self.assertIsInstance(proc, MatteProcessor)
        self.assertTrue(hasattr(proc, "process_frame"))
        self.assertTrue(hasattr(proc, "reset"))
        self.assertTrue(hasattr(proc, "cleanup"))

    @patch("vrautomatte.pipeline.rvm.download_model")
    @patch("torch.jit.load")
    def test_resnet50_via_factory(self, mock_load, mock_dl):
        """Factory-created resnet50 is a valid processor."""
        mock_load.return_value = MagicMock()
        mock_dl.return_value = "/fake/model.pth"

        proc = create_processor(
            "resnet50", device=torch.device("cpu")
        )
        self.assertIsInstance(proc, MatteProcessor)


class TestMatAnyone2Factory(unittest.TestCase):
    """Verify MatAnyone 2 factory path."""

    def test_requires_first_frame(self):
        """matanyone2 without first_frame raises."""
        with self.assertRaises(RuntimeError):
            create_processor("matanyone2")

    @unittest.skipUnless(
        _matanyone2_available(),
        "matanyone2/sam2 not installed",
    )
    def test_matanyone2_protocol_conformance(self):
        """MatAnyone2Processor matches protocol."""
        from vrautomatte.pipeline.matanyone2 import (
            MatAnyone2Processor,
        )
        self.assertTrue(
            issubclass(MatAnyone2Processor, MatteProcessor)
        )


class TestRunnerImport(unittest.TestCase):
    """Verify runner uses new factory correctly."""

    def test_runner_imports(self):
        """Runner module imports without error."""
        from vrautomatte.pipeline.runner import (
            Pipeline,
            PipelineConfig,
        )
        cfg = PipelineConfig(model_variant="matanyone2")
        self.assertEqual(cfg.model_variant, "matanyone2")

    def test_runner_config_variants(self):
        """PipelineConfig accepts all known variants."""
        from vrautomatte.pipeline.runner import PipelineConfig
        for v in ("mobilenetv3", "resnet50", "matanyone2"):
            cfg = PipelineConfig(model_variant=v)
            self.assertEqual(cfg.model_variant, v)


class TestCPUFallback(unittest.TestCase):
    """Verify CPU detection and warning path."""

    def test_get_device_returns_valid(self):
        """get_device returns a valid torch.device."""
        from vrautomatte.utils.gpu import get_device
        device = get_device()
        self.assertIsInstance(device, torch.device)
        self.assertIn(
            device.type, ("cpu", "cuda", "mps")
        )

    def test_get_device_info_has_name(self):
        """get_device_info returns a dict with name."""
        from vrautomatte.utils.gpu import get_device_info
        info = get_device_info()
        self.assertIn("device", info)
        self.assertIn("name", info)


class TestSettingsPersistence(unittest.TestCase):
    """Verify settings handle new model variant index."""

    def test_defaults_include_model_variant(self):
        """Default settings include model_variant."""
        from vrautomatte.utils.settings import _DEFAULTS
        self.assertIn("model_variant", _DEFAULTS)
        self.assertEqual(_DEFAULTS["model_variant"], 0)


if __name__ == "__main__":
    unittest.main()
