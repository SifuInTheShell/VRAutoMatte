"""Tests for the ONNX Runtime RVM processor.

Mocks onnxruntime so tests run without the package installed.
"""

import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

# ── Inject a fake onnxruntime module before any imports ──────────
_mock_ort = types.ModuleType("onnxruntime")
_mock_ort.get_available_providers = MagicMock(
    return_value=["CPUExecutionProvider"]
)
_mock_ort.InferenceSession = MagicMock()
_mock_ort.SessionOptions = MagicMock
_mock_ort.GraphOptimizationLevel = MagicMock()
_mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
sys.modules.setdefault("onnxruntime", _mock_ort)


class TestSelectProviders(unittest.TestCase):
    """Verify provider selection logic."""

    def test_directml_preferred(self):
        """DirectML listed first when available."""
        _mock_ort.get_available_providers.return_value = [
            "DmlExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        from vrautomatte.pipeline.rvm_onnx import (
            _select_providers,
        )
        result = _select_providers()
        self.assertEqual(result[0], "DmlExecutionProvider")
        self.assertEqual(result[-1], "CPUExecutionProvider")

    def test_cuda_when_no_directml(self):
        """CUDA selected when DirectML is absent."""
        _mock_ort.get_available_providers.return_value = [
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        from vrautomatte.pipeline.rvm_onnx import (
            _select_providers,
        )
        result = _select_providers()
        self.assertEqual(result[0], "CUDAExecutionProvider")
        self.assertNotIn("DmlExecutionProvider", result)

    def test_cpu_only_fallback(self):
        """CPU-only when no GPU providers available."""
        _mock_ort.get_available_providers.return_value = [
            "CPUExecutionProvider",
        ]
        from vrautomatte.pipeline.rvm_onnx import (
            _select_providers,
        )
        result = _select_providers()
        self.assertEqual(result, ["CPUExecutionProvider"])

    def test_coreml_on_mac(self):
        """CoreML selected on macOS."""
        _mock_ort.get_available_providers.return_value = [
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        from vrautomatte.pipeline.rvm_onnx import (
            _select_providers,
        )
        result = _select_providers()
        self.assertEqual(result[0], "CoreMLExecutionProvider")


class TestEmptyRec(unittest.TestCase):
    """Verify recurrent state initialisation."""

    def test_independent_arrays(self):
        """Each rec array is independent (not shared refs)."""
        from vrautomatte.pipeline.rvm_onnx import _empty_rec
        rec = _empty_rec()
        self.assertEqual(len(rec), 4)
        rec[0][0, 0, 0, 0] = 42.0
        for i in range(1, 4):
            self.assertEqual(
                rec[i][0, 0, 0, 0], 0.0,
                f"rec[{i}] was mutated by writing to rec[0]",
            )

    def test_correct_shape_and_dtype(self):
        """Rec arrays are [1,1,1,1] float32."""
        from vrautomatte.pipeline.rvm_onnx import _empty_rec
        rec = _empty_rec()
        for arr in rec:
            self.assertEqual(arr.shape, (1, 1, 1, 1))
            self.assertEqual(arr.dtype, np.float32)


def _make_mock_session():
    """Create a mock InferenceSession that returns valid outputs."""
    session = MagicMock()
    session.get_providers.return_value = [
        "CPUExecutionProvider"
    ]

    def mock_run(output_names, inputs):
        h, w = inputs["src"].shape[2], inputs["src"].shape[3]
        return [
            np.zeros([1, 3, h, w], dtype=np.float32),  # fgr
            np.full(
                [1, 1, h, w], 0.8, dtype=np.float32
            ),  # pha
            np.zeros([1, 1, 1, 1], dtype=np.float32),  # r1o
            np.zeros([1, 1, 1, 1], dtype=np.float32),  # r2o
            np.zeros([1, 1, 1, 1], dtype=np.float32),  # r3o
            np.zeros([1, 1, 1, 1], dtype=np.float32),  # r4o
        ]

    session.run.side_effect = mock_run
    return session


class TestRVMOnnxProcessor(unittest.TestCase):
    """Test processor with mocked ONNX session."""

    @patch(
        "vrautomatte.pipeline.rvm_onnx._download_onnx_model"
    )
    @patch(
        "vrautomatte.pipeline.rvm_onnx._select_providers"
    )
    def test_process_frame_returns_matte(
        self, mock_prov, mock_dl
    ):
        """process_frame returns uint8 HxW matte."""
        mock_dl.return_value = "/fake/model.onnx"
        mock_prov.return_value = ["CPUExecutionProvider"]
        _mock_ort.InferenceSession = MagicMock(
            return_value=_make_mock_session()
        )

        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )
        proc = RVMOnnxProcessor(variant="mobilenetv3")

        frame = np.random.randint(
            0, 255, (480, 640, 3), dtype=np.uint8
        )
        matte = proc.process_frame(frame)

        self.assertEqual(matte.shape, (480, 640))
        self.assertEqual(matte.dtype, np.uint8)
        # 0.8 * 255 = 204
        self.assertTrue(np.all(matte == 204))

    @patch(
        "vrautomatte.pipeline.rvm_onnx._download_onnx_model"
    )
    @patch(
        "vrautomatte.pipeline.rvm_onnx._select_providers"
    )
    def test_reset_clears_recurrent_state(
        self, mock_prov, mock_dl
    ):
        """reset() produces fresh independent arrays."""
        mock_dl.return_value = "/fake/model.onnx"
        mock_prov.return_value = ["CPUExecutionProvider"]
        _mock_ort.InferenceSession = MagicMock(
            return_value=_make_mock_session()
        )

        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )
        proc = RVMOnnxProcessor(variant="mobilenetv3")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        proc.process_frame(frame)

        proc.reset()
        for arr in proc._rec:
            self.assertTrue(np.all(arr == 0.0))

    @patch(
        "vrautomatte.pipeline.rvm_onnx._download_onnx_model"
    )
    @patch(
        "vrautomatte.pipeline.rvm_onnx._select_providers"
    )
    def test_gpu_failure_falls_back_to_cpu(
        self, mock_prov, mock_dl
    ):
        """If GPU provider fails, falls back to CPU."""
        mock_dl.return_value = "/fake/model.onnx"
        mock_prov.return_value = [
            "DmlExecutionProvider",
            "CPUExecutionProvider",
        ]

        cpu_session = _make_mock_session()
        call_count = {"n": 0}
        original_init = _mock_ort.InferenceSession

        def session_factory(*args, **kwargs):
            call_count["n"] += 1
            providers = kwargs.get("providers", [])
            if "DmlExecutionProvider" in providers:
                raise RuntimeError("DML init failed")
            return cpu_session

        _mock_ort.InferenceSession = MagicMock(
            side_effect=session_factory
        )

        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )
        proc = RVMOnnxProcessor(variant="mobilenetv3")

        # Should have called twice: first GPU, then CPU fallback.
        self.assertEqual(
            _mock_ort.InferenceSession.call_count, 2
        )
        second_call = (
            _mock_ort.InferenceSession.call_args_list[1]
        )
        self.assertEqual(
            second_call[1]["providers"],
            ["CPUExecutionProvider"],
        )

    @patch(
        "vrautomatte.pipeline.rvm_onnx._download_onnx_model"
    )
    @patch(
        "vrautomatte.pipeline.rvm_onnx._select_providers"
    )
    def test_cleanup_releases_session(
        self, mock_prov, mock_dl
    ):
        """cleanup() deletes session and resets state."""
        mock_dl.return_value = "/fake/model.onnx"
        mock_prov.return_value = ["CPUExecutionProvider"]
        _mock_ort.InferenceSession = MagicMock(
            return_value=_make_mock_session()
        )

        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )
        proc = RVMOnnxProcessor(variant="mobilenetv3")
        proc.cleanup()

        self.assertFalse(hasattr(proc, "_session"))
        for arr in proc._rec:
            self.assertTrue(np.all(arr == 0.0))

    @patch(
        "vrautomatte.pipeline.rvm_onnx._download_onnx_model"
    )
    @patch(
        "vrautomatte.pipeline.rvm_onnx._select_providers"
    )
    def test_cpu_only_no_gpu_works(
        self, mock_prov, mock_dl
    ):
        """Processor works on CPU-only system (no GPU at all)."""
        mock_dl.return_value = "/fake/model.onnx"
        mock_prov.return_value = ["CPUExecutionProvider"]
        _mock_ort.InferenceSession = MagicMock(
            return_value=_make_mock_session()
        )

        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )
        proc = RVMOnnxProcessor(
            variant="mobilenetv3", downsample_ratio=0.5
        )

        # Verify it selected CPU only.
        call_kwargs = (
            _mock_ort.InferenceSession.call_args[1]
        )
        self.assertEqual(
            call_kwargs["providers"],
            ["CPUExecutionProvider"],
        )

        # Verify inference works.
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        matte = proc.process_frame(frame)
        self.assertEqual(matte.shape, (240, 320))
        self.assertEqual(matte.dtype, np.uint8)


class TestRVMOnnxProtocol(unittest.TestCase):
    """Verify MatteProcessor protocol compliance."""

    def test_onnx_is_matte_processor(self):
        """RVMOnnxProcessor matches MatteProcessor protocol."""
        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )
        from vrautomatte.pipeline.matte import MatteProcessor
        self.assertTrue(
            issubclass(RVMOnnxProcessor, MatteProcessor)
        )

    def test_has_required_methods(self):
        """RVMOnnxProcessor has process_frame, reset, cleanup."""
        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )
        for method in ("process_frame", "reset", "cleanup"):
            self.assertTrue(
                hasattr(RVMOnnxProcessor, method),
                f"RVMOnnxProcessor missing {method}",
            )


class TestOnnxVariantsInFactory(unittest.TestCase):
    """Verify ONNX variants are wired into the factory."""

    def test_onnx_variants_in_list(self):
        """VARIANTS includes ONNX variants."""
        from vrautomatte.pipeline.matte import VARIANTS
        self.assertIn("mobilenetv3_onnx", VARIANTS)
        self.assertIn("resnet50_onnx", VARIANTS)

    @patch(
        "vrautomatte.pipeline.rvm_onnx._download_onnx_model"
    )
    @patch(
        "vrautomatte.pipeline.rvm_onnx._select_providers"
    )
    def test_factory_creates_onnx_processor(
        self, mock_prov, mock_dl
    ):
        """create_processor returns RVMOnnxProcessor."""
        mock_dl.return_value = "/fake/model.onnx"
        mock_prov.return_value = ["CPUExecutionProvider"]
        _mock_ort.InferenceSession = MagicMock(
            return_value=_make_mock_session()
        )

        from vrautomatte.pipeline.matte import create_processor
        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )

        proc = create_processor("mobilenetv3_onnx")
        self.assertIsInstance(proc, RVMOnnxProcessor)

    @patch(
        "vrautomatte.pipeline.rvm_onnx._download_onnx_model"
    )
    @patch(
        "vrautomatte.pipeline.rvm_onnx._select_providers"
    )
    def test_factory_creates_resnet50_onnx(
        self, mock_prov, mock_dl
    ):
        """create_processor returns RVMOnnxProcessor for resnet50_onnx."""
        mock_dl.return_value = "/fake/model.onnx"
        mock_prov.return_value = ["CPUExecutionProvider"]
        _mock_ort.InferenceSession = MagicMock(
            return_value=_make_mock_session()
        )

        from vrautomatte.pipeline.matte import create_processor
        from vrautomatte.pipeline.rvm_onnx import (
            RVMOnnxProcessor,
        )

        proc = create_processor("resnet50_onnx")
        self.assertIsInstance(proc, RVMOnnxProcessor)


if __name__ == "__main__":
    unittest.main()
