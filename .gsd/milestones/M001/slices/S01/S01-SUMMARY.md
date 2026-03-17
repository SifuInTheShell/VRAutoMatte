---
status: complete
started: 2026-03-17
completed: 2026-03-17
---

# S01: MatAnyone 2 + SAM2 Integration — Summary

## What Was Done

Added MatAnyone 2 (CVPR 2026) as a third matting model option alongside RVM mobilenetv3/resnet50, with automatic SAM2 first-frame mask generation.

### Key Changes

1. **Processor protocol** (`pipeline/matte.py`): Defined `MatteProcessor` protocol with `process_frame()`, `reset()`, `cleanup()`. All processors implement this interface. Factory function `create_processor()` handles variant routing.

2. **RVM extracted** (`pipeline/rvm.py`): Moved RVM-specific code (model download, loading, `RVMProcessor`) to its own module. No behavior change.

3. **MatAnyone 2 processor** (`pipeline/matanyone2.py`): New module with `MatAnyone2Processor` (wraps `InferenceCore.step()`) and `generate_first_frame_mask()` (uses SAM2 `SAM2AutomaticMaskGenerator`). Sequential GPU loading: SAM2 generates mask, unloads, then MatAnyone 2 loads (D003).

4. **Runner updated** (`pipeline/runner.py`): Uses `create_processor()` factory. For MatAnyone 2, extracts first frame and generates SAM2 mask before matting loop. Emits "Generating first-frame mask" progress.

5. **UI expanded** (`ui/main_window.py`): Model combo box now has three options. CPU warning displayed when no GPU detected. Model map includes `{2: "matanyone2"}`.

6. **Dependencies** (`pyproject.toml`): `matanyone2` and `sam2` as optional extras. `netifaces` and `cchardet` excluded via uv overrides (C build requirement, not needed at runtime).

### Architecture Decisions Honored

- D003: Sequential SAM2 → MatAnyone 2 GPU loading with explicit cleanup
- D004: Fully automatic mask via SAM2, zero user input
- D009: CPU fallback with `hiera-tiny` SAM2 variant + UI warning

## What Was Learned

- MatAnyone 2's `InferenceCore.step()` accepts `(image, mask, first_frame_pred=True)` for the first frame, then just `(image)` for subsequent — matches the recurrent RVM pattern well.
- SAM2 model loading via `SAM2ImagePredictor.from_pretrained()` is cleaner than `build_sam2()` which needs config paths.
- The `netifaces` transitive dependency from matanyone2 → iopath requires MSVC on Windows. Excluded via impossible Python version marker without runtime impact.
- uv worktree nested inside main project requires `--project .` flag to avoid parent discovery.

## Verification

- 17 unit/integration tests passing
- All processor protocol conformance verified
- Factory handles all three variants correctly
- CPU fallback path tested
- Settings persistence for new model index verified

## Files Changed

- `pyproject.toml` — fixed dep format, added optional extras, uv overrides
- `src/vrautomatte/pipeline/matte.py` — protocol + factory (99 LOC)
- `src/vrautomatte/pipeline/rvm.py` — extracted RVM processor (153 LOC)
- `src/vrautomatte/pipeline/matanyone2.py` — new MatAnyone 2 + SAM2 (183 LOC)
- `src/vrautomatte/pipeline/runner.py` — factory wiring, first-frame flow
- `src/vrautomatte/ui/main_window.py` — model combo, CPU warning
- `src/vrautomatte/utils/settings.py` — comment update
- `tests/test_matte_protocol.py` — 8 tests
- `tests/test_integration_matanyone2.py` — 9 tests
