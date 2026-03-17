# S01: MatAnyone 2 + SAM2 Integration

**Goal:** Add MatAnyone 2 as a selectable matting model with automatic SAM2 first-frame masking, GPU memory management, CPU fallback, and UI model selector expansion — without breaking existing RVM.

**Demo:** User selects "MatAnyone 2" from the model dropdown, hits Start on a video, and a matte video is produced automatically (SAM2 generates first-frame mask, MatAnyone 2 mattes the full video). Works on both GPU and CPU.

## Must-Haves

- MatAnyone 2 selectable alongside RVM variants (mobilenetv3, resnet50)
- SAM2 auto-generates first-frame segmentation mask without user interaction
- GPU memory managed: SAM2 unloaded before MatAnyone 2 loads
- CPU fallback with smaller SAM2 variant and visible UI warning
- Model weights auto-download on first use with progress indication
- Existing RVM paths completely unchanged
- Settings persist new model selection

## Proof Level

- This slice proves: integration
- Real runtime required: yes (model loading + inference)
- Human/UAT required: yes (visual matte quality, UI model selector)

## Verification

- `uv run python -m unittest discover -s tests -p "test_*.py"` — unit tests for processor protocol, factory function, device handling
- Manual: launch app, select MatAnyone 2, process a short video, verify matte output exists and is non-empty
- Manual: repeat on CPU-only (set `CUDA_VISIBLE_DEVICES=""`) — verify UI shows CPU warning and processing completes

## Observability / Diagnostics

- Runtime signals: loguru messages for model download, device selection, SAM2 mask generation, MatAnyone 2 inference start/complete, GPU memory cleanup
- Inspection surfaces: UI device label shows "CUDA: {name}" or "CPU (no GPU)"
- Failure visibility: model download failures logged with URL and error; SAM2 mask generation failure logged with frame size and error
- Redaction constraints: none

## Integration Closure

- Upstream surfaces consumed: `utils/gpu.py` `get_device()`, `pipeline/runner.py` `Pipeline` orchestrator, `ui/main_window.py` model combo box
- New wiring introduced: `MatAnyone2Processor` registered in processor factory; UI combo box gains third option; `runner.py` calls SAM2 mask generation before MatAnyone 2 inference
- What remains before milestone is truly usable: S02 (SBS), S03 (scrubber), S04 (drag & drop)

## Tasks

- [ ] **T01: Add matanyone2 and sam2 dependencies** `est:15m`
  - Why: MatAnyone 2 and SAM2 packages must be installable before any integration code
  - Files: `pyproject.toml`
  - Do: Add `matanyone2` (git dep from pq-yang/MatAnyone2) and `sam2` (PyPI) as optional dependencies under an extras group `[matanyone2]`. Keep them optional so the base install stays lightweight. Verify they resolve without conflicts with existing torch/torchvision pins.
  - Verify: `uv sync --extra matanyone2` succeeds; `uv run python -c "from matanyone2 import MatAnyone2; from sam2.build_sam import build_sam2; print('ok')"`
  - Done when: both packages importable in the project venv

- [ ] **T02: Create MatAnyone2Processor with processor protocol** `est:1h`
  - Why: Need a common interface so runner.py can use either RVM or MatAnyone 2 without branching. MatAnyone 2's InferenceCore.step() provides per-frame matting with recurrent state, matching RVM's pattern.
  - Files: `src/vrautomatte/pipeline/matte.py`, `tests/test_matte_protocol.py`
  - Do: (1) Define a `MatteProcessor` Protocol with `process_frame(frame: np.ndarray) -> np.ndarray`, `reset()`, and `cleanup()` methods. (2) Retrofit `RVMProcessor` to match the protocol (no behavior change). (3) Create `MatAnyone2Processor` implementing the same protocol, wrapping `matanyone2.InferenceCore`. (4) Add a `create_processor(variant, device, downsample_ratio)` factory function. (5) Add SAM2 first-frame mask generation as a separate `generate_first_frame_mask(frame, device)` function. (6) Handle sequential GPU loading: generate mask with SAM2, unload SAM2, then load MatAnyone 2. (7) On CPU, use `sam2-hiera-tiny` model; on GPU, use `sam2-hiera-small`.
  - Verify: `uv run python -m unittest tests/test_matte_protocol.py` — test protocol conformance with mocked models
  - Done when: factory returns correct processor for each variant; SAM2 mask generation function produces a binary mask from a test image

- [ ] **T03: Wire MatAnyone 2 into pipeline runner** `est:45m`
  - Why: The runner orchestrates the full pipeline; it needs to call SAM2 mask generation when MatAnyone 2 is selected, then use the MatAnyone2Processor for frame-by-frame matting with progress callbacks.
  - Files: `src/vrautomatte/pipeline/runner.py`
  - Do: (1) Update `Pipeline._run_matte_pass()` to use `create_processor()` factory instead of directly instantiating `RVMProcessor`. (2) Add a pre-matting step: if variant is "matanyone2", extract first frame, generate SAM2 mask, then initialize MatAnyone2Processor with the mask. (3) Ensure progress callbacks fire for both SAM2 mask generation (as "Generating mask...") and per-frame matting. (4) Add cleanup() call on processor when matting completes or is cancelled.
  - Verify: `uv run python -m unittest tests/test_runner_matanyone2.py` — test that runner correctly sequences SAM2 → MatAnyone 2
  - Done when: runner can execute a MatAnyone 2 pipeline from config to output file

- [ ] **T04: Expand UI model selector and add CPU warning** `est:30m`
  - Why: UI needs to offer MatAnyone 2 as a third option and warn when running on CPU
  - Files: `src/vrautomatte/ui/main_window.py`, `src/vrautomatte/utils/settings.py`
  - Do: (1) Add "MatAnyone 2 (quality)" to model combo box (index 2). (2) Update `model_map` to include `{2: "matanyone2"}`. (3) Expand settings `_DEFAULTS["model_variant"]` comment to document index 2. (4) Add device info label to the UI header showing detected device. (5) If device is CPU, show amber warning: "⚠️ CPU mode — processing will be slower". (6) Persist model selection index including the new option.
  - Verify: Launch app, verify combo box shows three options, select MatAnyone 2, restart app, verify selection persists
  - Done when: model selector has three entries, CPU warning appears when no GPU, settings round-trip works

- [ ] **T05: Integration test — end-to-end MatAnyone 2 pipeline** `est:30m`
  - Why: Prove the full chain works: UI selection → runner → SAM2 mask → MatAnyone 2 matte → output file
  - Files: `tests/test_integration_matanyone2.py`
  - Do: (1) Write integration test that creates a PipelineConfig with variant="matanyone2", provides a short test video (or synthetic frames), runs the pipeline, and asserts output file exists with non-zero size. (2) Test CPU fallback path by forcing device="cpu". (3) Test that RVM pipeline still works identically (regression). (4) Skip tests gracefully if matanyone2/sam2 packages not installed.
  - Verify: `uv run python -m unittest tests/test_integration_matanyone2.py`
  - Done when: integration test passes on both GPU and CPU paths; RVM regression test passes

## Files Likely Touched

- `pyproject.toml`
- `src/vrautomatte/pipeline/matte.py`
- `src/vrautomatte/pipeline/runner.py`
- `src/vrautomatte/ui/main_window.py`
- `src/vrautomatte/utils/settings.py`
- `tests/test_matte_protocol.py` (new)
- `tests/test_runner_matanyone2.py` (new)
- `tests/test_integration_matanyone2.py` (new)
