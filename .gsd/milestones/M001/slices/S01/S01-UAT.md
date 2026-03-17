# S01: MatAnyone 2 + SAM2 Integration — UAT

**Milestone:** M001
**Written:** 2026-03-17

## UAT Type

- UAT mode: mixed (artifact-driven + live-runtime)
- Why this mode is sufficient: protocol conformance is artifact-driven; model quality and UI need visual verification

## Preconditions

- `uv sync --project . --extra matanyone2` completed successfully
- FFmpeg installed and on PATH
- A test video file available (any MP4, 5-10 seconds is enough)

## Smoke Test

Launch the app: `.venv/Scripts/python -m vrautomatte.main`
Verify the model dropdown shows three options: mobilenetv3, resnet50, MatAnyone 2

## Test Cases

### 1. MatAnyone 2 model selection and persistence

1. Launch the app
2. Open the model dropdown
3. Verify three options: "mobilenetv3 (fast)", "resnet50 (quality)", "MatAnyone 2 (quality+)"
4. Select "MatAnyone 2 (quality+)"
5. Close the app
6. Reopen the app
7. **Expected:** MatAnyone 2 is still selected

### 2. MatAnyone 2 end-to-end processing (GPU)

1. Launch the app
2. Select "MatAnyone 2 (quality+)"
3. Browse to a test video
4. Set output path
5. Click Start
6. **Expected:** Progress shows "Generating first-frame mask" briefly, then "Generating mattes" with frame progress. A matte video file is created at the output path with non-zero size.

### 3. CPU fallback warning

1. Set `CUDA_VISIBLE_DEVICES=""` before launching
2. Launch the app
3. **Expected:** Device label shows "⚠️ CPU mode — processing will be slower" in amber text

### 4. CPU processing works

1. With CPU mode active, select MatAnyone 2
2. Process a short video (5-10 seconds)
3. **Expected:** Processing completes (slower), output matte file is created

### 5. RVM regression

1. Select "mobilenetv3 (fast)"
2. Process the same test video
3. **Expected:** Matte video produced, identical behavior to before this slice

## Edge Cases

### No matanyone2 package installed

1. In an environment without `--extra matanyone2`, select MatAnyone 2
2. Click Start
3. **Expected:** Clear error message about missing package, not a stack trace

### Very short video (1-2 frames)

1. Select MatAnyone 2
2. Process a 1-frame video
3. **Expected:** SAM2 generates mask, MatAnyone 2 processes the single frame, output created

## Failure Signals

- ImportError mentioning matanyone2 or sam2 — packages not installed
- CUDA out of memory — SAM2 not unloaded before MatAnyone 2 (D003 violation)
- No masks found by SAM2 — may happen with unusual video content
- model_combo only shows 2 options — UI update not applied

## Requirements Proved By This UAT

- R001 — MatAnyone 2 available as model option
- R002 — SAM2 auto-generates first-frame mask
- R003 — Model weights auto-download on first use
- R008 — GPU memory managed (SAM2 unloaded before MatAnyone 2)
- R009 — RVM pipeline unchanged
- R010 — Settings persist new model selection
- R011 — CPU fallback with UI warning

## Not Proven By This UAT

- R004-R005 — SBS processing (S02)
- R006 — Preview scrubber seek (S03)
- R007 — Drag & drop (S04)
- Matte quality comparison (subjective, not automatable)

## Notes for Tester

- First run will download MatAnyone 2 (~400MB) and SAM2 model weights — expect a wait
- CPU processing of MatAnyone 2 on a long video can take 10-30x longer than GPU
- The SAM2 mask picks the "largest" detected region — this works well for single-person VR but may pick a wrong object in multi-person or cluttered scenes
