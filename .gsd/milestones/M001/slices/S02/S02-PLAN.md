# S02: SBS Stereo Per-Eye Processing

**Goal:** SBS stereo VR videos are auto-detected and processed per-eye with correct recombination.
**Demo:** User loads an SBS stereo video → SBS is auto-detected → checkbox auto-set → matting runs per-eye → output is correct stereo.

## Must-Haves

- Auto-detection via aspect ratio heuristic (width/height ≥ 1.9)
- Frame split/merge utilities (split_frame, merge_frames, split_matte, merge_mattes)
- Runner SBS matte pass: left eye all frames → right eye all frames → merge
- UI checkbox with auto-detection label
- Works with both RVM and MatAnyone 2

## Verification

- `uv run python -m unittest tests/test_sbs.py` — 14 tests covering detection, split, merge
- SBS checkbox auto-sets when loading a 2:1+ aspect ratio video

## Tasks

- [x] **T01: SBS utilities and runner integration** `est:1h`
  - sbs.py with detect/split/merge, runner._run_sbs_matte_pass, UI checkbox + auto-detect
  - Done: all implemented and tested

## Files Touched

- `src/vrautomatte/utils/sbs.py`
- `src/vrautomatte/pipeline/runner.py`
- `src/vrautomatte/ui/main_window.py`
- `tests/test_sbs.py`
