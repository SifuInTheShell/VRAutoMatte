---
slice: S02
title: SBS Stereo Per-Eye Processing
status: complete
started: 2026-03-15
completed: 2026-03-17
tasks_completed: 1
tasks_total: 1
---

# S02 Summary: SBS Stereo Per-Eye Processing

## What Was Built

SBS (side-by-side) stereo VR video support with automatic detection and per-eye matting.

### Key Components

- **`utils/sbs.py`** (95 LOC) — `detect_sbs()` aspect ratio heuristic (≥1.9), `split_frame()` / `merge_frames()` for RGB, `split_matte()` / `merge_mattes()` for grayscale
- **`runner._run_sbs_matte_pass()`** — Splits all frames, runs left eye pass (full sequence), right eye pass (full sequence), merges. Separate processor instances per eye for clean state.
- **UI** — SBS checkbox with auto-detection on file load. Green "SBS detected" label when auto-detected.

### Design Decisions

- **Sequential eye processing** (all left frames → all right frames) rather than interleaved, to keep processor temporal state consistent within each eye.
- **Separate processor instances** per eye — clean SAM2/MatAnyone 2 initialization for each eye's perspective.
- **2x total frames in progress** — progress bar counts both eye passes.

## Test Coverage

14 tests in `test_sbs.py`:
- Detection: standard ratios, borderline 1.9, sub-1.9, zero height
- Split/merge: shape correctness, content preservation, round-trip, copy semantics

## What's Next

Nothing — feature complete for M001 scope.
