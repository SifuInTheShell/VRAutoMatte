# VRAutoMatte

## What This Is

Desktop application that uses AI to automatically generate alpha mattes from VR videos for DeoVR passthrough playback on Quest headsets. Takes any VR video, separates people from backgrounds using AI matting models, and produces alpha-packed files in DeoVR's format.

Currently supports Robust Video Matting (RVM) with a PySide6 GUI, batch processing, live preview, audio preservation, and auto-download of DeoVR fisheye masks.

## Core Value

One-click AI video matting for VR passthrough — no manual masking, no green screen, no expertise required.

## Current State

Working v0.1.0 with RVM-based matting pipeline. GUI supports file selection, batch queue, model selection (mobilenetv3/resnet50), CRF quality control, downsample ratio, DeoVR alpha pack output, and settings persistence. GPU detection auto-selects CUDA, MPS, or CPU fallback. FFmpeg handles frame extraction, fisheye conversion, red channel conversion, and alpha packing.

## Architecture / Key Patterns

- **Stack:** Python 3.10+, PySide6 (Qt6), PyTorch, FFmpeg (subprocess)
- **Entry point:** `uv run vrautomatte` → `src/vrautomatte/main.py`
- **Pipeline:** `pipeline/runner.py` orchestrates frame extraction → AI matting → reassembly → optional fisheye/alpha pack
- **Matting:** `pipeline/matte.py` — `RVMProcessor` with per-frame `process_frame()` and recurrent state
- **UI:** `ui/main_window.py` (836 LOC — over cap, but established), `ui/preview.py`, `ui/worker.py`
- **Utils:** `utils/ffmpeg.py` (split_sbs/stack_sbs/extract_frame helpers exist), `utils/gpu.py`, `utils/masks.py`, `utils/settings.py`
- **Conventions:** loguru logging, double quotes, 100 char lines, ≤150 LOC modules, unittest-based tests

## Capability Contract

See `.gsd/REQUIREMENTS.md` for the explicit capability contract, requirement status, and coverage mapping.

## Milestone Sequence

- [ ] M001: Next-Gen Matting & UX — MatAnyone 2 + SAM2 integration, SBS per-eye processing, preview scrubber seek, drag & drop
