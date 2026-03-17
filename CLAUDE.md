# CLAUDE.md

## Project: VRAutoMatte

Automated video matting and alpha channel generation for VR passthrough content.

## Stack

- **Language**: Python 3.10+
- **GUI**: PySide6 (Qt6) — cross-platform desktop app
- **AI Model**: Robust Video Matting (RVM) via PyTorch
- **Video Processing**: FFmpeg (subprocess)
- **Package Manager**: uv

## Commands

```bash
uv sync                     # Install dependencies
uv run vrautomatte           # Launch GUI
uv run python -m unittest discover -s . -p "test_*.py"   # Run tests
```

## Architecture

- `src/vrautomatte/main.py` — Entry point, launches PySide6 GUI
- `src/vrautomatte/ui/` — GUI layer (main window, preview widget, worker thread)
- `src/vrautomatte/pipeline/` — Processing pipeline (matte generation, orchestration)
- `src/vrautomatte/utils/` — FFmpeg wrappers, GPU detection

## Pipeline Flow

Input Video → Extract Frames → RVM Matte Generation → Matte Video
→ (optional) Equirect→Fisheye → Red Channel → DeoVR Alpha Pack

## Conventions

- **Logging**: `from loguru import logger` — never use `print()`.
- **Strings**: double quotes preferred.
- **Line length**: max 100 characters.
- **Imports**: stdlib → third-party → local, sorted alphabetically.
- **Module size**: target ≤150 LOC, hard cap 200 LOC.
- **Docstrings**: mandatory for modules, classes, and public functions.
- **Type hints**: add for new/modified functions.
- **Scope**: solve one problem per change. No drive-by refactors.
- **Testing**: name files `test_*.py`; use `unittest.mock` for GPU/FFmpeg mocking.
- **Commits**: conventional commit messages (`feat:`, `fix:`, `docs:`, `chore:`).
