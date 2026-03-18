# Claude Code Project Notes

## Project Overview
VRAutoMatte — Qt-based desktop app for automated video matting and alpha channel generation for VR passthrough content.

## Repository
- **Path:** `D:\CODE\VRAutoMatte\VRAutoMatte` (nested directory under `D:\CODE\VRAutoMatte`)
- **GitHub:** https://github.com/SifuInTheShell/VRAutoMatte
- **Branch:** master

## Architecture
- **Entry point:** `src/vrautomatte/main.py` → MainWindow (PySide6/Qt)
- **Pipeline:** `pipeline/runner.py` — orchestrates ffmpeg extraction → AI matting → video assembly
- **GPU bootstrap:** `utils/bootstrap.py` — detects NVIDIA GPU via nvidia-smi, installs correct CUDA PyTorch wheel at startup before any torch imports
- **Settings:** `utils/settings.py` → `~/.config/vrautomatte/settings.json`
- **Worker threading:** `ui/worker.py` — PipelineWorker (QThread) runs pipeline, InstallWorker handles in-app dependency install
- **GPU detection:** `utils/gpu.py` — get_device() and get_device_info() for CUDA/MPS/CPU

## User Hardware
- Windows 11 Pro laptop with NVIDIA RTX 5080 (Blackwell architecture, sm_120, CUDA 13.2)
- Requires PyTorch cu128 wheels (cu126 doesn't support Blackwell)
- Synology NAS at `\\192.168.1.95\usbshare1` for storing large VR video files
- Typical content: 8K (5800x2900) HEVC, 60fps, 100k+ frames

## Bugs Fixed (2026-03-17)

### 1. GPU not detected — `gpu.py` typo
- `torch.cuda.get_device_properties(0).total_mem` → `total_memory`
- Caused UI to show "Device: unknown" instead of GPU info

### 2. CPU-only PyTorch installed
- `pyproject.toml` had no CUDA index, so PyPI served CPU-only torch
- Fixed with `[[tool.uv.index]]` pointing to `https://download.pytorch.org/whl/cu128`
- Platform marker `sys_platform != 'darwin'` so macOS gets regular PyPI torch (MPS)
- Bootstrap (`utils/bootstrap.py`) acts as safety net: detects GPU via nvidia-smi, installs correct CUDA wheel if mismatch, restarts process

### 3. `uv sync` overwriting CUDA torch on every `uv run`
- Added explicit CUDA index to `pyproject.toml` with `[tool.uv.sources]` so `uv sync` resolves cu128 directly
- No more bootstrap reinstall on every launch

### 4. In-app MatAnyone2 install failing with DLL lock errors
- `uv sync --extra matanyone2` tried to replace loaded DLLs (numpy, torch)
- Changed `InstallWorker` to use `uv pip install` which only adds new packages

### 5. Frame extraction "hanging" at frame 977
- Root cause: ffmpeg's `select='between(n,start,end)'` filter decodes ALL frames in the video, not just the selected range. For a 161k-frame video with 1000 selected, ffmpeg silently decodes remaining 160k frames (~89 min) after outputting PNGs
- Fixed by adding `-frames:v <count>` to stop ffmpeg after outputting selected frames
- Progress tracking uses `subprocess.Popen` with DEVNULL pipes + `os.listdir()` polling

## Key Lessons / Gotchas

### Windows subprocess pipes
- **Avoid `subprocess.PIPE`** for long-running ffmpeg processes — use `DEVNULL` + directory polling
- `BufferedReader.read(n)` blocks until exactly n bytes arrive (not partial reads like Unix)
- ffmpeg stderr uses `\r` not `\n` for progress — line-based drain threads never yield
- Cross-pipe deadlocks happen around ~64KB of stderr (~977 frames × 67 bytes)

### ffmpeg select filter
- Always pair `select='between(...)'` with `-frames:v <count>` to prevent full-video decode
- Without `-frames:v`, ffmpeg processes every frame even after the selected range is done

### PyTorch CUDA on Windows
- RTX 50xx (Blackwell) needs cu128 minimum — cu126 detects GPU but can't run on sm_120
- `uv sync` overwrites manually installed torch — must configure index in `pyproject.toml`
- Bootstrap module runs before torch imports, uses env var `VRAUTOMATTE_TORCH_OK` to skip check on restart

## Open Issues

(None — CUDA OOM resolved by GPU auto-config + frame downscaling in 2026-03-18 changes)

## Features Added (2026-03-17)
- **Custom temp directory:** UI setting with browse/reset buttons, persisted in settings.json, passed to `tempfile.TemporaryDirectory(dir=...)`
- **Frame extraction progress:** Polls output directory every 0.5s, shows frame count, fps, and ETA in progress bar

## Features Added (2026-03-18)

### Chunked Extraction Pipeline
- Replaced extract-all-then-matte with chunked loop: extract N frames -> matte -> flush segment -> repeat
- Uses ffmpeg `-ss` keyframe seek (fast, ~1-2 frame imprecision at boundaries)
- Peak source PNG disk drops from `num_frames * PNG_size` to `chunk_size * PNG_size`
- New files: `pipeline/scaler.py`, `pipeline/checkpoint.py`

### Resumable Pipeline (Checkpoint)
- JSON checkpoint saved after each segment flush
- Deterministic temp dir: `vrautomatte_{stem}_{config_hash[:8]}/`
- Validates input_hash (first 64KB) + config_hash — stale checkpoint = fresh start
- Stale temp dirs (>7 days) cleaned on pipeline start
- UI checkbox "Auto-resume on restart"

### GPU OOM Fix — Frame Downscaling
- `pipeline/scaler.py`: FrameScaler pre-matte downscale + post-matte LANCZOS upscale
- No-op if frame fits within max_pixels budget
- For VR passthrough at 90+ FOV, quality impact is imperceptible

### Adaptive GPU Configuration
- `utils/gpu.py`: `auto_configure_gpu()` returns VRAM-tier recommendations
- Tiers: >=24GB (no limit), 16GB (1080p), 12GB (810p), 8GB (720p), <=6GB (540p)
- Auto-applied at pipeline start, only overrides defaults
- For SBS: thresholds apply per-eye (already halved)

### MPS FP16 Support
- MatAnyone2 + RVM FP16 now enabled on Apple MPS (was CUDA-only)
- Added MPS autocast path in matanyone2.py

### Memory Cleanup
- `gc.collect()` after SAM2 unload in sam2_masks.py
- `torch.cuda.empty_cache()` between chunks in chunked pipeline

### UI Additions
- Chunk size combo (100/250/500/1000) with tooltip
- Auto-resume checkbox
- Disk usage estimate label below progress bar
- GPU auto-config feedback in status bar
