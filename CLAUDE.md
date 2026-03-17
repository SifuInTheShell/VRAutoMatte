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

### CUDA OOM on 5800x2900 video (RTX 5080 16GB)
```
CUDA out of memory. Tried to allocate 4.09 GiB. GPU 0 has a total capacity of 15.92 GiB
of which 0 bytes is free. Of the allocated memory 18.48 GiB is allocated by PyTorch, and
4.62 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is
large try setting PYTORCH_ALLOC_CONF=expandable_segments:True to avoid fragmentation.
See documentation for Memory Management
(https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```
- Occurs during matting of 5800x2900 content on RTX 5080 (16GB VRAM)
- PyTorch allocated 18.48 GiB (exceeds 15.92 GiB capacity) + 4.62 GiB reserved but unused
- TODO: investigate — likely need to lower downsample_ratio, process in tiles, or free SAM2 before matting
- Try `PYTORCH_ALLOC_CONF=expandable_segments:True` as suggested by PyTorch
- May need to reduce batch size or frame resolution before feeding to the model

## Features Added (2026-03-17)
- **Custom temp directory:** UI setting with browse/reset buttons, persisted in settings.json, passed to `tempfile.TemporaryDirectory(dir=...)`
- **Frame extraction progress:** Polls output directory every 0.5s, shows frame count, fps, and ETA in progress bar
