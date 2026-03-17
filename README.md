# VRAutoMatte

Automated video matting and alpha channel generation for VR passthrough content.

## What it does

Takes a VR video → uses AI to separate human person(s) from the background → generates an alpha matte → optionally packs it into DeoVR's alpha format for passthrough playback on Quest headsets.

## Quick Start

```bash
cd VRAutoMatte
uv sync
uv run vrautomatte
```

### Requirements

- **Python 3.10+**
- **FFmpeg** on your PATH (`winget install ffmpeg` / `brew install ffmpeg`)
- **GPU recommended** (NVIDIA CUDA, Apple MPS, or CPU fallback)
- DeoVR fisheye mask is **auto-downloaded** when needed

## Features

- **AI Video Matting** — [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting), fully automatic, no manual input needed
- **Live Preview** — source frame and generated matte side-by-side with ETA and FPS counter
- **DeoVR Alpha Pack** — full pipeline: equirectangular → fisheye → alpha channel packing
- **Audio Preservation** — audio track carries through all processing steps
- **Batch Processing** — queue multiple files, process sequentially
- **Settings Persistence** — remembers your last-used model, quality, format, and window size
- **Auto-download Mask** — DeoVR fisheye mask (mask8k.png) fetched automatically
- **Cross-platform** — Windows, macOS, Linux (PySide6 GUI)
- **GPU accelerated** — CUDA, MPS, or CPU fallback

## GUI Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Files                                                      │
│  Input:  [________________________] [Browse] [+ Queue]      │
│  Output: [________________________] [Browse]                │
│  3840×1920 | 60 fps | 7200 frames | 120s | hevc            │
├─────────────────────────────────────────────────────────────┤
│  Settings                                                   │
│  Model: [mobilenetv3 ▼]     Output: [Matte Only ▼]         │
│  CRF: [====●========] 18    Downsample: [0.25 ▼]           │
│  ── VR Settings (DeoVR mode) ──                             │
│  Projection: [Equirect→Fisheye ▼]  FOV: 180°  Codec: HEVC  │
├─────────────────────────────────────────────────────────────┤
│  Preview                                          12.3 fps  │
│  ┌─ Source Frame ──┐  ┌─ Generated Matte ─┐                │
│  │                 │  │                    │                │
│  └─────────────────┘  └────────────────────┘                │
│  Frame 142 / 7,200  [═══●════════════]   ETA: 8m 34s       │
├─────────────────────────────────────────────────────────────┤
│  Batch Queue (3 files)                                      │
│  video1.mp4 → video1_matte.mp4                              │
│  video2.mp4 → video2_matte.mp4                              │
├─────────────────────────────────────────────────────────────┤
│  [▶ Start Processing] [Cancel]    [████████░░] 65%          │
│  Generating mattes — frame 4,680/7,200    Device: RTX 4070  │
└─────────────────────────────────────────────────────────────┘
```

## Pipeline

```
Input Video ──→ Extract Frames ──→ AI Matte Generation ──→ Matte Video + Audio
                                                              │
                                           ┌──────────────────┘
                                           │ (DeoVR mode only)
                                           ▼
                              Equirect→Fisheye ──→ Red Channel ──→ Alpha Pack ──→ _ALPHA.mp4
```

## Architecture

```
src/vrautomatte/
├── main.py                 # Entry point
├── ui/
│   ├── main_window.py      # Main GUI window (PySide6)
│   ├── preview.py          # Side-by-side preview + scrubber + ETA
│   └── worker.py           # Background thread for pipeline
├── pipeline/
│   ├── matte.py            # RVM-based matte generation
│   └── runner.py           # Pipeline orchestrator + audio preservation
└── utils/
    ├── ffmpeg.py            # FFmpeg wrappers (split, fisheye, pack, audio)
    ├── gpu.py               # Device detection
    ├── masks.py             # DeoVR mask auto-download
    └── settings.py          # Settings persistence (~/.config/vrautomatte/)
```

## Roadmap

- [ ] **MatAnyone 2** — CVPR 2026 SOTA model for higher edge quality
- [ ] **SBS split processing** — process left/right eyes independently
- [ ] **Preview scrubber seek** — preview any frame before processing
- [ ] **Drag & drop** — drop video files onto the window

## License

TBD
