# VRAutoMatte

Automated video matting and alpha channel generation for VR passthrough content.

## What it does

Takes a VR video → uses AI to separate human person(s) from the background → generates an alpha matte → optionally packs it into DeoVR's alpha format for passthrough playback on Quest headsets.

## Quick Start

```bash
# Install
cd VRAutoMatte
uv sync

# Run
uv run vrautomatte
```

### Requirements

- **Python 3.10+**
- **FFmpeg** on your PATH (`winget install ffmpeg` / `brew install ffmpeg`)
- **GPU recommended** (NVIDIA CUDA, Apple MPS, or CPU fallback)
- For DeoVR alpha packing: [mask8k.png](https://deovr.com/blog/123-converting-equirectangular-vr-footage-into-fisheye) from DeoVR

## Features

- **AI Video Matting** using [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) — fully automatic, no manual input needed
- **Live Preview** — see source frame and generated matte side-by-side during processing
- **DeoVR Alpha Pack** — full pipeline from equirectangular to fisheye + alpha channel packing
- **Cross-platform** — Windows, macOS, Linux (PySide6 GUI)
- **GPU accelerated** — CUDA, MPS, or CPU fallback

## Pipeline

```
Input Video ──→ Extract Frames ──→ AI Matte Generation ──→ Matte Video
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
│   ├── preview.py          # Side-by-side frame/matte preview
│   └── worker.py           # Background thread for pipeline
├── pipeline/
│   ├── matte.py            # RVM-based matte generation
│   └── runner.py           # Pipeline orchestrator
└── utils/
    ├── ffmpeg.py            # FFmpeg wrapper (split, fisheye, pack)
    └── gpu.py               # Device detection
```

## Status

🚧 **MVP** — core pipeline and GUI functional. See `docs/PLANNING.md` for roadmap.

## License

TBD
