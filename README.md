# VRAutoMatte

Automated AI video matting for VR passthrough content. Separate people from backgrounds in VR videos and generate alpha channel mattes for DeoVR passthrough playback on Meta Quest headsets.

## Features

- **Three AI matting models** — RVM MobileNetV3 (fast), RVM ResNet50 (balanced), MatAnyone 2 (best edges, CVPR 2026)
- **All-people detection** — RVM models detect every person in the frame (crowds, groups, any count). MatAnyone 2 tracks close-up subjects with superior edge quality.
- **Zero manual input** — SAM2 auto-generates first-frame masks for MatAnyone 2; RVM needs no mask at all
- **Chunked extraction** — processes video in chunks instead of extracting all frames upfront, reducing peak disk usage from full-video to chunk-size
- **Resumable pipeline** — checkpoint after each chunk; interrupted jobs resume from the last completed segment
- **GPU auto-config** — automatically adapts matting resolution and memory settings to your GPU's VRAM (24 GB → full res, 16 GB → 1080p, 12 GB → 810p, etc.)
- **Frame downscaling** — LANCZOS downscale before matting, upscale matte after; prevents OOM on 8K content with consumer GPUs
- **POV body removal** — automatically detect and exclude the camera operator's body from the matte
- **Scene change detection** — refreshes masks automatically when cuts or position changes occur
- **SBS stereo support** — auto-detects side-by-side VR videos and processes each eye independently
- **DeoVR alpha packing** — full pipeline: equirectangular → fisheye → red channel → vertical alpha pack
- **Live preview** — source frame and matte side-by-side with FPS counter, ETA, and frame scrubber
- **Batch processing** — queue multiple files, process sequentially
- **Drag & drop** — drop one video to set input, drop multiple to batch queue
- **Light / dark theme** — toggle with the 🌙/☀️ button; preference is saved
- **Settings persistence** — remembers model, quality, format, chunk size, and theme
- **Cross-platform** — Windows, macOS, Linux (PySide6 GUI)
- **GPU accelerated** — NVIDIA CUDA (FP16), AMD ROCm, Apple MPS (FP16), Intel XPU, or CPU fallback

## Quick Start

### Prerequisites

- **Python 3.10+**
- **FFmpeg** on your PATH
- **GPU recommended** (any CUDA/ROCm/MPS GPU; CPU works but is much slower)

```bash
# Windows
winget install ffmpeg

# macOS
brew install ffmpeg

# Linux (Debian/Ubuntu)
sudo apt install ffmpeg
```

### Install & Run

```bash
git clone https://github.com/SifuInTheShell/VRAutoMatte.git
cd VRAutoMatte

# Install with uv (recommended)
uv sync
uv run vrautomatte

# Or with pip
pip install -e .
vrautomatte
```

### Optional: MatAnyone 2 (higher quality matting)

MatAnyone 2 requires additional dependencies. Install them with:

```bash
uv sync --extra matanyone2
```

This pulls in [MatAnyone 2](https://github.com/pq-yang/MatAnyone2) and [SAM2](https://github.com/facebookresearch/sam2). Models are downloaded automatically on first use.

## Usage Guide

### Basic Workflow

1. **Launch** the app: `uv run vrautomatte`
2. **Load a video** — click Browse or drag a file onto the window
3. **Choose a model**:
   - `mobilenetv3` — fastest, detects all people (crowds, groups)
   - `resnet50` — better quality, detects all people
   - `MatAnyone 2` — sharpest edges, tracks close-up subjects (requires `--extra matanyone2`)
4. **Choose output format**:
   - `Matte Only` — just the alpha matte video
   - `DeoVR Alpha Pack` — full passthrough pipeline for Quest headsets
5. **Click Start** — watch the live preview as it processes

### Output Formats

| Format | What You Get | Use Case |
|--------|-------------|----------|
| **Matte Only** | Grayscale matte video (white = person, black = background) | Compositing in editors, custom pipelines |
| **DeoVR Alpha Pack** | `*_ALPHA.mp4` with video on top, red-channel matte on bottom | Direct playback in DeoVR with passthrough |

### Model Comparison

| Model | People | Edges | Speed | GPU Memory | Best For |
|-------|--------|-------|-------|------------|----------|
| RVM MobileNetV3 | All | Good | ~50 fps | ~1 GB | Previewing, crowds, lower-end GPUs |
| RVM ResNet50 | All | Better | ~30 fps | ~2 GB | Crowds, groups, general use |
| MatAnyone 2 | Close-up | Best | ~8 fps | ~6 GB | 1-3 subjects, hair/transparency |

*FPS measured at 1080p on RTX 4070. Actual performance varies by resolution and hardware.*

**Which model should I use?**
- Scenes with many people (street, event, crowd) → **resnet50** (detects every person automatically)
- Close-up with 1-3 subjects where edge quality matters → **MatAnyone 2** (SAM2 auto-selects all visible people)
- Quick preview or limited GPU → **mobilenetv3**

### POV Mode

Enable **POV mode** for first-person VR content where the camera operator's body is visible. The app uses SAM2 to detect the operator's body on the first frame and excludes it from the matte — only other people are kept.

- With **MatAnyone 2**: uses instance-level matting (best quality)
- With **RVM**: uses static mask subtraction (faster, rougher)

Both strategies automatically refresh when a scene change is detected.

### SBS Stereo Videos

Side-by-side stereo VR videos (aspect ratio ≥ 1.9:1) are **auto-detected**. When the SBS checkbox is active:

- Each eye is matted independently with its own processor instance
- Results are merged back to SBS format
- Progress bar shows both eye passes

You can also manually toggle SBS for files that don't match the auto-detection heuristic.

### Batch Processing

Process multiple videos unattended:

1. Set up your first file (input, output, settings)
2. Click **+ Queue** to add it to the batch
3. Repeat for more files, or drag multiple files onto the window
4. Click **Start** — files process sequentially

### Drag & Drop

- **Single file** → sets it as the current input
- **Multiple files** → adds all to the batch queue

Supported formats: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.wmv`

### DeoVR Alpha Pipeline

When using the **DeoVR Alpha Pack** output format, the full pipeline is:

```
Input Video
  → Extract Frames
  → AI Matte Generation
  → Assemble Matte Video
  → Equirectangular → Fisheye Projection
  → Convert Matte to Red Channel
  → Pack Video + Alpha Vertically
  → Output: *_ALPHA.mp4
```

**DeoVR settings:**
- **Projection**: Equirect → Fisheye (default) or Fisheye → Fisheye
- **FOV**: 180° – 220° (default 190°)
- **Codec**: HEVC (default, smaller) or H.264 (wider compatibility)
- **CRF**: 1 – 51 (default 18; lower = better quality, larger file)

The DeoVR fisheye mask (`mask8k.png`) is downloaded automatically on first use.

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| Drag file onto window | Set input / add to batch |
| 🌙 / ☀️ button | Toggle light / dark theme |

### Settings

All settings are saved automatically to `~/.config/vrautomatte/settings.json` (Linux/macOS) or `%APPDATA%/vrautomatte/settings.json` (Windows) and restored on next launch.

## Architecture

```
src/vrautomatte/
├── main.py                    # Entry point
├── pipeline/
│   ├── matte.py               # MatteProcessor protocol + factory
│   ├── rvm.py                 # RVM processor (MobileNetV3 / ResNet50)
│   ├── matanyone2.py          # MatAnyone 2 processor
│   ├── sam2_masks.py          # SAM2 mask generation + POV heuristics
│   ├── scene_detect.py        # Scene change detector (histogram correlation)
│   ├── scaler.py              # Frame downscaler for VRAM-constrained GPUs
│   ├── checkpoint.py          # Resumable pipeline checkpoints
│   └── runner.py              # Pipeline orchestrator (chunked extract → matte → pack)
├── ui/
│   ├── main_window.py         # Main GUI window
│   ├── preview.py             # Dual-pane preview + scrubber
│   ├── themes.py              # Light / dark theme stylesheets
│   └── worker.py              # Background processing thread
└── utils/
    ├── ffmpeg.py              # FFmpeg wrappers (extract, fisheye, pack)
    ├── gpu.py                 # Device detection + GPU auto-configuration
    ├── masks.py               # DeoVR mask auto-download
    ├── sbs.py                 # SBS stereo split/merge/detection
    └── settings.py            # Settings persistence
```

### Processing Pipeline

```
Chunked Pipeline (runner.py)
  for each chunk of N frames:
    1. ffmpeg -ss <timestamp> → extract chunk (keyframe seek)
    2. FrameScaler.downscale() if frame exceeds GPU budget
    3. MatteProcessor.process_frame() per frame
    4. FrameScaler.upscale_matte() back to original resolution
    5. Flush segment video, save checkpoint
  concat all segments → final matte

MatteProcessor Protocol
├── RVMProcessor          — recurrent forward pass, detects ALL people
├── MatAnyone2Processor   — SAM2 masks → InferenceCore, tracks close-up subjects
└── POVExclusionProcessor — wraps any processor, subtracts POV body mask

GPU Auto-Config (gpu.py)
└── VRAM tier → max_matting_pixels, mem_frames, downsample_ratio
```

## Development

```bash
# Run tests
uv run python -m unittest discover -s tests -p "test_*.py"

# Run a specific test file
uv run python -m unittest tests/test_sbs.py

# Check syntax
uv run python -c "import ast; ast.parse(open('src/vrautomatte/ui/main_window.py').read()); print('OK')"
```

### Test Coverage

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_matte_protocol.py` | 17 | Processor protocol, factory, settings, CPU fallback |
| `test_sbs.py` | 14 | SBS detection, frame split/merge, matte split/merge |
| `test_scene_detect.py` | 8 | Scene change detector, cooldown, threshold, reset |
| `test_pov_mask.py` | 6 | POV body mask selection, scoring, dilation |
| `test_integration_matanyone2.py` | 5 | MatAnyone 2 processor, SAM2 masks, re-exports |

### Code Conventions

- **Logging**: `from loguru import logger` — no `print()` in committed code
- **Strings**: double quotes preferred
- **Line length**: max 100 characters
- **Imports**: stdlib → third-party → local, alphabetical within groups
- **Type hints**: required for new/modified functions
- **Tests**: `test_*.py` or `*_test.py`, using `unittest`

## Requirements

### System

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| Python | 3.10 | 3.12 |
| FFmpeg | 4.x | 7.x |
| GPU VRAM | — (CPU ok) | 6 GB+ |
| RAM | 4 GB | 16 GB |
| Disk | 500 MB (app) | + space for video processing |

### GPU Support

| Platform | Framework | Status |
|----------|-----------|--------|
| NVIDIA | CUDA | ✅ Full support |
| AMD | ROCm | ✅ Full support |
| Apple | MPS | ✅ Full support |
| Intel | XPU (oneAPI) | ✅ Full support |
| CPU | — | ✅ Fallback (slower) |

## Model Downloads

Models are downloaded automatically on first use and cached locally:

| Model | Size | Cache Location |
|-------|------|----------------|
| RVM MobileNetV3 | ~15 MB | `~/.cache/vrautomatte/models/` |
| RVM ResNet50 | ~55 MB | `~/.cache/vrautomatte/models/` |
| MatAnyone 2 | ~2 GB | Managed by HuggingFace Hub |
| SAM2 | ~400 MB | Managed by HuggingFace Hub |
| DeoVR mask8k.png | ~2 MB | `~/.cache/vrautomatte/masks/` |

## License

[MIT](LICENSE)

## Acknowledgements

- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) — recurrent matting architecture
- [MatAnyone 2](https://github.com/pq-yang/MatAnyone2) — CVPR 2026 SOTA video matting
- [SAM2](https://github.com/facebookresearch/sam2) — Segment Anything Model 2 for automatic mask generation
- [DeoVR](https://deovr.com/) — VR video player with alpha passthrough support
