# VRAutoMatte

Automated AI video matting for VR passthrough content. Separate people from backgrounds in VR videos and generate alpha-channel mattes for DeoVR passthrough playback on Meta Quest headsets.

## Features

- **Multiple AI matting models** — RVM MobileNetV3 (fast) and RVM ResNet50 (quality) detect *every* person in frame; MatAnyone 2 and SAM2Matting add sharp single-subject instance matting. Torch/CUDA and ONNX/DirectML backends.
- **All-people detection** — RVM models matte every person in the frame (crowds, groups, any count); no manual selection.
- **Multi-subject tracking** — SAM2Matting can track and matte 1–4 chosen people independently, merging their alphas.
- **Zero manual input** — no masks, trimaps, or prompts required; subjects are seeded automatically.
- **Raw-video streaming pipeline** — ffmpeg decodes and scales straight into the matting loop over raw pipes (no PNG round-trips), and mattes stream back out into the segment encoder.
- **ROI-restricted matting** — mattes only a sticky, padded window around the tracked subject for a large speed-up when the subject is small in frame (default on, full-frame fallback when the subject is lost or fills the frame).
- **Half-rate matting** — optionally matte every 2nd frame and interpolate alpha for the rest (~2× faster on 60 fps content, exact frame count preserved).
- **Matting-resolution selector** — Auto (GPU-tier), 1080p, 720p, or 540p per eye.
- **Background chaptered assembly** — DeoVR fisheye conversion and alpha packing run *during* matting (different hardware engines) so assembly time hides almost entirely behind inference.
- **Temporal alpha smoothing** — EMA-based frame blending (off / light / medium / heavy) to reduce matte jitter.
- **Resumable pipeline** — checkpoint after each segment; interrupted jobs resume from the last completed segment.
- **GPU auto-config** — adapts matting resolution and memory to your GPU's VRAM (24 GB → full res, 16 GB → 1080p, 12 GB → 810p, …).
- **POV body removal** — detects and excludes the camera operator's own body from the matte.
- **Scene change detection** — refreshes masks automatically when cuts or large position changes occur.
- **SBS stereo support** — auto-detects side-by-side VR videos and mattes each eye (RVM batches both eyes in one forward pass; optional parallel-eye threads for MatAnyone 2).
- **DeoVR alpha packing** — follows the official DeoVR spec: 40 % scale matte, red-channel-only, 6-segment corner packing, AV1 encode.
- **Auto projection detection** — reads the lens profile from the filename (`_FISHEYE190`, `_MKX200`, `_MKX220`, `_VRCA220`, `_RF52`) and sets projection + FOV automatically.
- **Final-encode preset** — Fast (p2) / Balanced (p4) / Quality (p6) NVENC preset knob.
- **Live preview** — source frame and matte side-by-side with FPS counter, ETA, and frame scrubber (toggle on/off).
- **Batch processing** — queue multiple files with per-file projection detection; mixed equirectangular and fisheye batches work automatically.
- **Drag & drop** — drop one video to set input, drop multiple to batch-queue.
- **NVENC encoding** — hardware-accelerated encoding via NVENC (AV1, HEVC, H.264) with automatic CPU fallback.
- **Light / dark theme** — toggle with the 🌙/☀️ button; preference is saved.
- **Settings persistence** — remembers all settings across sessions.
- **Cross-platform** — Windows, macOS, Linux (PySide6 GUI).
- **GPU accelerated** — NVIDIA CUDA (FP16) and Apple MPS (FP16) via PyTorch; AMD, Intel, or any DirectML/CoreML GPU via the ONNX RVM variants; CPU fallback everywhere.

## Quick Start

### Prerequisites

- **Python 3.10+**
- **FFmpeg** on your PATH
- **GPU recommended** (any CUDA/MPS/DirectML GPU; CPU works but is much slower)

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

The default install ships the RVM models (torch + ONNX), which cover the all-people VR use case. MatAnyone 2 and SAM2Matting are optional extras (see below).

### Optional models

**MatAnyone 2** (experimental, flat/non-VR video):

```bash
uv sync --extra matanyone2
```

Pulls in [MatAnyone 2](https://github.com/pq-yang/MatAnyone2) and [SAM2](https://github.com/facebookresearch/sam2). Does **not** work with fisheye or equirectangular VR content — use for standard flat video only.

**SAM2Matting** (trial, best single-subject instance matte):

```bash
uv sync --extra sam2matting
```

Installs the `sam2` stack; the SAM2Matting model repo itself is **not** pip-installable and is auto-downloaded to `~/.cache/vrautomatte/sam2matting/` on first use (override with `VRAUTOMATTE_SAM2MATTING_PATH`). The model weights are **CC BY-NC-SA 4.0 (non-commercial)** — see [License](#license).

Models are downloaded automatically on first use.

## Usage Guide

### Basic Workflow

1. **Launch** the app: `uv run vrautomatte`
2. **Load a video** — click Browse or drag a file onto the window
3. **Choose a model** (see [Model Comparison](#model-comparison))
4. **Choose output format**:
   - `Matte Only` — just the alpha-matte video
   - `DeoVR Alpha Pack` — full passthrough pipeline for Quest headsets
5. **Click Start** — watch the live preview as it processes

### Matting Models

| UI label | `model_variant` | Backend | Detects |
|----------|-----------------|---------|---------|
| mobilenetv3 — all people, fast | `mobilenetv3` | Torch/CUDA·MPS | Everyone |
| resnet50 — all people, quality | `resnet50` | Torch/CUDA·MPS | Everyone |
| mobilenetv3 ONNX — DirectML, any GPU | `mobilenetv3_onnx` | ONNX Runtime (DirectML/CoreML/CPU) | Everyone |
| resnet50 ONNX — DirectML, any GPU | `resnet50_onnx` | ONNX Runtime (DirectML/CoreML/CPU) | Everyone |
| MatAnyone 2 (experimental, non-VR) | `matanyone2` | Torch | One subject (flat video) |
| SAM2Matting (trial) — 1 person, fast + sharp | `sam2matting` | Torch | 1–4 chosen subjects |

The ONNX variants run on **AMD, Intel, and any DirectML/CoreML GPU** where the torch/CUDA path isn't available. MatAnyone 2 and SAM2Matting appear as *"click to install"* until their extra is installed.

### Auto-Detection

When you load a video, the app automatically detects:

| Property | How It's Detected | Fallback |
|----------|------------------|----------|
| **SBS stereo** | Aspect ratio ≥ 1.9:1 | Manual checkbox |
| **Projection** | Filename tags: `_FISHEYE`, `_MKX200`, `_MKX220`, `_VRCA220`, `_RF52` | Defaults to Equirectangular → Fisheye |
| **FOV** | Extracted from tag (e.g. `_FISHEYE190` → 190°, `_MKX200` → 200°) | FOV slider value |
| **GPU settings** | VRAM tier auto-config (resolution, downsample ratio, memory frames) | Manual override |

In batch mode, projection and FOV are detected **per file** from each filename — mixed batches of equirectangular and fisheye content process correctly.

### Output Formats

| Format | What You Get | Use Case |
|--------|-------------|----------|
| **Matte Only** | Grayscale matte video (white = person, black = background) | Compositing in editors, custom pipelines |
| **DeoVR Alpha Pack** | `*_alpha.mp4` with red-channel alpha packed into fisheye corners | Direct playback in DeoVR with passthrough |

### Model Comparison

Measured on an RTX 5080 Laptop (16 GB), 2880×1440 SBS HEVC @ 59.94 fps, POV mode on, half-rate matting (every 2nd frame), ROI on — 1000 mid-video frame pairs. "Effective fps" is source frames per second at the app's half-rate setting; "Full video" is a 147,317-frame file. See [`docs/BENCHMARKS_2026-07-08.md`](docs/BENCHMARKS_2026-07-08.md) for the full method.

| Model | Config | People | Effective fps | Full video | Best For |
|-------|--------|--------|---------------|-----------|----------|
| resnet50 (torch) | fp16, eye-batched, ROI | All | **113** | ~25 min | Throughput + quality on NVIDIA |
| mobilenetv3 (torch) | fp16, eye-batched, ROI | All | 110 | ~25 min | Same speed; slightly softer edges |
| mobilenetv3 ONNX | DirectML, eye-batched | All | 22 | ~1.8 h | AMD / Intel / any GPU |
| resnet50 ONNX | DirectML, eye-batched | All | 21 | ~1.9 h | AMD / Intel / any GPU |
| SAM2Matting (tiny) | bf16, stride 2, chunked | 1–4 chosen | 11 | ~3.7 h | Sharpest single-subject matte |
| MatAnyone 2 | fp16, ROI, threaded eyes | One | 9 | ~4.4 h | Flat video, quality A/B only |

On NVIDIA hardware the torch RVM models are frame-feeder-bound, not network-bound — resnet50 costs nothing over mobilenetv3, so there's no reason to pick mobilenetv3 on an NVIDIA card. Real-world end-to-end throughput adds ffmpeg decode + NVENC encode around the matting (~30–45 min per full video for the torch models).

**Which model should I use?**
- Crowds / groups / anyone-in-frame on NVIDIA → **resnet50** (torch)
- AMD / Intel / non-CUDA GPU → **resnet50 ONNX** or **mobilenetv3 ONNX**
- One or a few specific people, sharpest edges → **SAM2Matting** (trial, non-commercial)
- Flat (non-VR) video, single subject → **MatAnyone 2** (experimental)

### POV Mode

Enable **POV mode** for first-person VR content where the camera operator's body is visible. The app uses SAM2 to detect the operator's body on the first frame and excludes it from the matte — only other people are kept.

- With **RVM**: static mask subtraction (fast)
- Automatically refreshes when a scene change is detected

### Multi-Subject (SAM2Matting)

With SAM2Matting selected, the **Subjects** control (1–4) tracks that many people as independent objects. Each subject gets its own SAM2 object; per-frame alphas merge via max. A subject that is briefly lost keeps its previous mask for re-acquisition. RVM (all-people) and MatAnyone 2 (single union mask) ignore this control.

### SBS Stereo Videos

Side-by-side stereo VR videos (aspect ratio ≥ 1.9:1) are **auto-detected**. When SBS is active:

- Each eye is matted independently. RVM batches both eyes in one forward pass (shared model); MatAnyone 2 can optionally run the two eyes on parallel CUDA-stream threads (**Parallel eyes**, ~2× peak VRAM, auto-enabled only on ≥ 24 GB GPUs).
- Results are merged back to SBS format.

You can also manually toggle SBS for files that don't match the auto-detection heuristic.

### Speed Controls

| Control | Options | Effect |
|---------|---------|--------|
| **Matting Rate** | Every frame / Every 2nd frame | Matte every Nth frame and lerp skipped alpha (~2× on 60 fps) |
| **Matting Resolution** | Auto (GPU) / 1080p / 720p / 540p | Per-eye inference resolution; lower = faster |
| **ROI cropping** | On (default) / off | Matte only a window around the subject |
| **Downsample** | 0.125 / 0.25 / 0.5 / 1.0 | RVM recurrent downsample ratio |
| **Chunk size** | 100 / 250 / **500** / 1000 | Frames per extract→matte→flush cycle |
| **Final Encode** | Fast (p2) / Balanced (p4) / Quality (p6) | NVENC preset for the output encode |

### Temporal Smoothing

Reduces frame-to-frame alpha jitter using an exponential moving average (EMA):

| Setting | Weight | Effect |
|---------|--------|--------|
| Off | 1.0 | Raw matte output |
| Light | 0.85 | Subtle stabilisation |
| Medium | 0.7 | Moderate smoothing |
| Heavy | 0.5 | Strong smoothing |

Useful for RVM on VR content where edges can flicker between frames. (Skipped for chunk-level processors such as SAM2Matting.)

### Batch Processing

1. Set up your first file (input, output, settings)
2. Click **+ Queue** to add it to the batch
3. Repeat, or drag multiple files onto the window
4. Click **Start** — files process sequentially

Each file gets its own projection/FOV detection from its filename. Output filenames are auto-generated with the correct DeoVR tags.

### Drag & Drop

- **Single file** → sets it as the current input
- **Multiple files** → adds all to the batch queue

Supported formats: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.wmv`

### DeoVR Alpha Pipeline

With the **DeoVR Alpha Pack** output format, the pipeline follows the official DeoVR alpha-packing spec. By default, assembly is **fused into a single ffmpeg pass** and runs **chaptered in the background during matting** — the equirect→fisheye conversion(s) and the 6-segment alpha pack happen in one filter graph per chapter, then chapters are concatenated (stream copy) and the source audio muxed in.

**For equirectangular sources** (no fisheye tag): source and matte are each decoded once, both converted to fisheye (video with the DeoVR mask, matte without), the alpha is packed (scale to 40 %, red-channel-only, split into 6 segments overlaid into the fisheye corners), and the result is encoded once → `*_FISHEYE{fov}_alpha.mp4`.

**For already-fisheye sources** (`_FISHEYE190`, `_MKX200`, …): no projection conversion — the same 6-segment packing runs directly → `*_{lens_tag}_alpha.mp4`.

Escape hatches (environment variables):

| Variable | Effect |
|----------|--------|
| `VRAUTOMATTE_NO_BG_ASSEMBLY=1` | Single fused pass after matting instead of background chapters |
| `VRAUTOMATTE_NO_FUSED_ASSEMBLY=1` | Legacy multi-pass chain (trim → fisheye → fisheye → pack) |
| `VRAUTOMATTE_CHAPTER_FRAMES=N` | Chapter length in frames (default 6000) |
| `VRAUTOMATTE_NO_STREAM=1` | Force the file-based (PNG) pipeline instead of raw-video streaming |

**Supported DeoVR lens profiles:**

| Filename Tag | Lens | FOV |
|-------------|------|-----|
| `_FISHEYE` | Generic fisheye | 180° |
| `_FISHEYE190` | Canon VR lens | 190° |
| `_MKX200` | Metalenz MKX | 200° |
| `_MKX220` | Metalenz MKX | 220° |
| `_VRCA220` | VRCA lens | 220° |
| `_RF52` | Canon RF 5.2mm | 190° |

**DeoVR encode:** the alpha-pack output is encoded as **AV1** (`av1_nvenc` when supported, SVT-AV1 on CPU otherwise) — the format Quest 3/3S decode natively. A **Codec** selector (HEVC / H.264) is also exposed for the legacy assembly path. **CRF** ranges 10–30 (default 18; lower = better quality, larger file). The DeoVR fisheye `mask8k.png` is downloaded automatically on first use.

### Settings

All settings are saved automatically to `~/.config/vrautomatte/settings.json` (Linux/macOS) or `%APPDATA%/vrautomatte/settings.json` (Windows) and restored on next launch.

## Architecture

```
src/vrautomatte/
├── main.py                    # Entry point
├── pipeline/
│   ├── matte.py               # MatteProcessor protocol, create_processor factory,
│   │                          #   AlphaSmoother (EMA) + POVExclusion wrappers
│   ├── rvm.py                 # RVM torch processor (MobileNetV3 / ResNet50, FP16 recurrent)
│   ├── rvm_onnx.py            # RVM via ONNX Runtime (DirectML / CoreML / CUDA / CPU)
│   ├── matanyone2.py          # MatAnyone 2 processor (experimental, non-VR)
│   ├── sam2matting.py         # SAM2Matting unified tracker+matting (1–4 subjects)
│   ├── sam2_masks.py          # SAM2 first-frame masks + POV / multi-person heuristics
│   ├── scene_detect.py        # Scene-change detector (histogram correlation)
│   ├── roi.py                 # ROICropper — matte only a window around the subject
│   ├── framestream.py         # Raw-video (rgb24) streaming between ffmpeg and matting
│   ├── assembly.py            # ChapteredAssembler — background DeoVR assembly
│   ├── scaler.py              # FrameScaler — VRAM-budget downscale/upscale
│   ├── checkpoint.py          # Resumable-pipeline checkpoints
│   └── runner.py              # Pipeline orchestrator (stream/chunk extract → matte → assemble)
├── ui/
│   ├── main_window.py         # Main GUI window + DeoVR lens detection
│   ├── preview.py             # Dual-pane preview + scrubber
│   ├── themes.py              # Light / dark theme stylesheets
│   └── worker.py              # Background processing / install threads
└── utils/
    ├── ffmpeg.py              # FFmpeg wrappers (extract, fisheye, assemble_deovr, pack_alpha)
    ├── gpu.py                 # Device detection + GPU auto-configuration
    ├── masks.py               # DeoVR mask auto-download
    ├── sbs.py                 # SBS stereo split/merge/detection
    ├── settings.py            # Settings persistence
    └── bootstrap.py           # Correct-CUDA-torch bootstrap before torch import
```

### Processing Pipeline

```
Streaming Pipeline (runner.py + framestream.py, default)
  ffmpeg (NVDEC decode + scale → raw rgb24 pipe)
    → split eyes → MatteProcessor → merged matte (uint8)
    → raw gray pipe → segment encoder
  checkpoint after each segment; concat segments → final matte
  (VRAUTOMATTE_NO_STREAM=1 falls back to the chunked PNG path)

DeoVR Assembly (assembly.py, runs during matting)
  per fully-covered ~6000-frame chapter:
    fused assemble_deovr — one decode, one filter graph
      (equirect→fisheye ×2 + 40 % red-channel 6-segment pack), one encode
  concat chapters (stream copy) + mux source audio → *_alpha.mp4

MatteProcessor Protocol
├── RVMProcessor / RVMOnnxProcessor  — recurrent forward pass, detects ALL people
├── SAM2MattingProcessor             — chunk-level tracker+matting, 1–4 subjects
├── MatAnyone2Processor              — SAM2 masks → InferenceCore (experimental)
├── AlphaSmoother                    — EMA wrapper, reduces jitter
├── ROICropper                       — crops to the subject window, pastes matte back
└── POVExclusionProcessor            — wraps any processor, subtracts POV body mask

GPU Auto-Config (gpu.py)
└── VRAM tier → max_matting_pixels, mem_frames, downsample_ratio, parallel_eyes
```

## Development

```bash
# Run tests
uv run python -m unittest discover -s tests -p "test_*.py"

# Run a specific test file
uv run python -m unittest tests.test_sbs

# Check syntax
uv run python -c "import ast; ast.parse(open('src/vrautomatte/ui/main_window.py').read()); print('OK')"
```

### Test Coverage

97 tests across 8 files. Tests mock GPU and filesystem operations — no GPU or FFmpeg required to run them.

| Test File | Tests | What It Covers |
|-----------|-------|----------------|
| `test_matte_protocol.py` | 14 | Processor protocol, `VARIANTS`, `create_processor` factory, CPU fallback |
| `test_sbs.py` | 14 | SBS detection, frame/matte split & merge |
| `test_pov_mask.py` | 14 | POV body-mask selection, scoring, `POVExclusionProcessor` |
| `test_rvm_onnx.py` | 16 | ONNX Runtime RVM processor (mocked `onnxruntime`) |
| `test_multisubject.py` | 13 | Multi-subject mask selection + alpha plane merging |
| `test_roi.py` | 9 | ROI-restricted matting (`ROICropper`, bbox tracking) |
| `test_integration_matanyone2.py` | 9 | MatAnyone 2 factory/runner wiring, SAM2 masks, re-exports |
| `test_scene_detect.py` | 8 | Scene-change detector, cooldown, threshold, reset |

### Code Conventions

- **Logging**: `from loguru import logger` — no `print()` in committed code
- **Strings**: double quotes preferred
- **Line length**: max 100 characters
- **Imports**: stdlib → third-party → local, alphabetical within groups
- **Type hints**: required for new/modified functions
- **Tests**: `test_*.py`, using `unittest`

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full contributor guide.

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

| Platform | Backend | Status |
|----------|---------|--------|
| NVIDIA | CUDA (PyTorch, FP16) | Full support |
| Apple | MPS (PyTorch, FP16) | Full support |
| AMD / Intel | DirectML (ONNX RVM) | RVM models |
| Any GPU | DirectML / CoreML (ONNX RVM) | RVM models |
| CPU | — | Fallback (slower) |

MatAnyone 2 and SAM2Matting run on the PyTorch backend (CUDA/MPS/CPU).

## Model Downloads

Models are downloaded automatically on first use and cached locally:

| Model | Size | Cache Location |
|-------|------|----------------|
| RVM MobileNetV3 (torch + ONNX) | ~15 MB | `~/.cache/vrautomatte/models/` |
| RVM ResNet50 (torch + ONNX) | ~55 MB | `~/.cache/vrautomatte/models/` |
| SAM2Matting (tiny) | ~160 MB | `~/.cache/vrautomatte/sam2matting/` |
| MatAnyone 2 | ~2 GB | Managed by Hugging Face Hub |
| SAM2 | ~400 MB | Managed by Hugging Face Hub |
| DeoVR `mask8k.png` | ~2 MB | `~/.cache/vrautomatte/masks/` |

## License

[GPL-3.0-or-later](LICENSE) — © 2026 VRAutoMatte Contributors.

VRAutoMatte is **copyleft**: you may use, study, modify, and redistribute it, but if you distribute a modified version you must also release your source under the GPL, so improvements flow back to the community.

### Third-party model licenses

- **SAM2Matting** model weights are **CC BY-NC-SA 4.0 (non-commercial)**. The `sam2matting` extra is an optional install; do **not** ship it in a commercial product.
- **MatAnyone 2** and **SAM2** are distributed under their respective upstream licenses (see their repositories).
- RVM weights follow the [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) license.

## Acknowledgements

- [Robust Video Matting](https://github.com/PeterL1n/RobustVideoMatting) — recurrent all-people matting architecture
- [SAM2Matting](https://github.com/FudanCVL/SAM2Matting) — unified tracker + matting
- [MatAnyone 2](https://github.com/pq-yang/MatAnyone2) — CVPR 2026 SOTA video matting
- [SAM2](https://github.com/facebookresearch/sam2) — Segment Anything Model 2 for first-frame masks
- [DeoVR](https://deovr.com/) — VR video player with alpha passthrough support
