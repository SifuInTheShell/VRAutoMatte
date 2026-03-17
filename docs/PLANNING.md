# VRAutoMatte — Planning

## Vision

Automated video matting and alpha channel generation for VR passthrough content.
Take any VR video, use AI to separate human person(s) from the background,
and produce a file ready for DeoVR passthrough playback on Quest headsets.

## Decisions Made

### D1: GUI Framework → PySide6

- Cross-platform (Windows, Linux, macOS)
- Same language as AI pipeline (Python + PyTorch)
- Direct GPU access, no web bridge needed
- Native file dialogs, sliders, image display

### D2: Primary Matting Model → RVM (Robust Video Matting)

- Fully automatic — no first-frame mask needed
- Recurrent architecture gives temporal consistency (no flicker)
- Real-time capable (4K @ 76fps on GTX 1080 Ti)
- Well-tested, simple API, small model size
- MobileNetV3 variant for speed, ResNet50 for quality

### D3: Output Format → DeoVR Alpha Pack

- Fisheye projection with alpha channel packed vertically
- Requires `_ALPHA` in filename for DeoVR recognition
- Uses DeoVR's documented FFmpeg conversion pipeline
- Red channel only for the alpha matte

### D4: MatAnyone 2 + SAM2 Integration

- CVPR 2026 SOTA matting model with best edge quality
- SAM2 auto-generates first-frame mask — zero manual input
- Sequential GPU loading (SAM2 → unload → MatAnyone 2) to fit in 6-8GB VRAM
- Protocol-based processor design for model interchangeability

### D5: POV Body Removal

- SAM2 detects POV body via inverse scoring heuristic (bottom-heavy, edge-touching)
- MatAnyone 2: instance-level exclusion (quality)
- RVM: static mask subtraction via POVExclusionProcessor wrapper (fast)

### D6: Scene Change Detection

- Histogram correlation (Pearson) per frame
- Threshold 0.4 with 30-frame cooldown
- Triggers SAM2 mask refresh on cuts or major position changes

## Feature Status

- [x] **RVM matting** — MobileNetV3 and ResNet50 variants
- [x] **MatAnyone 2** — CVPR 2026 SOTA, SAM2 auto-mask
- [x] **POV body removal** — auto-detect and exclude camera operator
- [x] **Scene change detection** — auto-refresh masks on cuts
- [x] **SBS stereo processing** — per-eye matting with auto-detection
- [x] **DeoVR alpha packing** — equirect → fisheye → red channel → pack
- [x] **Batch processing** — queue and process multiple videos
- [x] **Live preview** — dual-pane with scrubber, ETA, FPS
- [x] **Drag & drop** — single file or batch
- [x] **Settings persistence** — all preferences saved
- [x] **Light / dark theme** — toggle with persistence
- [x] **Auto-download DeoVR mask** — fetched on first use
- [x] **Audio preservation** — audio track carries through
- [ ] **VideoMaMa support** — Adobe's generative matting (future)
- [ ] **Manual click-to-segment** — user-guided SAM2 prompting (future)
- [ ] **Per-batch ETA** — overall batch progress estimate (future)

## Research Notes

### Video Matting Models Compared

| Model | Type | Input Required | Quality | Speed | Year |
|-------|------|---------------|---------|-------|------|
| **RVM** | Recurrent | None (auto) | Good | Very fast | 2021 |
| **MatAnyone 2** | Memory-based + MQE | Auto (SAM2) | Best | Medium | 2026 |
| **VideoMaMa** | Diffusion-based | Coarse mask | Very good | Slow | 2026 |

### DeoVR Alpha Format

- Alpha passthrough only works with **fisheye projection**
- Alpha channel packed **vertically** (video on top, matte on bottom)
- Matte uses **red channel only**
- Filename must contain `_ALPHA` for DeoVR to recognize it
- Equirectangular → fisheye conversion via FFmpeg v360 filter
- Requires mask8k.png from DeoVR for proper circle masking
