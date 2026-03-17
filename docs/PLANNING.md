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

## Future Enhancements

- [ ] **MatAnyone 2** support — CVPR 2026 state-of-the-art, better edge quality
      (requires SAM2 for first-frame mask auto-generation)
- [ ] **VideoMaMa** support — Adobe's generative matting approach
- [x] **Batch processing** — queue multiple videos
- [x] **Auto-download DeoVR mask** — bundle or download mask8k.png
- [ ] **SBS split mode** — process left/right eyes independently for better quality
- [x] **Audio preservation** — ensure audio track carries through all steps
- [x] **Preview scrubber** — frame scrubber with ETA/FPS display
- [x] **Settings persistence** — remember last-used settings
- [ ] **Drag & drop** — drop video files onto the window
- [ ] **ETA per-batch** — overall batch ETA, not just per-file

## Research Notes

### Video Matting Models Compared

| Model | Type | Input Required | Quality | Speed | Year |
|-------|------|---------------|---------|-------|------|
| **RVM** | Recurrent | None (auto) | Good | Very fast | 2021 |
| **MatAnyone** | Memory-based | First-frame mask | Very good | Medium | 2025 |
| **MatAnyone 2** | Memory-based + MQE | First-frame mask | Best | Medium | 2026 |
| **VideoMaMa** | Diffusion-based | Coarse mask | Very good | Slow | 2026 |
| **rembg** | Per-frame | None | Basic | Fast | 2022 |

### DeoVR Alpha Format

- Alpha passthrough only works with **fisheye projection**
- Alpha channel packed **vertically** (video on top, matte on bottom)
- Matte uses **red channel only**
- Filename must contain `_ALPHA` for DeoVR to recognize it
- Equirectangular → fisheye conversion via FFmpeg v360 filter
- Requires mask8k.png from DeoVR for proper circle masking
