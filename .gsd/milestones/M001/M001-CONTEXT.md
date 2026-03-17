# M001: Next-Gen Matting & UX Enhancements

**Gathered:** 2026-03-17
**Status:** Ready for planning

## Project Description

VRAutoMatte is a desktop application that uses AI to automatically generate alpha mattes from VR videos for DeoVR passthrough playback. This milestone adds the CVPR 2026 MatAnyone 2 matting model (higher quality edges), SBS stereo per-eye processing, preview scrubber seek, and drag & drop file input.

## Why This Milestone

The existing RVM model works but produces segmentation-like edges — MatAnyone 2 preserves fine details like hair and semi-transparent regions. SBS per-eye processing improves matte quality for stereo VR content. Preview seek and drag & drop are high-value UX improvements that complete the interaction model.

## User-Visible Outcome

### When this milestone is complete, the user can:

- Select "MatAnyone 2" from the model dropdown, hit Start, and get higher-quality mattes automatically
- Process SBS stereo VR videos with per-eye matting (auto-detected or manually toggled)
- Drag the preview scrubber to inspect any source frame before or during processing
- Drop video files onto the window to set input or batch-queue multiple files
- Use all features on CPU-only machines (slower but functional)

### Entry point / environment

- Entry point: `uv run vrautomatte` (PySide6 desktop app)
- Environment: local dev / desktop (Windows, macOS, Linux)
- Live dependencies involved: none (all processing is local)

## Completion Class

- Contract complete means: models load and produce mattes, SBS split/recombine works, UI interactions respond correctly
- Integration complete means: full pipeline from video input → MatAnyone 2 matte → DeoVR alpha pack works end-to-end
- Operational complete means: none (desktop app, no service lifecycle)

## Final Integrated Acceptance

To call this milestone complete, we must prove:

- MatAnyone 2 produces a matte video from a VR video with zero manual masking steps
- An SBS stereo video is correctly split, matted per-eye, and recombined
- Preview scrubber shows source frames responsively during active processing
- Drag & drop works for single and multi-file cases
- All of the above work on both GPU and CPU-only machines

## Risks and Unknowns

- **MatAnyone 2 API stability** — the library was released March 2026, API may change. InferenceCore's frame-by-frame `step()` method needs verification for live preview callbacks.
- **SAM2 auto-mask quality on VR content** — equirectangular projection distorts people; SAM2's automatic mask generator may struggle with distorted body shapes at frame edges.
- **GPU memory pressure** — SAM2 + MatAnyone 2 together could exceed 6-8GB consumer GPU VRAM. Sequential loading with explicit cleanup is the mitigation.
- **MatAnyone 2 CPU performance** — transformer-based model on CPU will be very slow for long videos (potentially 10-30x slower than GPU). Acceptable but needs clear UX communication.

## Existing Codebase / Prior Art

- `pipeline/matte.py` — `RVMProcessor` with `process_frame()` pattern; new MatAnyone 2 processor should follow same interface
- `pipeline/runner.py` — orchestrator with `PipelineConfig.is_sbs` field (unused); `split_sbs()`/`stack_sbs()` in ffmpeg.py ready
- `ui/preview.py` — `frame_scrubbed` signal and `set_scrubber_enabled()` exist but aren't wired
- `utils/ffmpeg.py` — `extract_frame()` exists for scrubber seek; `split_sbs()`/`stack_sbs()` ready for SBS
- `utils/gpu.py` — `get_device()` already handles CUDA/MPS/CPU fallback
- `utils/settings.py` — simple JSON persistence with defaults pattern

> See `.gsd/DECISIONS.md` for all architectural and pattern decisions — it is an append-only register; read it during planning, append to it during execution.

## Relevant Requirements

- R001-R003 — MatAnyone 2 + SAM2 integration (S01)
- R004-R005 — SBS per-eye processing (S02)
- R006 — Preview scrubber seek (S03)
- R007 — Drag & drop (S04)
- R008-R011 — Cross-cutting: GPU memory, RVM compat, settings, CPU fallback (S01/S02)

## Scope

### In Scope

- MatAnyone 2 as additional model option (alongside RVM)
- SAM2 automatic first-frame mask generation
- SBS per-eye split/matte/recombine pipeline
- SBS auto-detection with manual override toggle
- Preview scrubber source frame seeking
- Drag & drop file input (single → input, multiple → batch)
- CPU fallback for all models with UI warning
- Settings persistence for new options

### Out of Scope / Non-Goals

- VideoMaMa or other matting models
- Manual click-to-segment first frame (deferred)
- Per-batch ETA (deferred)
- Model fine-tuning or training
- Cloud/remote processing

## Technical Constraints

- PySide6 for all GUI work (no web UI)
- PyTorch for all model inference
- FFmpeg via subprocess for video processing
- Module size cap: ≤150 LOC target, 200 LOC hard cap
- MatAnyone 2 licensed under NTU S-Lab License 1.0

## Integration Points

- **MatAnyone 2 (PyPI/GitHub)** — `matanyone2` package installed via uv, `InferenceCore` for video processing
- **SAM2 (PyPI)** — `sam2` package, `SAM2ImagePredictor` or `SAM2AutomaticMaskGenerator` for first-frame mask
- **FFmpeg** — existing subprocess wrappers for SBS split/stack and frame extraction

## Open Questions

- Does MatAnyone 2's `InferenceCore` expose a per-frame `step()` method for live preview? The MatAnyone v1 inference script uses `processor.step()` — need to verify v2 compatibility.
- Which SAM2 model variant to use? `hiera-tiny` for CPU speed, `hiera-large` for GPU quality? Or always use `hiera-small` as a balance?
- Does SAM2's automatic mask generator reliably find people in equirectangular VR frames, or do we need to extract a center crop first?
