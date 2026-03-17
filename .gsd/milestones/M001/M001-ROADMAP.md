# M001: Next-Gen Matting & UX Enhancements

**Vision:** Add MatAnyone 2 (CVPR 2026 SOTA) as an alternative matting model with automatic SAM2 first-frame masking, SBS stereo per-eye processing, preview scrubber seek, and drag & drop file input — all with CPU fallback.

## Success Criteria

- User can select MatAnyone 2 from the model dropdown and produce mattes without any manual masking step
- SBS stereo VR videos are auto-detected and processed per-eye with correct recombination
- Preview scrubber responds to user drag with source frame display before and during processing
- Dropping files onto the window sets input or adds to batch queue
- All features work on CPU-only machines (slower, with UI warning)
- Existing RVM pipeline works identically to before

## Key Risks / Unknowns

- MatAnyone 2 API stability — library released March 2026, API and per-frame interface not yet verified in our integration
- SAM2 auto-mask quality on equirectangular VR content — distorted projection may confuse person detection
- GPU memory pressure — SAM2 + MatAnyone 2 on consumer GPUs (6-8GB VRAM) may require sequential loading

## Proof Strategy

- MatAnyone 2 API + SAM2 auto-mask → retire in S01 by proving the full pipeline produces a matte from a video with zero user input
- SBS split/recombine correctness → retire in S02 by proving an SBS video round-trips through split → matte → recombine with correct geometry
- GPU memory management → retire in S01 by proving sequential SAM2 → MatAnyone 2 loading doesn't OOM on target hardware

## Verification Classes

- Contract verification: unittest for processor interfaces, FFmpeg wrapper correctness, settings persistence
- Integration verification: end-to-end pipeline run producing a valid matte video file
- Operational verification: none (desktop app)
- UAT / human verification: visual inspection of matte quality, scrubber responsiveness, drag & drop behavior

## Milestone Definition of Done

This milestone is complete only when all are true:

- MatAnyone 2 is selectable and produces mattes end-to-end with auto-masking
- SBS stereo videos are correctly split, matted per-eye, and recombined
- Preview scrubber responds to drag with source frames before and during processing
- Drag & drop works for single file (→ input) and multiple files (→ batch queue)
- CPU fallback works for all models with visible UI warning
- Existing RVM variants (mobilenetv3, resnet50) produce identical results to before
- Settings persist for new model selection and SBS toggle
- All success criteria re-checked against live behavior

## Requirement Coverage

- Covers: R001, R002, R003, R004, R005, R006, R007, R008, R009, R010, R011
- Partially covers: none
- Leaves for later: R012 (manual click-to-segment), R013 (per-batch ETA)
- Orphan risks: none

## Slices

- [x] **S01: MatAnyone 2 + SAM2 Integration** `risk:high` `depends:[]`
  > After this: user selects "MatAnyone 2" from model dropdown, hits Start, video is auto-matted with higher edge quality than RVM — works on both GPU and CPU

- [ ] **S02: SBS Stereo Per-Eye Processing** `risk:medium` `depends:[]`
  > After this: user loads an SBS stereo VR video, SBS is auto-detected, matting runs per-eye independently, output is correct stereo — works with both RVM and MatAnyone 2

- [ ] **S03: Preview Scrubber Seek** `risk:low` `depends:[]`
  > After this: user drags the preview scrubber before or during processing, source frame at that position appears instantly in the preview pane

- [ ] **S04: Drag & Drop File Input** `risk:low` `depends:[]`
  > After this: user drags one video file onto the window and it becomes the input; drags multiple files and they all go into the batch queue

## Boundary Map

### S01 → S02

Produces:
- `pipeline/matte.py` → `MatteProcessor` protocol/ABC with `process_frame(frame) → matte` and `reset()` methods
- `pipeline/matte.py` → `MatAnyone2Processor` class implementing the protocol (alongside existing `RVMProcessor`)
- `pipeline/matte.py` → `create_processor(variant, downsample_ratio, device)` factory function
- `pipeline/runner.py` → `PipelineConfig.model_variant` expanded to include "matanyone2"
- `utils/gpu.py` → `get_device()` unchanged but verified with new models

Consumes:
- nothing (first slice)

### S01 → S03

Produces:
- No direct dependency — S03 wires preview scrubber to FFmpeg extract_frame, independent of matting model

Consumes:
- nothing

### S01 → S04

Produces:
- No direct dependency — S04 is pure UI drag & drop

Consumes:
- nothing

### S02 → (standalone)

Produces:
- `pipeline/runner.py` → SBS-aware pipeline stages (split → matte each eye → recombine)
- `ui/main_window.py` → SBS toggle checkbox with auto-detection
- `utils/settings.py` → `is_sbs` setting persisted

Consumes from S01:
- `create_processor()` factory function — SBS processing uses whatever model is selected
