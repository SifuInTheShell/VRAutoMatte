# Requirements

This file is the explicit capability and coverage contract for the project.

## Active

### R001 — MatAnyone 2 model available as selection option
- Class: core-capability
- Status: active
- Description: User can select "MatAnyone 2" from the matting model dropdown alongside existing RVM variants
- Why it matters: MatAnyone 2 (CVPR 2026) produces significantly higher quality mattes with better edge detail than RVM
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: unmapped
- Notes: Model auto-downloads ~400MB checkpoint on first use via HuggingFace

### R002 — SAM2 auto-generates first-frame mask for MatAnyone 2
- Class: core-capability
- Status: active
- Description: When MatAnyone 2 is selected, SAM2 automatically detects and segments the person in frame 1 — no user input required
- Why it matters: Keeps the zero-input UX consistent with RVM — user just hits Start
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: unmapped
- Notes: Use SAM2AutomaticMaskGenerator for auto-detection; use smaller SAM2 variant (hiera-tiny or hiera-small) on CPU

### R003 — Model weights auto-download on first use
- Class: launchability
- Status: active
- Description: MatAnyone 2 checkpoint and SAM2 weights download automatically on first use, with progress indication
- Why it matters: No manual model download steps for the user
- Source: inferred
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: unmapped
- Notes: Both models support from_pretrained with auto-download; cache to ~/.cache/vrautomatte/

### R004 — SBS stereo videos processed per-eye
- Class: core-capability
- Status: active
- Description: Side-by-side stereo VR videos are split into left/right eyes, matted independently, then recombined
- Why it matters: Per-eye matting avoids quality loss at the center seam where left/right eyes meet
- Source: user
- Primary owning slice: M001/S02
- Supporting slices: none
- Validation: unmapped
- Notes: FFmpeg split_sbs() and stack_sbs() helpers already exist in utils/ffmpeg.py

### R005 — SBS auto-detection with manual override
- Class: primary-user-loop
- Status: active
- Description: App auto-detects SBS stereo based on aspect ratio (2:1 or wider) with a toggle for manual override
- Why it matters: Most VR videos are SBS; auto-detection reduces friction while toggle handles edge cases
- Source: user
- Primary owning slice: M001/S02
- Supporting slices: none
- Validation: unmapped
- Notes: Detection heuristic: width/height >= 1.9 suggests SBS

### R006 — Preview scrubber seeks source frames
- Class: primary-user-loop
- Status: active
- Description: User can drag the preview scrubber before and during processing to view the source frame at any position
- Why it matters: Lets users inspect their video before committing to processing and monitor progress non-linearly
- Source: user
- Primary owning slice: M001/S03
- Supporting slices: none
- Validation: unmapped
- Notes: Uses existing extract_frame() from ffmpeg.py; matte pane shows latest processed frame only

### R007 — Drag & drop file input
- Class: primary-user-loop
- Status: active
- Description: Dropping one video file onto the window sets it as current input; dropping multiple files adds them all to the batch queue
- Why it matters: Faster file input than Browse dialogs, especially for batch workflows
- Source: user
- Primary owning slice: M001/S04
- Supporting slices: none
- Validation: unmapped
- Notes: Standard PySide6 dragEnterEvent/dropEvent on MainWindow

### R008 — GPU memory managed across SAM2 → MatAnyone 2 pipeline
- Class: quality-attribute
- Status: active
- Description: SAM2 is unloaded from GPU memory before MatAnyone 2 loads, preventing OOM on consumer GPUs
- Why it matters: Both models together could exceed VRAM on 6-8GB consumer GPUs
- Source: inferred
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: unmapped
- Notes: Load SAM2, generate mask, del SAM2 + torch.cuda.empty_cache(), then load MatAnyone 2

### R009 — Existing RVM pipeline unchanged
- Class: continuity
- Status: active
- Description: All existing RVM matting functionality (mobilenetv3, resnet50) continues to work identically
- Why it matters: No regressions for existing users
- Source: inferred
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: unmapped
- Notes: RVM code paths must not be altered by MatAnyone 2 addition

### R010 — Settings persistence for new options
- Class: continuity
- Status: active
- Description: New settings (model selection including MatAnyone 2, SBS toggle) are saved and restored between sessions
- Why it matters: Consistency with existing settings persistence behavior
- Source: inferred
- Primary owning slice: M001/S02
- Supporting slices: M001/S01
- Validation: unmapped
- Notes: Extend existing settings.py defaults and save/restore in main_window.py

### R011 — CPU fallback for all models with UI warning
- Class: quality-attribute
- Status: active
- Description: MatAnyone 2 and SAM2 run on CPU when no GPU is detected, with a visible UI warning about slower speed
- Why it matters: Users without GPUs can still use all features, just slower
- Source: user
- Primary owning slice: M001/S01
- Supporting slices: none
- Validation: unmapped
- Notes: SAM2 uses smaller variant (hiera-tiny) on CPU for mask generation; MatAnyone 2 works on CPU via PyTorch device parameter; UI shows "⚠️ No GPU — processing will be slower"

## Deferred

### R012 — Manual click-to-segment first frame
- Class: core-capability
- Status: deferred
- Description: Option to display first frame and let user click on person to guide SAM2 mask generation
- Why it matters: More control for edge cases where auto-detection picks wrong person
- Source: research
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred — auto-mask covers the common case; manual option adds UI complexity

### R013 — Per-batch ETA
- Class: primary-user-loop
- Status: deferred
- Description: Show overall batch ETA across all queued files, not just per-file ETA
- Why it matters: Better time estimation for large batch jobs
- Source: user
- Primary owning slice: none
- Supporting slices: none
- Validation: unmapped
- Notes: Deferred — per-file ETA exists; batch ETA is nice-to-have

## Out of Scope

### R014 — VideoMaMa integration
- Class: core-capability
- Status: out-of-scope
- Description: Adobe's diffusion-based matting approach
- Why it matters: Prevents scope creep into a third matting model
- Source: research
- Primary owning slice: none
- Supporting slices: none
- Validation: n/a
- Notes: Slow inference, different paradigm; revisit if demand exists

## Traceability

| ID | Class | Status | Primary owner | Supporting | Proof |
|---|---|---|---|---|---|
| R001 | core-capability | active | M001/S01 | none | unmapped |
| R002 | core-capability | active | M001/S01 | none | unmapped |
| R003 | launchability | active | M001/S01 | none | unmapped |
| R004 | core-capability | active | M001/S02 | none | unmapped |
| R005 | primary-user-loop | active | M001/S02 | none | unmapped |
| R006 | primary-user-loop | active | M001/S03 | none | unmapped |
| R007 | primary-user-loop | active | M001/S04 | none | unmapped |
| R008 | quality-attribute | active | M001/S01 | none | unmapped |
| R009 | continuity | active | M001/S01 | none | unmapped |
| R010 | continuity | active | M001/S02 | M001/S01 | unmapped |
| R011 | quality-attribute | active | M001/S01 | none | unmapped |
| R012 | core-capability | deferred | none | none | unmapped |
| R013 | primary-user-loop | deferred | none | none | unmapped |
| R014 | core-capability | out-of-scope | none | none | n/a |

## Coverage Summary

- Active requirements: 11
- Mapped to slices: 11
- Validated: 0
- Unmapped active requirements: 0
