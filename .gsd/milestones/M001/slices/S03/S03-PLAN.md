# S03: Preview Scrubber Seek

**Goal:** User can drag the preview scrubber to any frame and see the source frame instantly.
**Demo:** User drags scrubber before or during processing → source frame at that position appears in the preview pane.

## Must-Haves

- Scrubber slider with frame range matching video length
- Frame extraction via FFmpeg `extract_frame()` on scrub
- Preview updates with source frame at scrubbed position
- Works before and during processing

## Verification

- Manual: drag scrubber after loading a video, source frame updates
- `extract_frame()` in ffmpeg.py already tested via existing integration

## Tasks

- [x] **T01: Scrubber widget + frame extraction** `est:30m`
  - PreviewWidget scrubber signal, MainWindow._on_frame_scrubbed handler, extract_frame call
  - Done: implemented and wired

## Files Touched

- `src/vrautomatte/ui/preview.py`
- `src/vrautomatte/ui/main_window.py`
- `src/vrautomatte/utils/ffmpeg.py`
