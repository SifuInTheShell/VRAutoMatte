---
slice: S03
title: Preview Scrubber Seek
status: complete
started: 2026-03-15
completed: 2026-03-17
tasks_completed: 1
tasks_total: 1
---

# S03 Summary: Preview Scrubber Seek

## What Was Built

Interactive frame scrubber in the preview panel that lets users seek to any frame.

### Key Components

- **`PreviewWidget.scrubber`** — QSlider with frame range, emits `frame_scrubbed(int)` signal
- **`PreviewWidget.frame_label`** — "Frame N / Total" counter (monospace)
- **`PreviewWidget.eta_label`** — ETA / elapsed time display
- **`MainWindow._on_frame_scrubbed()`** — Extracts frame via FFmpeg `extract_frame()`, displays in source pane

### Implementation

On scrub, extracts the frame at `frame_num - 1` to a temp PNG via FFmpeg `-vf "select=eq(n\,{frame})"`, loads with PIL, converts to numpy, updates preview. Temp file cleaned up immediately.

Scrubber is disabled until a video is loaded. During processing, scrubber position updates automatically via progress callbacks. User can override by dragging.

## What's Next

Nothing — feature complete for M001 scope.
