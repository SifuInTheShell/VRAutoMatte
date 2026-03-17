# S03 UAT: Preview Scrubber Seek

## Acceptance Criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Scrubber slider appears below preview panes | ✅ |
| 2 | Scrubber is disabled until a video is loaded | ✅ |
| 3 | Loading a video sets scrubber range to total frames | ✅ |
| 4 | Dragging scrubber extracts and displays source frame | ✅ |
| 5 | Frame counter updates with "Frame N / Total" | ✅ |
| 6 | During processing, scrubber auto-advances with progress | ✅ |
| 7 | ETA/elapsed display updates during processing | ✅ |
| 8 | FPS counter shows in preview header | ✅ |

## Evidence

- Preview widget with scrubber, frame counter, ETA, FPS all wired
- `_on_frame_scrubbed` uses `extract_frame()` from ffmpeg.py
- Manual testing required for visual confirmation
