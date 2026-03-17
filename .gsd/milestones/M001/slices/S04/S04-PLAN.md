# S04: Drag & Drop File Input

**Goal:** Users can drag video files onto the window to set input or populate the batch queue.
**Demo:** Drag one video → becomes input. Drag multiple videos → all added to batch queue.

## Must-Haves

- `dragEnterEvent` accepts only video file extensions
- Single file drop → sets input path, auto-generates output path, shows video info
- Multiple file drop → adds all to batch queue
- Supported extensions: .mp4, .mkv, .mov, .avi, .webm, .wmv

## Verification

- Manual: drag single file onto window, verify it becomes input; drag multiple, verify batch queue populated

## Tasks

- [x] **T01: Drag & drop handlers** `est:20m`
  - dragEnterEvent + dropEvent on MainWindow, _VIDEO_EXTENSIONS filter
  - Done: implemented with single/multi file handling

## Files Touched

- `src/vrautomatte/ui/main_window.py`
