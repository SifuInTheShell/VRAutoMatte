---
slice: S04
title: Drag & Drop File Input
status: complete
started: 2026-03-15
completed: 2026-03-17
tasks_completed: 1
tasks_total: 1
---

# S04 Summary: Drag & Drop File Input

## What Was Built

Drag-and-drop support for video files on the main window.

### Implementation

- **`MainWindow._VIDEO_EXTENSIONS`** — Set of accepted extensions: `.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`, `.wmv`
- **`dragEnterEvent()`** — Accepts drag only if URLs contain at least one video file
- **`dropEvent()`** — Single file: sets input path, auto-generates output, shows video info. Multiple files: adds all to batch queue via `_add_file_to_batch()`.
- **`setAcceptDrops(True)`** called in `__init__`

### Design

Simple and predictable: one file = input, multiple = batch. No ambiguity. Non-video files are silently filtered out.

## What's Next

Nothing — feature complete for M001 scope.
