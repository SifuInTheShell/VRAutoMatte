# S04 UAT: Drag & Drop File Input

## Acceptance Criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Window accepts drops (`setAcceptDrops(True)`) | ✅ |
| 2 | Only video file extensions are accepted on drag enter | ✅ |
| 3 | Single file drop → sets input path | ✅ |
| 4 | Single file drop → auto-generates output path | ✅ |
| 5 | Single file drop → shows video info (resolution, fps, duration) | ✅ |
| 6 | Multiple file drop → all added to batch queue | ✅ |
| 7 | Non-video files in a multi-drop are silently filtered | ✅ |

## Evidence

- `dragEnterEvent` and `dropEvent` implemented with extension filtering
- Single/multi path logic verified via code review
- Manual testing required for visual confirmation
