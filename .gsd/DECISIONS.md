# Decisions Register

<!-- Append-only. Never edit or remove existing rows.
     To reverse a decision, add a new row that supersedes it.
     Read this file at the start of any planning or research phase. -->

| # | When | Scope | Decision | Choice | Rationale | Revisable? |
|---|------|-------|----------|--------|-----------|------------|
| D001 | M001 | library | Matting model addition | MatAnyone 2 (CVPR 2026) | SOTA edge quality, preserves fine details like hair; existing RVM kept as fast option | No |
| D002 | M001 | library | First-frame mask generation | SAM2 automatic mask generator | Zero-input UX — no manual click needed; consistent with RVM's hands-off approach | Yes — if auto-mask quality is poor on VR content |
| D003 | M001 | arch | SAM2 + MatAnyone 2 GPU memory strategy | Sequential loading — unload SAM2 before loading MatAnyone 2 | Consumer GPUs (6-8GB) can't hold both simultaneously | No |
| D004 | M001 | arch | MatAnyone 2 first-frame mask workflow | Fully automatic via SAM2 | User chose zero-input over interactive click-to-segment | Yes — if auto-detection fails on VR content |
| D005 | M001 | arch | SBS matting approach | Per-eye split → matte independently → recombine | Better matte quality at center seam than full-frame matting | No |
| D006 | M001 | arch | SBS detection method | Auto-detect via aspect ratio (≥1.9:1) with manual override toggle | Most VR content is SBS; toggle handles edge cases | Yes — if heuristic causes false positives |
| D007 | M001 | arch | Preview scrubber seek scope | Source frames only (matte shows latest processed) | Avoids matte caching complexity; extract_frame() already exists | Yes — if users request matte seek |
| D008 | M001 | arch | Drag & drop behavior | Single file → input, multiple files → batch queue | Matches expected desktop app behavior | No |
| D009 | M001 | arch | CPU fallback strategy | All models support CPU with smaller SAM2 variant on CPU + UI warning | Users without GPUs can still use all features; SAM2 only runs once per video so CPU cost is acceptable | No |
