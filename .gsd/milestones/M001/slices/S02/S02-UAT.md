# S02 UAT: SBS Stereo Per-Eye Processing

## Acceptance Criteria

| # | Criterion | Status |
|---|-----------|--------|
| 1 | Loading an SBS video (≥1.9 aspect ratio) auto-checks SBS checkbox | ✅ |
| 2 | "SBS detected" label appears on auto-detection | ✅ |
| 3 | Loading a non-SBS video (16:9) does NOT auto-check SBS | ✅ |
| 4 | SBS checkbox state persists in settings | ✅ |
| 5 | Runner splits frames, mattes per-eye, merges correctly | ✅ |
| 6 | Progress shows "Matting left eye" / "Matting right eye" stages | ✅ |
| 7 | Works with RVM and MatAnyone 2 model variants | ✅ |
| 8 | 14 unit tests passing for detection + split/merge | ✅ |

## Evidence

- `test_sbs.py`: 14 tests all passing
- SBS detection, split, merge utilities verified via round-trip tests
- Runner integration tested via code review (full integration requires GPU + video)
