# Performance Analysis — Why Matting Runs at ~1 FPS and How to Fix It

**Context:** RTX 5080 Laptop (16 GB VRAM), 8K SBS HEVC input (5800x2900 @ 60fps),
MatAnyone2 pipeline. Observed throughput: ~1 frame/sec end-to-end, i.e. an
hour of 60fps video (216,000 frames) takes ~60 hours.

**Conclusion up front:** the GPU is *not* the bottleneck. Profiling the code
path shows the model inference accounts for roughly 20–30% of each frame's
wall time. The other 70–80% is CPU-side image plumbing: PNG encoding/decoding
of full-resolution 8K frames, single-threaded PIL LANCZOS resizes, and disk
round-trips — all executed serially on one thread while the GPU idles.
Fixing the data path (not the model) is worth an estimated **5–10x**; GPU-side
tuning adds another **1.5–3x** on top.

---

## 1. Where each second goes today

The per-frame hot path in `_run_chunked_pipeline()` (`pipeline/runner.py`) for
8K SBS content, with the 16 GB auto-config (`max_matting_pixels = 1920*1080`
per eye → each 2900x2900 eye matted at ~1440x1440):

| # | Step | Where | Est. cost/frame | Device |
|---|------|-------|-----------------|--------|
| 1 | ffmpeg decodes 8K HEVC and writes a **full-res 5800x2900 PNG** to disk (amortized over chunk) | `_extract_chunk` | 100–300 ms | CPU (PNG deflate), GPU idle |
| 2 | `Image.open(...).convert("RGB")` — decode that 8K PNG back into RAM | runner loop | 200–400 ms | CPU, single thread |
| 3 | `split_frame()` — two 25 MB array copies | `utils/sbs.py` | ~20 ms | CPU |
| 4 | 2 × PIL LANCZOS downscale 2900² → 1440² | `FrameScaler.downscale` | 120–250 ms | CPU, single thread |
| 5 | 2 × MatAnyone2 `step()` at 1440x1440 FP16 | `matanyone2.py` | 200–400 ms | **GPU (the only GPU step)** |
| 6 | 2 × PIL LANCZOS upscale matte 1440² → 2900² | `FrameScaler.upscale_matte` | 100–200 ms | CPU |
| 7 | `merge_mattes` + PIL **saves a full-res 5800x2900 grayscale PNG** | runner loop | 150–400 ms | CPU (PNG deflate) |

Total: ~900–2000 ms/frame → the observed ~1 FPS. Three structural problems:

1. **Everything is serial.** Extraction of a whole 500-frame chunk completes
   before matting starts (GPU idle), then matting runs with no decode-ahead
   (CPU cores idle). Steps 2–7 run sequentially on one Python thread.
2. **Full-resolution PNG round-trips.** Every frame is compressed to an 8K PNG,
   written to disk, read back, decompressed, downscaled — and every matte is
   upscaled back to 8K and compressed to PNG again, only for ffmpeg to
   immediately re-decode it for the segment encode. Six full-res
   compress/decompress passes per frame for data the model consumes at
   ~1440x1440.
3. **The 8K matte is wasted work anyway.** In the DeoVR path, `pack_alpha()`
   scales the matte down to 40% of frame size. We upscale 1440² → 2900² in
   Python (slow) so ffmpeg can later downscale it again.

The FPS counter shown in the UI (`_emit_matte_progress`) measures from
`_matte_start_time`, which is set once before the chunk loop — so displayed
FPS includes all extraction stalls. It is a true end-to-end number, which is
what the user experiences.

---

## 2. Tier 1 — Fix the data path (biggest win, est. 5–10x combined)

### 2.1 Extract at the matting resolution, not 8K  ⭐ cheapest big win

`FrameScaler` already knows the target size before extraction starts. Move the
downscale into ffmpeg:

```python
# _extract_chunk — after computing scaler.target_size (per-eye tw, th)
full_w = tw * 2 if use_sbs else tw   # SBS: both eyes side by side
cmd = [
    "ffmpeg", "-y",
    *_hwaccel_args(),
    "-ss", f"{timestamp:.6f}",
    "-i", str(input_path),
    "-frames:v", str(num_frames),
    "-vf", f"scale={full_w}:{th}:flags=lanczos",
    str(frames_dir / "frame_%06d.png"),
]
```

Effects:
- PNG files shrink ~8x (2880x1440 instead of 5800x2900) → encode, disk write,
  and `Image.open` decode all shrink proportionally (steps 1–2).
- `FrameScaler.downscale` becomes a no-op (step 4 disappears).
- ffmpeg's scaler is multithreaded SIMD — far faster than PIL LANCZOS, and
  with `-hwaccel cuda -vf scale_cuda=...` the decode+scale can stay entirely
  on the GPU's dedicated NVDEC/copy engines (they do not compete with the
  CUDA cores running the model).
- Bonus: chunk PNG disk usage drops ~8x, so `chunk_size` can be raised.

### 2.2 Never upscale the matte in Python

Save mattes at model resolution and let the segment encoder do one scale:

```python
# _flush_matte_segment — add before the output args
"-vf", f"scale={orig_w}:{orig_h}:flags=lanczos",
```

Steps 6–7 collapse to a small 2880x1440 grayscale PNG write (~10x less deflate
work). The matte is a soft, low-frequency signal — a LANCZOS upscale in ffmpeg
is visually identical to one in PIL. Better still, for `DEOVR_ALPHA` output,
keep segments at matte resolution and scale once inside `pack_alpha`'s filter
graph (it downscales to 40% regardless, so upscaling first is pure loss).

### 2.3 Prefetch: extract chunk N+1 while matting chunk N

ffmpeg runs as a subprocess (no GIL contention). Use two alternating frame
directories and a single background thread:

```python
with ThreadPoolExecutor(max_workers=1) as pool:
    pending = pool.submit(extract, chunk_0, dir_a)
    for chunk in chunks:
        frames = pending.result()
        if next_chunk:
            pending = pool.submit(extract, next_chunk, other_dir)
        matte_all(frames)   # GPU works while ffmpeg fills the other dir
```

Extraction time is fully hidden behind matting → end-to-end FPS becomes
`min(extract_fps, matte_fps)` instead of the harmonic combination of both.

### 2.4 The endgame: raw-video pipes, zero PNG, zero temp frames

The PNG intermediaries exist only as a transport between ffmpeg and Python.
Replace them with `rawvideo` pipes:

```
ffmpeg (NVDEC decode 8K HEVC, scale to 2880x1440, -f rawvideo -pix_fmt rgb24 pipe:1)
   │  fixed-size reads: exactly w*h*3 bytes per frame
   ▼
Python: np.frombuffer → split eyes → MatAnyone2 ×2 → merged matte (uint8)
   │  fixed-size writes: w*h bytes per frame
   ▼
ffmpeg (-f rawvideo -pix_fmt gray -s 2880x1440 -i pipe:0 → NVENC segment encode)
```

This removes **all** PNG codec work, all frame disk I/O, and the temp-space
problem for source frames entirely (checkpoint/resume still works at segment
granularity — on resume, seek to `resume_frames / fps` and restart the reader).

Note on the Windows pipe gotchas in CLAUDE.md: those deadlocks came from
ffmpeg's *stderr* progress stream (`\r`-terminated, unbounded) and unread
pipes filling their 64 KB buffer. A `rawvideo` **stdout** stream with
`stderr=DEVNULL` and exact `read(w*h*3)` calls is the safe pattern — the
"blocks until exactly n bytes" behavior of Windows `BufferedReader.read` is
precisely what a fixed-size frame reader wants. Keep a `readinto()` on a
preallocated buffer to avoid per-frame allocations, and drain nothing else.

Expected state after Tier 1: pipeline is **GPU-bound on MatAnyone2 inference**,
i.e. roughly 5–10 FPS end-to-end depending on eye resolution.

---

## 3. Tier 2 — Make the GPU step itself faster (est. 1.5–3x on inference)

### 3.1 Batch both SBS eyes through RVM in one forward pass

`RVMProcessor.process_frame` takes `[1, C, H, W]`, and RVM's recurrent states
carry a batch dimension — both eyes can go through as a `[2, C, H, W]` batch
with per-eye recurrent state, halving kernel-launch overhead and improving
occupancy (~1.7–2x for the RVM SBS path). Today the runner instead creates two
full `RVMProcessor` instances (two model copies in VRAM) and calls them
sequentially.

MatAnyone2's `InferenceCore` is single-object stateful, so true batching needs
upstream changes — but the two eye processors can run on **separate CUDA
streams** from two Python threads (inference releases the GIL inside CUDA
ops), overlapping the left eye's memory-bank readout with the right eye's
convolutions. Worth ~1.2–1.4x.

### 3.2 Pinned memory + non-blocking transfers

`_to_tensor()` does a pageable H2D copy, and `_extract_matte()` does a
synchronous `.cpu()` that stalls the pipeline every frame. Use a preallocated
pinned staging buffer and `to(device, non_blocking=True)`, and copy the matte
out with `non_blocking=True` into pinned memory, synchronizing one frame late.
Small per-call win, but it is 4 sync points per frame in the SBS path.

### 3.3 Turn on `ma2_compile_model` by default on CUDA

The `torch.compile(mode="reduce-overhead")` path already exists behind a
default-off flag. MatAnyone2 at fixed resolution is exactly the CUDA-graph
sweet spot (identical shapes every frame). Triton now ships Windows wheels
that work with cu128/Blackwell — try enabling it by default with the existing
graceful fallback. Claimed 15–30% per frame.

### 3.4 Stop calling `torch.cuda.empty_cache()` between chunks

The chunk loop empties the CUDA cache every 500 frames. With
`expandable_segments:True` already set, fragmentation is handled; emptying the
cache just forces slow re-allocation (and cuDNN benchmark re-warm) at the
start of every chunk. Keep it only in `cleanup()` / on OOM recovery.

Also note: `configure_cuda_performance()` sets `PYTORCH_CUDA_ALLOC_CONF` via
`os.environ.setdefault`, but the UI calls `get_device_info()` at startup —
if that initializes the CUDA context first, the env var is read too late.
Set it in `bootstrap.py` before any torch import instead.

---

## 4. Tier 3 — Process fewer / smaller frames (optional, up to 2x more)

These trade (imperceptible) quality for speed and can be UI toggles:

1. **Matte at half frame-rate, interpolate alpha.** 60fps source → matte every
   2nd frame, synthesize skipped mattes by blending neighbors (alpha is soft
   and temporally smooth; MatAnyone2's own memory model assumes exactly this
   coherence). Straight 2x. Implement as `-vf "select=not(mod(n\,2))"` +
   matte-side `tblend`/`minterpolate` or a numpy lerp before segment encode.
2. **Lower the per-eye matting resolution.** Inference cost scales ~linearly
   with pixels. The 16 GB tier picks 1920x1080 as a *VRAM* cap, but
   1280x720/eye typically looks identical for passthrough silhouettes at
   VR viewing distance — ~2x faster inference. Expose as a "quality/speed"
   selector rather than a raw pixel number.
3. **RVM for content where it holds up.** RVM mobilenetv3 at
   `downsample_ratio 0.25` runs 100+ FPS at HD — an order of magnitude faster
   than MatAnyone2. With Tier 1 data-path fixes plus eye-batching (3.1), the
   RVM path could approach real-time for preview passes, keeping MatAnyone2
   for final renders.

---

## 5. Suggested implementation order

| Step | Change | Effort | Expected end-to-end gain | Status |
|------|--------|--------|--------------------------|--------|
| 1 | 2.1 extract at target resolution (`-vf scale`) | ~10 lines | 2–3x | ✅ done (2026-07-07) |
| 2 | 2.2 matte saved at model res, upscale in segment encode | ~10 lines | 1.3–1.6x | ✅ done (2026-07-07) |
| 3 | 2.3 chunk prefetch thread | ~40 lines | 1.2–1.5x | ✅ done (2026-07-07) |
| 4 | 3.4 drop per-chunk `empty_cache`; alloc-conf in bootstrap | ~5 lines | 1.1–1.3x | ✅ done (2026-07-07) |
| 5 | 2.4 rawvideo pipe pipeline (replaces 1–3) | ~200 lines | GPU-bound | ✅ done (2026-07-07) — `pipeline/framestream.py`; PNG path kept for SAM2Matting + `VRAUTOMATTE_NO_STREAM=1` escape hatch |
| 6 | 3.1 batched/streamed eyes, 3.2 pinned transfers | moderate | 1.3–2x | open |
| 7 | Tier 3 toggles (half-rate matting, res selector) | small | up to 2x | open |
| 8 | SAM2Matting variant (model-level speedup, see MODEL_RESEARCH_2026-07.md) | done | ~5–10x vs MA2 if claims hold | ✅ integrated (2026-07-07), needs on-GPU A/B |

Note: 3.3 (torch.compile default-on) was deliberately NOT flipped — the
existing try/except only catches wrap-time errors, but Triton failures on
Windows typically surface at the first forward pass, which would crash the
pipeline mid-run. It stays opt-in until guarded by a warmup-frame test.

Steps 1–4 are low-risk, independent, and should take the pipeline from ~1 FPS
to roughly **4–6 FPS** on the RTX 5080. Step 5 makes the pipeline GPU-bound
(~**6–12 FPS**), and steps 6–7 push toward **10–20+ FPS** — turning a
60-hour job into roughly 3–6 hours.
