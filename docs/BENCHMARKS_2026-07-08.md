# Matting Model Benchmarks — 2026-07-08

Measured on real production content, after the 2026-07-08 round of
fixes and optimizations (see git history of this date for details).

## Test system

- Windows 11 Pro laptop, NVIDIA RTX 5080 Laptop GPU (16 GB VRAM,
  Blackwell sm_120), 31 GB RAM
- PyTorch 2.10 cu128, ONNX Runtime DirectML 1.24, CUDA driver 13.2
- Test video: a production SBS VR clip — 2880×1440 SBS HEVC, 59.94 fps,
  147,317 frames

## Methodology

- **1000 consecutive SBS frame pairs starting at frame 75,000**
  (mid-video: high motion, harder poses than the intro).
- Frames extracted per-eye to 1440×1440 JPEG (q2), matching the
  pipeline's own extraction.
- Frames fed by decode-ahead threads (bounded queues, one thread per
  eye), mirroring the real pipeline's overlapped decode. Feeder
  ceiling ≈ 125 pairs/s — the torch numbers are partly feeder-bound,
  so treat them as a floor.
- Every model in its production configuration: POV mode on,
  first-frame seeding excluded from timing, 3-pair warmup.
- **Effective fps** = source frames per second at the app's
  half-rate matting setting (every 2nd frame matted, alpha lerped),
  which all models honor.

## Results — 1000 pairs @ frame 75,000

| Rank | Model | Config | Effective fps | Full video (147k frames) |
|---|---|---|---|---|
| 1 | resnet50 torch/CUDA | fp16, eye-batched, ROI | **113.0** | **~25 min** |
| 2 | mobilenetv3 torch/CUDA | fp16, eye-batched, ROI | 110.4 | ~25 min |
| 3 | mobilenetv3 ONNX/DirectML | eye-batched | 22.3 | ~1.8 h |
| 4 | resnet50 ONNX/DirectML | eye-batched | 21.0 | ~1.9 h |
| 5 | SAM2Matting tiny | bf16 autocast, stride 2, 250-frame chunks | 11.2 | ~3.7 h |
| 6 | MatAnyone 2 | fp16 autocast, ROI, threaded eyes, no long-term mem | 9.3 | ~4.4 h |

Secondary run (60 pairs, video intro, preloaded frames — no feeder
cap): torch RVM reached 176–177 effective fps, confirming the
1000-pair torch numbers are feeder-limited; ONNX/MA2/SAM2Matting
matched the main run within noise.

## Reading the numbers

- **resnet50 torch is the throughput choice.** The quality RVM
  variant costs nothing over mobilenetv3 on this GPU — both are
  bottlenecked by frame feeding, not the network. There is no reason
  to pick mobilenetv3 on NVIDIA hardware.
- **ONNX/DirectML runs a consistent ~5x behind torch/CUDA** on an
  NVIDIA card. It exists for hardware portability (AMD/Intel), not
  speed.
- **SAM2Matting beats MatAnyone 2 on both speed and (subjectively)
  single-subject matte quality** on this content. MA2's per-frame
  cost is constant; SAM2Matting amortizes per-chunk overhead over
  sustained runs.
- Real-world pipeline throughput adds ffmpeg decode + NVENC encode
  around the matting; for the torch models that overhead dominates
  (expect ~30–45 min per full video), for the slower models it is
  negligible.

## Practical recommendation

- **Throughput / all-people matting:** resnet50 torch (~30–45 min
  per video).
- **Best single-subject instance matte:** SAM2Matting (~4 h per
  video).
- ONNX variants only when the app must run on non-NVIDIA GPUs.
- MatAnyone 2: only if a quality A/B on specific content shows it
  beating SAM2Matting where it matters.

## Same-day context

These speeds exist because of the 2026-07-08 fixes; the day started
with SAM2Matting unable to run at all and MatAnyone 2 crashing on
fp16. Highlights: bf16/fp16 autocast (2x SAM2Matting, MA2 unbroken),
matte_stride honored in the chunk path (2x), MA2 long-term memory
off by default (16x — Windows allocator stalls), grid-prompted
SAM2 subject seeding, scene-detector histogram downsampling, POV
exclusion bbox-limited, ONNX + POV eye batching, NVENC probe fixed
for RTX 50xx.
