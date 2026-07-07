# Video Matting Model Research — July 2026

**Question:** Is there a faster and/or higher-quality alternative to the current
RVM (fast path) / SAM2 + MatAnyone2 (quality path) stack for matting hour-long
8K SBS VR video locally on an RTX 5080 Laptop (16 GB, Blackwell)?

**Method:** Multi-agent research sweep (5 search angles → 20 primary-source
fetches → claim extraction → adversarial verification). The verification phase
was stopped early by request; the 11 completed verification votes all
**confirmed** their claims (MatAnyone2 repo/weights/license/workflow). The
SAM2Matting headline claims were additionally hand-verified against the
official repo and paper. Remaining claims are single-source from primary
material (arXiv papers, official GitHub READMEs) — trustworthy for facts,
self-reported for benchmarks.

---

## TL;DR — Ranked recommendation

| Rank | Model | Role | Why |
|------|-------|------|-----|
| **1** | **SAM2Matting (Tiny)** — trial now | New default candidate | Claims to beat MatAnyone2 on quality **and** run ~40 FPS @1080p in <5 GB VRAM. Single model replaces the SAM2+MA2 two-stage pipeline. Biggest possible win if claims hold on real VR content. |
| **2** | **MatAnyone 2** — keep | Quality fallback | Confirmed quality ceiling among memory-based models. Already integrated. Nothing replaces it for guaranteed quality today. |
| **3** | **RVM** — keep | Draft/preview path | Still the fastest deployable matting model (100+ FPS at HD). 2021-era quality, but unbeaten throughput per watt of integration effort. |
| — | VideoMaMa, GVM (diffusion) | Not viable | Quality is real, throughput is not: 16-frame batches, ≤1024×576, "keep videos under 5 s". Wrong tool for 100k+ frame jobs. |
| — | BiVM, RVM+, SAMA | Not adoptable | No desktop-GPU numbers and/or no released code/weights, or image-only. |

**Decision for VRAutoMatte:** integrate SAM2Matting as a third
`MatteProcessor` variant and A/B it against MatAnyone2 on a real 8K clip. If
quality holds, it becomes the default (fast **and** high quality); MA2 stays as
the quality fallback; RVM stays as the draft mode. This also decides the next
engineering step: the rawvideo-pipe work (Tier 1 step 5) benefits all three
paths and should follow regardless of which model wins.

---

## 1. The main contenders in detail

### 1.1 SAM2Matting (Fudan, June 2026) — the challenger

A decoupled tracker-to-matting framework: a frozen SAM2.1/SAM3 VOS tracker
with a region-proposal bridge and dedicated matting heads. Accepts mask,
point, box (and text with SAM3) prompts.

- **Speed (self-reported):** Tiny variant ~40 FPS on a 200-frame 1080p video
  in <5 GB VRAM; Tiny and Base+ both >30 FPS. "Minimal FPS and VRAM overhead
  over the trackers", stable across resolutions. Benchmark GPU not stated —
  expect lower on a laptop 5080, and halved for SBS (two eyes).
- **Quality (self-reported):** zero-shot (trained on image matting data only),
  claims to outperform MatAnyone2, MatAnyone, MaGGIe, FTP-VM, and RVM on
  V-HIM60 and VideoMatte; lowest dtSSD (temporal coherence) of all compared
  methods — consistency inherited from the VOS tracker, claimed to hold on
  extended videos.
- **Deployment:** code + checkpoints released (GitHub `FudanCVL/SAM2Matting`,
  Hugging Face; three variants: SAM2.1-T, SAM2.1-B+, SAM3). Python 3.10,
  PyTorch. Built on the same SAM2 stack VRAutoMatte already runs on this
  hardware. Optional `--compiled` flag.
- **License:** CC BY-NC-SA 4.0 — non-commercial research only. Fine for
  personal use; same practical restriction as MatAnyone2 (NTU S-Lab).
- **Risks:** brand-new repo, no independent replication of the benchmarks yet;
  benchmark hardware unstated; streaming (frame-by-frame) API vs whole-clip
  processing needs hands-on verification; "matting heads on a frozen tracker"
  may inherit SAM2-family long-video drift (see §3).

**Verdict:** the only candidate that plausibly improves *both* axes at once.
Must be validated on real VR passthrough content before committing.

### 1.2 MatAnyone 2 (NTU S-Lab + SenseTime, CVPR 2026 Highlight) — the incumbent

All core claims **verified** (11/11 adversarial votes confirmed).

- **Quality:** state of the art among published memory-based models. On
  YouTubeMatte @1080p: MAD 1.61 / MSE 0.50 / Grad 7.13 / dtSSD 1.53 vs RVM's
  MAD 4.27 (3.37 retrained) — ~2.6× lower semantic error than RVM. 27.1%
  Grad / 22.4% Conn improvement over MatAnyone v1. Trained on VMReal (28K
  clips / 2.4M frames) curated by a learned Matting Quality Evaluator.
- **Long video:** reference-frame training strategy explicitly targets
  appearance drift beyond the local memory window, at no extra inference
  memory — genuinely relevant for hour-long footage.
- **Speed:** no FPS or VRAM figures published anywhere (repo, paper, project
  page). The MQE innovation is training-time only — inference is not faster
  than MatAnyone v1. This is the model behind the current ~1 FPS experience.
- **Deployment:** PyTorch-only (no official ONNX/TensorRT), weights
  auto-download, NTU S-Lab License 1.0 (non-commercial).

**Verdict:** keep as the quality path. It will not get faster; speed must come
from the pipeline (done in Tier 1) and/or a model switch.

### 1.3 RVM (2021) — the fast baseline

- **Speed (published, verified against official repo):** mobilenetv3: 104 FPS
  @HD / 76 FPS @4K on a GTX 1080 Ti FP32; 172 FPS @HD / 154 FPS @4K on an RTX
  3090 FP16. A 5080 Laptop exceeds the 1080 Ti comfortably.
- **Quality bound is architectural:** the coarse semantic pass runs at only
  256–512 px (downsample_ratio), then refines — it structurally cannot see the
  detail MatAnyone-class models see. 2021-era, unmaintained since.
- **Deployment:** the most deployment-ready model in the field — official
  TorchScript/ONNX/CoreML/TF exports (TensorRT via ONNX). GPL-3.0.

**Verdict:** keep for previews/drafts. Optimization worth doing if kept hot:
batch both SBS eyes in one forward pass (recurrent state supports batch=2).

---

## 2. Ruled out (and why)

| Model | Type | Fatal flaw for this use case |
|-------|------|------------------------------|
| **VideoMaMa** (CVPR 2026, Adobe/KU) | Diffusion, mask-guided | Quality is real (MAD 1.737 YouTubeMatte vs MaGGIe 1.9499) but runs 16-frame chunks @1024×576, docs recommend ≤5 s videos; batch-seam artifacts on long video (needs crossfade workarounds). CC BY-NC + Stability AI license. |
| **GVM** (SIGGRAPH 2025) | Diffusion (SVD-based) | Demo caps resolution at 960 px; diffusion-scale compute; academic-only BSD. |
| **BiVM** (TPAMI 2025) | Binarized RVM-class | Benchmarked on phone CPUs only; no NVIDIA numbers; no confirmed weights release. |
| **RVM+** (Sensors 2025) | RVM mod | No code, weights, or license discoverable anywhere. Also: claims ConvGRU as novel — RVM already uses ConvGRU. |
| **SAMA** (AAAI 2026) | SAM + matting head | Image-only; video explicitly deferred to future work. |
| **BiRefNet HR-matting** (MIT!) | Per-frame image matting | Only MIT-licensed quality option, official TensorRT support — but no temporal mechanism at all and ~17 FPS @1024² on a 4090. Wrong shape for video; noted for possible future edge-refinement duty. |

Notable near-miss: the VideoMaMa authors also fine-tuned SAM2 on their MA-V
dataset to produce **SAM2-Matte** — the same "make SAM2 output mattes" idea as
SAM2Matting. Two independent groups converging on this design in 2026 is a
strong signal that tracker-native matting is the current direction, and that
SAM2Matting isn't a one-off.

---

## 3. Long-video reality check (contrarian findings)

- MatAnyone v1's issue tracker documents OOM on 4K **even on an 80 GB GPU**
  (#50), GPU memory leaks across repeated inference (#45), and tens of GB of
  system RAM for minutes-long 1080p videos (#56) — mostly unanswered by
  maintainers. VRAutoMatte's chunked pipeline + downscaling + `use_long_term`
  is what makes MA2 usable at all; this architecture must be preserved
  regardless of model choice.
- All SAM2-family memory models drift on long videos (SAM2Long: error
  accumulation, distractor lock-on; HiM2SAM: drift and mask fragmentation).
  SAM2Matting inherits this family trait — periodic re-prompting at chunk
  boundaries (which the current architecture supports) is the standard
  mitigation. SAM2Long itself (training-free, +3–4 J&F on long video) is
  CC BY-NC and publishes no speed overhead numbers.
- **Sammie-Roto 2** (GPL-3.0 desktop app, actively maintained) ships this
  exact stack (SAM2/EfficientTAM → MatAnyone/MA2/VideoMaMa) on consumer GPUs.
  Two directly transferable tricks from its changelog:
  1. **ROI-restricted matting** — compute the matte only inside the region
     the segmentation mask defines (large claimed speed/VRAM wins). For VR
     passthrough where the subject fills a fraction of the frame, this could
     be a further ~2× on top of everything else, model-agnostic.
  2. Half-precision segmentation for the tracker stage.
- **EfficientTAM** (ICCV 2025, open weights): ~2× faster than SAM2 Hiera-B+ at
  comparable quality — a drop-in if the SAM2 mask stage is kept.

---

## 4. Licensing summary

| Model | License | Personal use | Ship in a commercial app |
|-------|---------|--------------|--------------------------|
| SAM2Matting | CC BY-NC-SA 4.0 | ✅ | ❌ (contact authors) |
| MatAnyone 2 | NTU S-Lab 1.0 | ✅ | ❌ (contact authors) |
| VideoMaMa | CC BY-NC 4.0 + Stability AI | ✅ | ❌ |
| RVM | GPL-3.0 | ✅ | ⚠️ copyleft obligations |
| BiRefNet | MIT | ✅ | ✅ |
| SAM2 / EfficientTAM | Apache 2.0 | ✅ | ✅ |

VRAutoMatte is MIT — all non-commercial models must stay optional extras the
user installs (as `matanyone2` already is), never bundled.

---

## 5. What this decides

1. **Next step: SAM2Matting trial.** Integrate as `model_variant="sam2matting"`
   (optional extra), A/B against MA2 on a real 8K SBS clip for quality and
   measured FPS. Effort: small — same mask-guided workflow, same SAM2 substrate.
2. **If it wins:** it becomes the default. The follow-up engineering
   (rawvideo pipes, ROI-restricted matting, per-eye CUDA streams) then applies
   to a model that's already ~5–10× faster than MA2 — compounding, not
   redundant.
3. **If it loses on quality:** stay on MA2 and proceed with the Tier 1 step 5
   rawvideo pipe + Tier 2 GPU work as planned. The pipeline work is
   model-agnostic and never wasted.
4. **Either way:** adopt ROI-restricted matting (Sammie-Roto-2-style) as a
   pipeline feature — it multiplies with every model.

---

## Sources

- SAM2Matting — https://github.com/FudanCVL/SAM2Matting · arXiv:2606.27339
- MatAnyone 2 — https://github.com/pq-yang/MatAnyone2 · arXiv:2512.11782 · https://pq-yang.github.io/projects/MatAnyone2/
- MatAnyone (v1) — arXiv:2501.14677 · issues: https://github.com/pq-yang/MatAnyone/issues (#45, #50, #55, #56)
- RVM — https://github.com/PeterL1n/RobustVideoMatting (WACV 2022)
- VideoMaMa — arXiv:2601.14255 · HF SammyLim/VideoMaMa
- GVM — arXiv:2508.07905 (SIGGRAPH 2025) · GitHub aim-uofa/GVM
- BiVM — arXiv:2507.04456 (TPAMI 2025)
- RVM+ — MDPI Sensors 25(5):1278
- SAMA — arXiv:2601.12147 (AAAI 2026)
- EfficientTAM — arXiv:2411.18933 (ICCV 2025)
- SAM2Long — arXiv:2410.16268 (ICCV 2025)
- BiRefNet — https://github.com/ZhengPeng7/BiRefNet
- Sammie-Roto 2 — https://github.com/Zarxrax/Sammie-Roto-2
- LearnOpenCV MatAnyone hands-on — https://learnopencv.com/matanyone-for-better-video-matting/
