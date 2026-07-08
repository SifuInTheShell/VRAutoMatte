"""SAM2Matting processor — unified tracking + matting.

SAM2Matting (FudanCVL, 2026) puts matting heads on a frozen SAM2.1
VOS tracker: one model tracks the subject AND produces the alpha
matte, replacing the two-stage SAM2-mask -> MatAnyone2 pipeline.
Paper-reported speed: ~40 FPS @1080p in <5 GB VRAM (Tiny variant).

Deployment notes
----------------
- The upstream repo (github.com/FudanCVL/SAM2Matting) is NOT a pip
  package; it ships a *fork* of the ``sam2`` package with the
  matting heads added. We download the repo into the model cache
  and prepend it to ``sys.path`` so ``import sam2`` resolves to the
  fork. The fork is NOT a superset of stock SAM2 — it ships neither
  ``automatic_mask_generator`` nor the standard model configs, so
  first-frame mask generation must run against the STOCK install
  BEFORE the fork is activated (see the sam2matting branch in
  ``matte.create_processor``).
- Because of that package shadowing, ``prepare_environment`` purges
  any stock ``sam2`` modules from ``sys.modules`` so the fork
  imports fresh afterwards.
- Checkpoints (e.g. SAM2Matting-SAM2.1Tiny.pt) go into the repo's
  ``checkpoints/`` directory. We attempt a Hugging Face download;
  if that fails the error tells the user where to put the file.
- License: CC BY-NC-SA 4.0 (non-commercial research). Like
  MatAnyone2, this stays an optional user-installed component.

Chunk-level API
---------------
SAM2's video predictor wants a directory of frames, not a stream,
so this processor exposes ``process_chunk(frames_dir)`` instead of
``process_frame``. The pipeline extracts JPEG chunks and calls it
once per chunk; the final matte of each chunk (binarized) seeds
the next chunk's mask prompt, carrying the subject across chunk
boundaries.
"""

import gc
import os
import shutil
import sys
import urllib.request
import warnings
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import numpy as np
from loguru import logger

_REPO_ZIP_URL = (
    "https://codeload.github.com/FudanCVL/SAM2Matting/"
    "zip/refs/heads/main"
)
_REPO_ENV = "VRAUTOMATTE_SAM2MATTING_PATH"
_HF_REPO = "FudanCVL/SAM2Matting"

# model_size -> (checkpoint filename, config name candidates)
_MODEL_FILES = {
    "tiny": (
        "SAM2Matting-SAM2.1Tiny.pt",
        ["sam2matting-sam2.1tiny"],
    ),
    "baseplus": (
        "SAM2Matting-SAM2.1Base+.pt",
        ["sam2matting-sam2.1base+", "sam2matting-sam2.1baseplus"],
    ),
}


def _cache_dir() -> Path:
    d = Path.home() / ".cache" / "vrautomatte" / "sam2matting"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_repo() -> Path:
    """Locate or download the SAM2Matting repository.

    Resolution order:
    1. ``VRAUTOMATTE_SAM2MATTING_PATH`` env var (user clone).
    2. Cached copy under ~/.cache/vrautomatte/sam2matting.
    3. Fresh download of the repo zip from GitHub.
    """
    env_path = os.environ.get(_REPO_ENV)
    if env_path:
        p = Path(env_path)
        if (p / "sam2").is_dir():
            return p
        raise RuntimeError(
            f"{_REPO_ENV}={env_path} does not look like a "
            "SAM2Matting checkout (no sam2/ directory)."
        )

    cached = _cache_dir() / "SAM2Matting-main"
    if (cached / "sam2").is_dir():
        return cached

    logger.info(
        "Downloading SAM2Matting repository "
        f"({_REPO_ZIP_URL})..."
    )
    zip_path = _cache_dir() / "repo.zip"
    urllib.request.urlretrieve(_REPO_ZIP_URL, str(zip_path))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(_cache_dir())
    zip_path.unlink(missing_ok=True)
    if not (cached / "sam2").is_dir():
        raise RuntimeError(
            "SAM2Matting download did not produce the expected "
            f"layout under {cached}. Clone the repo manually and "
            f"set {_REPO_ENV} to its path."
        )
    logger.info(f"SAM2Matting repo ready at {cached}")
    return cached


def prepare_environment() -> Path:
    """Ensure the SAM2Matting fork of ``sam2`` will be imported.

    If the stock sam2 package is loaded (e.g. from first-frame
    mask generation), its modules are purged so the next
    ``import sam2`` resolves to the fork. Returns the repo path.
    """
    repo = ensure_repo()
    repo_str = str(repo)

    if "sam2" in sys.modules:
        mod_file = getattr(
            sys.modules["sam2"], "__file__", ""
        ) or ""
        if not mod_file.startswith(repo_str):
            stale = [
                name for name in sys.modules
                if name == "sam2"
                or name.startswith("sam2.")
            ]
            for name in stale:
                del sys.modules[name]
            gc.collect()
            logger.debug(
                f"Purged {len(stale)} stock sam2 modules "
                "so the SAM2Matting fork loads fresh"
            )
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        logger.debug(f"sys.path[0] = {repo_str}")
    return repo


def _fork_path_hint() -> str | None:
    """Where the fork lives if present — never downloads."""
    env_path = os.environ.get(_REPO_ENV)
    if env_path and (Path(env_path) / "sam2").is_dir():
        return env_path
    cached = (
        Path.home() / ".cache" / "vrautomatte"
        / "sam2matting" / "SAM2Matting-main"
    )
    if (cached / "sam2").is_dir():
        return str(cached)
    return None


@contextmanager
def stock_sam2():
    """Make the stock ``sam2`` package importable in the block.

    SAM2 mask generation (first-frame subjects, POV body)
    needs stock sam2 — automatic mask generator + standard
    model configs, neither of which the SAM2Matting fork
    ships. Once the fork is active (any earlier SAM2Matting
    processor in the session) it must be moved off sys.path
    and out of sys.modules for the duration, then restored.
    No-op when no fork exists on this machine.
    """
    repo_str = _fork_path_hint()
    if repo_str is None or repo_str not in sys.path:
        # Fork absent or never activated this session —
        # stock sam2 resolves naturally.
        yield
        return
    sys.path.remove(repo_str)
    loaded = [
        name for name in sys.modules
        if name == "sam2" or name.startswith("sam2.")
    ]
    for name in loaded:
        del sys.modules[name]
    if loaded:
        gc.collect()
        logger.debug(
            f"stock_sam2: sidelined {len(loaded)} "
            "sam2 modules"
        )
    try:
        yield
    finally:
        # Drop whatever the block imported (stock modules);
        # prepare_environment() reloads the fork afterwards.
        stale = [
            name for name in sys.modules
            if name == "sam2" or name.startswith("sam2.")
        ]
        for name in stale:
            del sys.modules[name]
        gc.collect()
        sys.path.insert(0, repo_str)


def _ensure_checkpoint(repo: Path, model_size: str) -> Path:
    """Locate the checkpoint, downloading from HF if needed."""
    filename, _ = _MODEL_FILES[model_size]
    ckpt_dir = repo / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    ckpt = ckpt_dir / filename
    if ckpt.exists():
        return ckpt

    try:
        from huggingface_hub import hf_hub_download
        logger.info(
            f"Downloading {filename} from Hugging Face "
            f"({_HF_REPO})..."
        )
        # The HF repo stores weights under checkpoints/;
        # local_dir=repo preserves that layout, so the file
        # lands exactly at the manual-placement path (ckpt).
        path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=f"checkpoints/{filename}",
            local_dir=str(repo),
        )
        return Path(path)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"SAM2Matting checkpoint not found and automatic "
            f"download failed ({exc}). Download {filename} "
            f"from the links in the SAM2Matting README "
            f"(github.com/FudanCVL/SAM2Matting) and place it "
            f"at: {ckpt}"
        ) from exc


def _find_config(repo: Path, model_size: str) -> str:
    """Find the hydra config name for the given model size.

    Mirrors the upstream inference script, which passes e.g.
    ``configs/sam2matting-sam2.1tiny.yaml``. Falls back to
    scanning the repo for a matching yaml.
    """
    _, candidates = _MODEL_FILES[model_size]
    for yaml in repo.rglob("sam2matting*.yaml"):
        stem = yaml.stem.lower()
        for cand in candidates:
            if stem == cand:
                # hydra config path relative to its search dir —
                # upstream uses "configs/<name>.yaml"
                return f"configs/{yaml.name}"
    # Last resort: upstream's documented default
    return f"configs/{candidates[0]}.yaml"


def alpha_to_planes(alpha) -> np.ndarray:
    """Normalize a predictor alpha output to uint8 [N, H, W].

    Accepts torch tensors or arrays shaped [N,1,H,W], [N,H,W],
    [1,H,W] or [H,W] in either 0..1 or 0..255 range. One plane
    per tracked object.
    """
    a = alpha
    if hasattr(a, "detach"):
        a = a.detach().float().cpu().numpy()
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > 3 and a.shape[1] == 1:
        a = a[:, 0]          # [N,1,H,W] -> [N,H,W]
    while a.ndim > 3:
        a = a[0]             # defensive: drop leading dims
    if a.ndim == 2:
        a = a[None]          # [H,W] -> [1,H,W]
    if a.size and a.max() <= 1.5:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


class SAM2MattingProcessor:
    """Chunk-level matting via the SAM2Matting video predictor.

    Supports 1..N tracked subjects: each first-frame mask is
    registered as its own SAM2 object; the per-object alphas
    are merged (max) into one output matte, and each object's
    final matte seeds its mask prompt for the next chunk.

    Args:
        first_frame_mask: Binary mask (H, W) uint8 of the
            subject — or a LIST of masks for multi-subject
            tracking — at matting resolution.
        device: torch device (auto if None).
        model_size: 'tiny' (recommended) or 'baseplus'.
        compile_model: Pass the upstream vos_optimized/compile
            flag (their --compiled option).
    """

    chunk_level = True

    def __init__(
        self,
        first_frame_mask,
        device=None,
        *,
        model_size: str = "tiny",
        compile_model: bool = False,
        matte_stride: int = 1,
    ):
        import torch  # local import — after bootstrap

        repo = prepare_environment()
        ckpt = _ensure_checkpoint(repo, model_size)
        cfg = _find_config(repo, model_size)

        if device is None:
            from vrautomatte.utils.gpu import get_device
            device = get_device()
        self._device = device
        self._torch = torch

        from sam2.build_sam import (
            build_sam2matting_video_predictor,
        )

        logger.info(
            f"Loading SAM2Matting ({model_size}) on "
            f"{device}..."
        )
        kwargs = {}
        if compile_model:
            kwargs["vos_optimized"] = True
        try:
            self._predictor = (
                build_sam2matting_video_predictor(
                    cfg, str(ckpt),
                    device=str(device), **kwargs,
                )
            )
        except TypeError:
            # Older/newer builder signature without kwargs
            self._predictor = (
                build_sam2matting_video_predictor(
                    cfg, str(ckpt), device=str(device),
                )
            )
        logger.info("SAM2Matting loaded")
        self._patch_alpha_head_devices()
        self._patch_frame_loader_dtype()
        # The fork warns about its missing optional _C
        # extension on every chunk — benign per SAM2's own
        # message, and it floods the log twice per chunk.
        warnings.filterwarnings(
            "ignore",
            message=r"cannot import name '_C'",
        )

        if isinstance(first_frame_mask, np.ndarray):
            self._next_masks = [first_frame_mask]
        else:
            self._next_masks = list(first_frame_mask)
        if len(self._next_masks) > 1:
            logger.info(
                f"Tracking {len(self._next_masks)} subjects"
            )

        self._stride = max(1, int(matte_stride))
        if self._stride > 1:
            logger.info(
                f"Half-rate matting: every {self._stride}. "
                "frame propagated, alpha interpolated"
            )

    def _autocast(self):
        """Mixed-precision context for propagation on CUDA.

        The fork runs fp32 without this — bf16 autocast
        measured ~2x propagation speed at <1.5% matte
        deviation on real content.
        """
        torch = self._torch
        dev_type = getattr(
            self._device, "type", str(self._device)
        )
        if dev_type == "cuda":
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            return torch.autocast("cuda", dtype=dtype)
        import contextlib
        return contextlib.nullcontext()

    def _patch_alpha_head_devices(self) -> None:
        """Work around a fork bug in the alpha-head path.

        _run_single_frame_inference feeds the raw frame to the
        alpha heads straight from inference_state['images'],
        which lives on the storage device — CPU when the video
        is offloaded — and crashes torch.cat inside
        _detect_unknown_region. Wrap _forward_alpha_heads and
        move its tensor inputs to the model device first.
        """
        torch = self._torch
        predictor = self._predictor
        orig = getattr(
            predictor, "_forward_alpha_heads", None
        )
        if orig is None:
            return  # upstream changed; nothing to patch
        device = self._device

        def to_dev(obj):
            if torch.is_tensor(obj):
                return obj.to(device, non_blocking=True)
            if isinstance(obj, tuple):
                return tuple(to_dev(o) for o in obj)
            if isinstance(obj, list):
                return [to_dev(o) for o in obj]
            return obj

        def wrapper(*args, **kwargs):
            args = [to_dev(a) for a in args]
            kwargs = {
                k: to_dev(v) for k, v in kwargs.items()
            }
            return orig(*args, **kwargs)

        predictor._forward_alpha_heads = wrapper

    def _patch_frame_loader_dtype(self) -> None:
        """Work around a fork bug in the frame loader.

        _load_img_as_tensor divides a uint8 numpy array by
        255.0, which promotes to float64. The synchronous
        loader masks this by copying into a preallocated
        float32 tensor, but AsyncVideoFrameLoader hands the
        double tensors straight to the model and conv2d
        rejects them. Cast to float32 at the source (also
        halves CPU RAM per loaded chunk).
        """
        try:
            from sam2.utils import misc as fork_misc
        except ImportError:
            return
        orig = getattr(
            fork_misc, "_load_img_as_tensor", None
        )
        if orig is None or getattr(
            orig, "_f32_patched", False
        ):
            return

        def load_f32(img_path, image_size):
            img, h, w = orig(img_path, image_size)
            return img.float(), h, w

        load_f32._f32_patched = True
        fork_misc._load_img_as_tensor = load_f32

    # ── chunk-level API ─────────────────────────────────────

    def _strided_dir(
        self, frames_dir: Path, files: list, stride: int
    ) -> Path:
        """Hardlink every Nth frame into a sibling directory."""
        sub = frames_dir.parent / (
            frames_dir.name + f"_s{stride}"
        )
        if sub.exists():
            shutil.rmtree(sub, ignore_errors=True)
        sub.mkdir(exist_ok=True)
        for f in files[::stride]:
            target = sub / f.name
            if target.exists():
                continue
            try:
                os.link(f, target)
            except OSError:
                try:
                    shutil.copy2(f, target)
                except shutil.SameFileError:
                    pass
        return sub

    def process_chunk(
        self, frames_dir: Path
    ) -> Iterator[np.ndarray]:
        """Matte one chunk directory of frames.

        Yields one uint8 (H, W) matte per frame (max-merged
        over all tracked subjects). Each subject's final matte
        of the chunk is kept (binarized) as its mask prompt for
        the next chunk, carrying every subject across chunk
        boundaries.

        With matte_stride > 1 only every Nth frame is
        propagated; in-between mattes are linearly
        interpolated (temporally smooth, so lerp is visually
        equivalent at stride 2 for 60fps content).
        """
        torch = self._torch
        predictor = self._predictor
        frames_dir = Path(frames_dir)
        stride = self._stride
        files = sorted(
            f for f in frames_dir.iterdir()
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        n_frames = len(files)
        run_dir = frames_dir
        sub = None
        if stride > 1 and n_frames > stride:
            sub = self._strided_dir(frames_dir, files, stride)
            run_dir = sub

        state = self._init_state(str(run_dir))
        try:
            last_planes = None
            prev_idx = -1
            prev_matte = None
            emitted = 0
            with torch.no_grad(), self._autocast():
                for obj_idx, m in enumerate(
                    self._next_masks
                ):
                    mask = torch.from_numpy(m > 127)
                    predictor.add_new_mask(
                        inference_state=state,
                        frame_idx=0,
                        obj_id=obj_idx + 1,
                        mask=mask.to(self._device),
                    )

                for k, out in enumerate(
                    predictor.propagate_in_video(state)
                ):
                    # Upstream yields:
                    # (frame_idx, obj_ids, mask_logits,
                    #  alpha, ...)
                    planes = alpha_to_planes(out[3])
                    last_planes = planes
                    merged = planes.max(axis=0)
                    orig = k * stride
                    if prev_matte is not None:
                        span = orig - prev_idx
                        for i in range(prev_idx + 1, orig):
                            w = (i - prev_idx) / span
                            interp = (
                                prev_matte.astype(np.float32)
                                * (1.0 - w)
                                + merged.astype(np.float32)
                                * w
                            )
                            yield interp.astype(np.uint8)
                            emitted += 1
                    yield merged
                    emitted += 1
                    prev_idx = orig
                    prev_matte = merged

            # Tail frames past the last propagated one
            # (odd chunk lengths at stride > 1).
            if prev_matte is not None:
                while emitted < n_frames:
                    yield prev_matte.copy()
                    emitted += 1

            self._update_handoff(last_planes)
        finally:
            self._release_state(state)
            if sub is not None:
                shutil.rmtree(sub, ignore_errors=True)

    def _update_handoff(self, planes) -> None:
        """Seed next chunk's mask prompts from final mattes.

        Per-object when the predictor returns one alpha plane
        per tracked subject; otherwise the merged matte carries
        on as a single object (subjects fuse — logged once).
        Objects whose matte went empty (subject left frame)
        keep their previous mask so they can be re-acquired.
        """
        if planes is None:
            return
        n_obj = len(self._next_masks)
        if planes.shape[0] == n_obj:
            for i in range(n_obj):
                if planes[i].max() > 127:
                    self._next_masks[i] = planes[i]
                else:
                    logger.debug(
                        f"Subject {i + 1} not visible at "
                        "chunk end — keeping previous mask"
                    )
        else:
            if n_obj > 1 and not getattr(
                self, "_merge_warned", False
            ):
                logger.warning(
                    "Predictor returned a combined alpha "
                    f"({planes.shape[0]} plane(s) for "
                    f"{n_obj} subjects) — continuing with "
                    "merged tracking"
                )
                self._merge_warned = True
            merged = planes.max(axis=0)
            if merged.max() > 127:
                self._next_masks = [merged]

    def _init_state(self, frames_dir: str):
        """init_state with frames offloaded to CPU RAM.

        async_loading_frames decodes JPEGs in a background
        thread so GPU matting starts on frame 0 immediately
        instead of blocking on the full chunk load.
        """
        try:
            return self._predictor.init_state(
                video_path=frames_dir,
                offload_video_to_cpu=True,
                async_loading_frames=True,
            )
        except TypeError:
            try:
                return self._predictor.init_state(
                    video_path=frames_dir,
                    offload_video_to_cpu=True,
                )
            except TypeError:
                return self._predictor.init_state(
                    video_path=frames_dir,
                )

    def _release_state(self, state) -> None:
        try:
            self._predictor.reset_state(state)
        except Exception:  # noqa: BLE001
            pass
        del state

    # ── MatteProcessor protocol ─────────────────────────────

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        raise RuntimeError(
            "SAM2MattingProcessor is chunk-level — use "
            "process_chunk()."
        )

    def reset(self) -> None:
        """Reset for a new video (mask must be re-seeded)."""

    def cleanup(self) -> None:
        """Release model and GPU memory."""
        if hasattr(self, "_predictor"):
            del self._predictor
        try:
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
        logger.debug("SAM2Matting processor cleaned up")
