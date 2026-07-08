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
  fork. The fork is a superset of stock SAM2, so the existing SAM2
  first-frame mask generation keeps working against it.
- Because of that package shadowing, stock ``sam2`` must not be
  imported before the fork's path is installed. ``prepare_environment``
  enforces this and asks for an app restart if it's too late.
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

import os
import sys
import urllib.request
import zipfile
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

    Must run BEFORE anything imports ``sam2`` in this process.
    Returns the repo path.
    """
    repo = ensure_repo()
    repo_str = str(repo)

    if "sam2" in sys.modules:
        mod_file = getattr(
            sys.modules["sam2"], "__file__", ""
        ) or ""
        if not mod_file.startswith(repo_str):
            raise RuntimeError(
                "The stock 'sam2' package was already imported "
                "in this session — SAM2Matting needs its own "
                "sam2 fork loaded first. Restart the app and "
                "select SAM2Matting before running any "
                "MatAnyone2/SAM2 job."
            )
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        logger.debug(f"sys.path[0] = {repo_str}")
    return repo


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
        path = hf_hub_download(
            repo_id=_HF_REPO, filename=filename,
            local_dir=str(ckpt_dir),
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

        if isinstance(first_frame_mask, np.ndarray):
            self._next_masks = [first_frame_mask]
        else:
            self._next_masks = list(first_frame_mask)
        if len(self._next_masks) > 1:
            logger.info(
                f"Tracking {len(self._next_masks)} subjects"
            )

    # ── chunk-level API ─────────────────────────────────────

    def process_chunk(
        self, frames_dir: Path
    ) -> Iterator[np.ndarray]:
        """Matte one chunk directory of frames.

        Yields one uint8 (H, W) matte per frame (max-merged
        over all tracked subjects). Each subject's final matte
        of the chunk is kept (binarized) as its mask prompt for
        the next chunk, carrying every subject across chunk
        boundaries.
        """
        torch = self._torch
        predictor = self._predictor

        state = self._init_state(str(frames_dir))
        try:
            for obj_idx, m in enumerate(self._next_masks):
                mask = torch.from_numpy(m > 127)
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=0,
                    obj_id=obj_idx + 1,
                    mask=mask.to(self._device),
                )

            last_planes = None
            with torch.no_grad():
                for out in predictor.propagate_in_video(
                    state
                ):
                    # Upstream yields:
                    # (frame_idx, obj_ids, mask_logits,
                    #  alpha, ...)
                    planes = alpha_to_planes(out[3])
                    last_planes = planes
                    yield planes.max(axis=0)

            self._update_handoff(last_planes)
        finally:
            self._release_state(state)

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
        """init_state with video frames offloaded to CPU RAM."""
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
