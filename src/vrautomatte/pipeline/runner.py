"""Pipeline orchestrator — chains all steps from input to output.

Steps:
1-2. Extract frames and generate mattes (chunked)
3. Reassemble matte frames into a video
4. (Optional) Convert equirectangular -> fisheye
5. (Optional) Pack alpha channel for DeoVR format
"""

import math
import os
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from PIL import Image

from vrautomatte.pipeline.checkpoint import (
    PipelineCheckpoint,
    cleanup_stale_dirs,
    deterministic_temp_name,
    hash_config,
    hash_file_head,
)
from vrautomatte.pipeline.matte import AlphaSmoother, create_processor
from vrautomatte.pipeline.scaler import FrameScaler
from vrautomatte.utils.ffmpeg import (
    apply_fisheye_mask,
    check_ffmpeg,
    convert_to_fisheye,
    get_video_info,
    matte_to_red_channel,
    pack_alpha,
)
from vrautomatte.utils.gpu import auto_configure_gpu
from vrautomatte.utils.sbs import (
    detect_sbs,
    merge_mattes,
    split_frame,
)

# Minimum free space to keep on the drive (1 GB).
_MIN_FREE_BYTES = 1_073_741_824
# How often to check disk during matting (every N frames).
_DISK_CHECK_INTERVAL = 50


class OutputFormat(str, Enum):
    """Output format options."""
    MATTE_ONLY = "matte_only"       # Just the alpha matte video
    DEOVR_ALPHA = "deovr_alpha"     # Full DeoVR alpha-packed file


class ProjectionType(str, Enum):
    """Input video projection type."""
    EQUIRECTANGULAR = "equirectangular"
    FISHEYE = "fisheye"


@dataclass
class PipelineConfig:
    """Configuration for the matting pipeline."""
    input_path: str = ""
    output_path: str = ""

    # Matting settings
    model_variant: str = "mobilenetv3"
    downsample_ratio: float = 0.125

    # Output settings
    output_format: OutputFormat = OutputFormat.MATTE_ONLY
    codec: str = "libx265"
    crf: int = 18

    # VR-specific
    projection: ProjectionType = ProjectionType.EQUIRECTANGULAR
    fisheye_fov: int = 180
    fisheye_mask_path: str = ""

    # SBS processing
    is_sbs: bool = False

    # POV mode
    pov_mode: bool = False

    # Frame range (1-based, inclusive). 0 = unset (use all).
    start_frame: int = 0
    end_frame: int = 0

    # Custom temp directory (empty = system default).
    temp_dir: str = ""

    # ── MatAnyone 2 performance settings ──────────────────────
    use_fp16: bool = True
    ma2_internal_size: int = 480
    ma2_mem_frames: int = 3
    ma2_use_long_term: bool = True
    ma2_compile_model: bool = False

    # ── Temporal smoothing ──────────────────────────────────────
    # EMA weight for alpha smoothing (1.0 = off).
    temporal_smoothing: float = 1.0

    # ── Performance options ─────────────────────────────────────
    # Matte only a padded window around the tracked subject
    # (pipeline/roi.py). Big win when the person fills a
    # fraction of the frame, as in VR passthrough.
    roi_matting: bool = True
    # Matte every Nth frame and interpolate alpha in between
    # (stream path only). 1 = every frame, 2 = half-rate (~2x).
    matte_stride: int = 1
    # Run the two SBS eye models concurrently (only applies when
    # eyes can't share one batched model, i.e. MatAnyone2).
    # Roughly doubles peak VRAM — auto-enabled on >=24 GB GPUs.
    sbs_parallel_eyes: bool = False
    # Number of people to track (SAM2Matting only; each subject
    # is a separate SAM2 object). RVM mattes everyone natively;
    # MatAnyone2 tracks one subject (union mask for several).
    max_subjects: int = 1

    # ── Disk management ───────────────────────────────────────
    chunk_size: int = 500

    # ── GPU auto-config ───────────────────────────────────────
    # Max frame pixels for matting. 0 = no limit.
    # Auto-configured from GPU VRAM if not set manually.
    max_matting_pixels: int = 0

    # ── Resume ────────────────────────────────────────────────
    # Save checkpoint after each segment for resume on restart.
    auto_resume: bool = True


@dataclass
class PipelineProgress:
    """Progress information emitted during processing."""
    stage: str = ""
    stage_num: int = 0
    total_stages: int = 0
    frame_num: int = 0
    total_frames: int = 0
    source_frame: np.ndarray | None = None
    matte_frame: np.ndarray | None = None
    elapsed_sec: float = 0.0
    eta_sec: float = 0.0
    fps: float = 0.0
    estimated_disk_gb: float = 0.0


class Pipeline:
    """Orchestrates the full video matting pipeline.

    Args:
        config: Pipeline configuration.
        on_progress: Callback for progress updates.
    """

    def __init__(
        self, config: PipelineConfig,
        on_progress: Callable[[PipelineProgress], None] | None = None,
    ):
        self.config = config
        self.on_progress = on_progress
        self._cancelled = False
        self._start_time = 0.0
        self._matte_start_time = 0.0
        self._eye_pool = None

    def cancel(self) -> None:
        """Request cancellation of the running pipeline."""
        self._cancelled = True

    def _emit(self, progress: PipelineProgress) -> None:
        """Emit progress update if callback is set."""
        if self.on_progress:
            progress.elapsed_sec = (
                time.monotonic() - self._start_time
            )
            self.on_progress(progress)

    # ── Extraction ────────────────────────────────────────────

    def _extract_chunk(
        self, input_path: Path, frames_dir: Path,
        timestamp: float, num_frames: int,
        scale_to: tuple[int, int] | None = None,
        quiet: bool = False,
        fmt: str = "png",
        sbs_split: bool = False,
    ) -> list[Path]:
        """Extract frames using fast keyframe seek.

        Uses ``-ss`` before ``-i`` for keyframe-based seeking.
        ~1-2 frame imprecision at chunk boundaries is acceptable
        for VR content.  Polls the output directory so the UI
        stays responsive and shows extraction progress.

        When ``scale_to`` is set, ffmpeg downscales frames to the
        matting resolution during extraction — its multithreaded
        scaler replaces the per-frame PIL LANCZOS resize.
        ``quiet`` suppresses progress emission for background
        prefetch extractions.

        ``fmt='jpg'`` writes JPEGs (q2) instead of PNGs — much
        faster codec, and the format SAM2-family loaders expect.
        ``sbs_split=True`` crops the two SBS eyes into left/ and
        right/ subdirectories in one ffmpeg pass (``scale_to``
        is then per-eye); the returned list contains the LEFT
        eye files — right files share the same basename under
        right/.
        """
        for pat in ("*.png", "*.jpg", "*/*.png", "*/*.jpg"):
            for f in frames_dir.glob(pat):
                try:
                    f.unlink()
                except OSError:
                    pass

        from vrautomatte.utils.ffmpeg import _hwaccel_args
        quality = ["-q:v", "2"] if fmt == "jpg" else []
        cmd = [
            "ffmpeg", "-y",
            *_hwaccel_args(),
            "-ss", f"{timestamp:.6f}",
            "-i", str(input_path),
        ]

        if sbs_split:
            left_dir = frames_dir / "left"
            right_dir = frames_dir / "right"
            left_dir.mkdir(exist_ok=True)
            right_dir.mkdir(exist_ok=True)
            scale = ""
            if scale_to is not None:
                w, h = scale_to
                scale = f",scale={w}:{h}:flags=lanczos"
            cmd += [
                "-filter_complex",
                (
                    f"[0:v]crop=iw/2:ih:0:0{scale}[L];"
                    f"[0:v]crop=iw/2:ih:iw/2:0{scale}[R]"
                ),
                "-map", "[L]",
                "-frames:v", str(num_frames), *quality,
                str(left_dir / f"%06d.{fmt}"),
                "-map", "[R]",
                "-frames:v", str(num_frames), *quality,
                str(right_dir / f"%06d.{fmt}"),
            ]
            poll_dir = left_dir
            result_glob = (left_dir, f"*.{fmt}")
        else:
            cmd += ["-frames:v", str(num_frames)]
            if scale_to is not None:
                w, h = scale_to
                cmd += ["-vf", f"scale={w}:{h}:flags=lanczos"]
            cmd += [*quality, str(frames_dir / f"%06d.{fmt}")]
            poll_dir = frames_dir
            result_glob = (frames_dir, f"*.{fmt}")

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        extract_start = time.monotonic()
        last_count = 0

        while process.poll() is None:
            if self._cancelled:
                process.terminate()
                process.wait()
                raise InterruptedError(
                    "Pipeline cancelled"
                )
            time.sleep(0.5)
            if quiet:
                continue
            try:
                count = len(os.listdir(poll_dir))
            except OSError:
                continue
            if count != last_count:
                last_count = count
                elapsed = (
                    time.monotonic() - extract_start
                )
                fps = (
                    count / elapsed if elapsed > 0 else 0
                )
                remaining = num_frames - count
                eta = (
                    remaining / fps if fps > 0 else 0
                )
                self._emit(PipelineProgress(
                    stage="Extracting frames",
                    stage_num=1,
                    total_stages=self._total_stages(),
                    frame_num=count,
                    total_frames=num_frames,
                    fps=fps,
                    eta_sec=eta,
                ))

        if process.returncode != 0:
            raise RuntimeError(
                "ffmpeg chunk extraction failed "
                f"(exit code {process.returncode})"
            )

        glob_dir, glob_pat = result_glob
        return sorted(glob_dir.glob(glob_pat))

    def _extract_frames_with_progress(
        self, cmd, frames_dir, expected, total_stages,
    ):
        """Run ffmpeg extraction with directory-polling progress.

        Runs ffmpeg with no pipe redirection (avoiding Windows
        pipe-buffer issues) and polls the output directory to
        track progress.
        """
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        extract_start = time.monotonic()
        last_count = 0

        while process.poll() is None:
            if self._cancelled:
                process.terminate()
                process.wait()
                raise InterruptedError("Pipeline cancelled")

            time.sleep(0.5)
            try:
                count = len(os.listdir(frames_dir))
            except OSError:
                continue
            if count != last_count:
                last_count = count
                elapsed = time.monotonic() - extract_start
                fps = count / elapsed if elapsed > 0 else 0
                remaining = expected - count
                eta = remaining / fps if fps > 0 else 0
                self._emit(PipelineProgress(
                    stage="Extracting frames",
                    stage_num=1,
                    total_stages=total_stages,
                    frame_num=count,
                    total_frames=expected,
                    fps=fps, eta_sec=eta,
                ))

        if process.returncode != 0:
            raise RuntimeError(
                "ffmpeg frame extraction failed "
                f"(exit code {process.returncode})"
            )

        count = len(os.listdir(frames_dir))
        self._emit(PipelineProgress(
            stage="Extracting frames",
            stage_num=1, total_stages=total_stages,
            frame_num=count, total_frames=expected,
        ))

    # ── GPU auto-config ──────────────────────────────────────

    def _apply_gpu_config(
        self, config: PipelineConfig,
    ) -> dict:
        """Apply GPU auto-configuration. Only overrides defaults."""
        gpu_cfg = auto_configure_gpu()
        defaults = PipelineConfig()

        if config.max_matting_pixels == defaults.max_matting_pixels:
            config.max_matting_pixels = (
                gpu_cfg["max_matting_pixels"]
            )
        if config.ma2_internal_size == defaults.ma2_internal_size:
            config.ma2_internal_size = (
                gpu_cfg["ma2_internal_size"]
            )
        if config.ma2_mem_frames == defaults.ma2_mem_frames:
            config.ma2_mem_frames = gpu_cfg["ma2_mem_frames"]
        if config.downsample_ratio == defaults.downsample_ratio:
            config.downsample_ratio = (
                gpu_cfg["downsample_ratio"]
            )
        if (
            config.sbs_parallel_eyes
            == defaults.sbs_parallel_eyes
        ):
            config.sbs_parallel_eyes = gpu_cfg.get(
                "sbs_parallel_eyes", False
            )
        return gpu_cfg

    # ── Temp directory management ────────────────────────────

    def _setup_temp_dir(
        self, config: PipelineConfig,
        input_path: Path, cfg_hash: str,
    ) -> tuple[Path, bool]:
        """Create or locate the temp directory.

        If auto_resume is True, uses a deterministic name so
        the directory survives for resume.

        Returns:
            (temp_dir_path, is_deterministic)
        """
        if config.temp_dir:
            tmp_base = Path(config.temp_dir)
        else:
            tmp_base = Path(tempfile.gettempdir())
        tmp_base.mkdir(parents=True, exist_ok=True)

        if config.auto_resume:
            cleanup_stale_dirs(tmp_base)
            name = deterministic_temp_name(
                input_path, cfg_hash
            )
            tmp = tmp_base / name
            tmp.mkdir(exist_ok=True)
            return tmp, True

        tmp = Path(tempfile.mkdtemp(
            prefix="vrautomatte_", dir=str(tmp_base)
        ))
        return tmp, False

    # ── Main pipeline ────────────────────────────────────────

    def run(self) -> Path:
        """Execute the full pipeline.

        Returns:
            Path to the final output file.

        Raises:
            RuntimeError: If ffmpeg is not available or fails.
            InterruptedError: If cancelled by user.
        """
        if not check_ffmpeg():
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and "
                "ensure it is on your PATH."
            )

        self._cancelled = False
        self._start_time = time.monotonic()
        config = self.config
        input_path = Path(config.input_path)
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        info = get_video_info(input_path)
        logger.info(
            f"Input: {input_path.name} — "
            f"{info['width']}x{info['height']} @ "
            f"{info['fps']}fps, {info['num_frames']} frames"
        )

        # Auto-configure GPU-dependent settings
        self._apply_gpu_config(config)

        num_to_process = info["num_frames"]
        if config.start_frame > 0 or config.end_frame > 0:
            s = max(config.start_frame, 1)
            e = config.end_frame or info["num_frames"]
            num_to_process = min(
                e, info["num_frames"]
            ) - s + 1

        # Setup temp directory (deterministic for resume)
        cfg_hash = hash_config(config)
        tmp, is_deterministic = self._setup_temp_dir(
            config, input_path, cfg_hash
        )

        # Write log to temp directory
        log_path = tmp / "vrautomatte.log"
        log_id = logger.add(
            str(log_path), level="DEBUG",
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | "
                "{level: <8} | {message}"
            ),
        )

        completed = False
        try:
            frames_dir = tmp / "frames"
            mattes_dir = tmp / "mattes"
            segments_dir = tmp / "segments"
            for d in (frames_dir, mattes_dir, segments_dir):
                d.mkdir(exist_ok=True)

            # Check for resume checkpoint
            resume_seg = 0
            resume_frames = 0
            if config.auto_resume and is_deterministic:
                ckpt = PipelineCheckpoint.load(tmp)
                if ckpt and ckpt.validate(
                    input_path, cfg_hash
                ):
                    # Verify that all prior segment files exist
                    all_present = all(
                        (segments_dir
                         / f"segment_{i:06d}.mp4").exists()
                        for i in range(ckpt.completed_segments)
                    )
                    if all_present:
                        resume_seg = ckpt.completed_segments
                        resume_frames = ckpt.completed_frames
                        logger.info(
                            f"Resuming from segment "
                            f"{resume_seg} "
                            f"({resume_frames:,} frames done)"
                        )
                        self._emit(PipelineProgress(
                            stage=(
                                f"Resuming from segment "
                                f"{resume_seg} "
                                f"({resume_frames:,} frames "
                                f"done)"
                            ),
                            stage_num=2,
                            total_stages=self._total_stages(),
                        ))
                    else:
                        logger.warning(
                            "Checkpoint found but segment "
                            "files are missing — restarting"
                        )
                        PipelineCheckpoint.delete(tmp)

            # Clean leftover PNGs from a partial chunk
            # (cancelled mid-chunk before flush).
            if resume_frames > 0:
                prefetch_dir = tmp / "frames_prefetch"
                for d in (frames_dir, mattes_dir, prefetch_dir):
                    for pat in (
                        "*.png", "*.jpg", "*/*.png", "*/*.jpg",
                    ):
                        for f in d.glob(pat):
                            try:
                                f.unlink()
                            except OSError:
                                pass

            # Create scaler (per-eye dims for SBS) — determines
            # the extraction resolution and the disk estimate.
            use_sbs = config.is_sbs and detect_sbs(
                info["width"], info["height"]
            )
            if use_sbs:
                eye_w = info["width"] // 2
                scaler = FrameScaler(
                    config.max_matting_pixels,
                    (eye_w, info["height"]),
                )
            else:
                scaler = FrameScaler(
                    config.max_matting_pixels,
                    (info["width"], info["height"]),
                )
            if scaler.active:
                tw, th = scaler.target_size
                extract_w = tw * 2 if use_sbs else tw
                extract_h = th
            else:
                extract_w = info["width"]
                extract_h = info["height"]

            # Pre-flight disk check
            estimated = self._estimate_disk_bytes(
                info["width"], info["height"],
                num_to_process,
                total_frames=info["num_frames"],
                input_size=input_path.stat().st_size,
                is_deovr=(
                    config.output_format
                    == OutputFormat.DEOVR_ALPHA
                ),
                chunk_size=config.chunk_size,
                extract_w=extract_w,
                extract_h=extract_h,
            )
            self._check_disk_space(tmp, estimated)
            est_gb = estimated / (1024 ** 3)
            logger.info(
                f"Estimated temp space: {est_gb:.1f} GB"
            )

            # ── Stages 1+2: Chunked extract + matte ──
            total_stages = self._total_stages()
            self._matte_start_time = time.monotonic()

            # Chunk-level models (SAM2Matting) need frames on
            # disk; per-frame models stream via rawvideo pipes
            # (no frame PNGs at all). VRAUTOMATTE_NO_STREAM=1
            # forces the file-based path as an escape hatch.
            chunk_level = config.model_variant == "sam2matting"
            no_stream = (
                os.environ.get("VRAUTOMATTE_NO_STREAM") == "1"
            )
            if chunk_level or no_stream:
                self._run_chunked_pipeline(
                    config, info, input_path,
                    frames_dir, mattes_dir, segments_dir,
                    num_to_process, total_stages,
                    fps_str=info["fps_str"],
                    use_sbs=use_sbs,
                    scaler=scaler,
                    resume_seg=resume_seg,
                    resume_frames=resume_frames,
                    cfg_hash=cfg_hash,
                    estimated_disk_gb=est_gb,
                )
            else:
                self._run_stream_pipeline(
                    config, info, input_path, segments_dir,
                    num_to_process, total_stages,
                    fps_str=info["fps_str"],
                    use_sbs=use_sbs,
                    scaler=scaler,
                    resume_seg=resume_seg,
                    resume_frames=resume_frames,
                    cfg_hash=cfg_hash,
                    estimated_disk_gb=est_gb,
                )

            # ── Stage 3: Concatenate matte segments ──
            logger.info(
                "Stage 3: Concatenating matte segments..."
            )
            self._emit(PipelineProgress(
                stage="Assembling matte video", stage_num=3,
                total_stages=total_stages,
            ))
            matte_video = tmp / "matte.mp4"
            self._concat_matte_segments(
                segments_dir, matte_video, info["fps_str"],
                config.crf,
            )

            if config.output_format == OutputFormat.MATTE_ONLY:
                if scaler.active:
                    # Segments were encoded at matting
                    # resolution — upscale once here.
                    # File-to-file, so the NVENC fallback in
                    # _run_ffmpeg_logged can safely retry.
                    logger.info(
                        "Upscaling matte to original "
                        "resolution..."
                    )
                    self._emit(PipelineProgress(
                        stage="Upscaling matte video",
                        stage_num=3,
                        total_stages=total_stages,
                    ))
                    from vrautomatte.utils.ffmpeg import (
                        _encode_args,
                        _run_ffmpeg_logged,
                    )
                    full_res = tmp / "matte_full.mp4"
                    up_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(matte_video),
                        "-vf", (
                            f"scale={info['width']}:"
                            f"{info['height']}:flags=lanczos"
                        ),
                        *_encode_args("libx264", config.crf),
                        "-pix_fmt", "yuv420p",
                        str(full_res),
                    ]
                    _run_ffmpeg_logged(
                        up_cmd, "matte-upscale",
                        total_frames=num_to_process,
                    )
                    matte_video = full_res
                self._copy_with_audio(
                    matte_video, input_path, output_path
                )
                logger.info(
                    f"Done! Matte saved to: {output_path}"
                )
                self._emit(PipelineProgress(
                    stage="Complete",
                    stage_num=total_stages,
                    total_stages=total_stages,
                ))
                completed = True
                return output_path

            # ── Stage 4: Convert to fisheye ──
            if (
                config.projection
                == ProjectionType.EQUIRECTANGULAR
            ):
                logger.info(
                    "Stage 4: Converting to fisheye..."
                )
                self._emit(PipelineProgress(
                    stage="Converting to fisheye",
                    stage_num=4,
                    total_stages=total_stages,
                ))

                # Trim source to the processed frame range so
                # we don't re-encode the entire original video.
                trimmed_src = tmp / "source_trimmed.mp4"
                fps = info["fps"]
                start_0 = 0
                if config.start_frame > 0:
                    start_0 = config.start_frame - 1
                ss_sec = start_0 / fps
                dur_sec = num_to_process / fps

                from vrautomatte.utils.ffmpeg import (
                    _hwaccel_args,
                    _run_ffmpeg_logged,
                )
                trim_cmd = [
                    "ffmpeg", "-y",
                    *_hwaccel_args(),
                    "-ss", f"{ss_sec:.4f}",
                    "-i", str(input_path),
                    "-t", f"{dur_sec:.4f}",
                    "-c", "copy",
                    str(trimmed_src),
                ]
                logger.info(
                    f"Trimming source to "
                    f"{num_to_process} frames "
                    f"(ss={ss_sec:.1f}s, dur={dur_sec:.1f}s)"
                )
                _run_ffmpeg_logged(
                    trim_cmd, "trim-source",
                    total_frames=num_to_process,
                )

                fisheye_video = tmp / "fisheye_video.mp4"
                fisheye_matte = tmp / "fisheye_matte.mp4"

                convert_to_fisheye(
                    trimmed_src, config.fisheye_mask_path,
                    fisheye_video, config.fisheye_fov,
                    config.codec, config.crf,
                )
                convert_to_fisheye(
                    matte_video, None,
                    fisheye_matte, config.fisheye_fov,
                    config.codec, config.crf,
                )

                # Clean up trimmed source
                try:
                    trimmed_src.unlink()
                except OSError:
                    pass
            else:
                # Already fisheye — trim source to match the
                # matte's frame range, then apply mask to clean
                # up pixels outside the circular fisheye area.
                fps = info["fps"]
                start_0 = 0
                if config.start_frame > 0:
                    start_0 = config.start_frame - 1
                ss_sec = start_0 / fps
                dur_sec = num_to_process / fps

                needs_trim = (
                    config.start_frame > 0
                    or config.end_frame > 0
                )
                if needs_trim:
                    from vrautomatte.utils.ffmpeg import (
                        _hwaccel_args,
                        _run_ffmpeg_logged,
                    )
                    trimmed_src = tmp / "source_trimmed.mp4"
                    trim_cmd = [
                        "ffmpeg", "-y",
                        *_hwaccel_args(),
                        "-ss", f"{ss_sec:.4f}",
                        "-i", str(input_path),
                        "-t", f"{dur_sec:.4f}",
                        "-c", "copy",
                        str(trimmed_src),
                    ]
                    logger.info(
                        f"Trimming fisheye source to "
                        f"{num_to_process} frames"
                    )
                    _run_ffmpeg_logged(
                        trim_cmd, "trim-fisheye",
                        total_frames=num_to_process,
                    )
                    src_video = trimmed_src
                else:
                    src_video = input_path

                # Already-fisheye content fills the entire frame —
                # the DeoVR mask is only for equirect→fisheye
                # conversion artifacts, not native fisheye video.
                fisheye_video = src_video
                fisheye_matte = matte_video

            # ── Stage 5: Pack alpha into video ──
            logger.info(
                "Stage 5: Compositing alpha into video..."
            )
            self._emit(PipelineProgress(
                stage="Packing alpha channel", stage_num=5,
                total_stages=total_stages,
            ))
            pack_alpha(
                fisheye_video, fisheye_matte, output_path,
                "libsvtav1", config.crf,
            )

            logger.info(
                f"Done! Alpha-packed video: {output_path}"
            )
            self._emit(PipelineProgress(
                stage="Complete",
                stage_num=total_stages,
                total_stages=total_stages,
            ))
            completed = True
            return output_path

        finally:
            if self._eye_pool is not None:
                self._eye_pool.shutdown(wait=True)
                self._eye_pool = None
            logger.remove(log_id)
            # Copy log next to output before cleanup
            if completed and log_path.exists():
                dest_log = output_path.with_suffix(".log")
                try:
                    shutil.copy2(log_path, dest_log)
                    logger.info(f"Log saved to {dest_log}")
                except OSError:
                    pass
            # Completed or non-resume: clean temp dir.
            # Incomplete + resume: leave dir for resume.
            if completed or not is_deterministic:
                shutil.rmtree(tmp, ignore_errors=True)

    # ── Chunked pipeline ─────────────────────────────────────

    def _run_chunked_pipeline(
        self, config, info, input_path,
        frames_dir, mattes_dir, segments_dir,
        num_to_process, total_stages,
        *, fps_str, use_sbs, scaler,
        resume_seg=0, resume_frames=0,
        cfg_hash="", estimated_disk_gb=0.0,
    ):
        """Run extraction and matting in interleaved chunks.

        For each chunk:
          1. Extract N frames via ffmpeg keyframe seek, downscaled
             by ffmpeg to the matting resolution (no PIL resizing)
          2. Matte each frame, flush segment, delete PNGs
          3. Save checkpoint for resume

        While chunk N is being matted, chunk N+1 is extracted by a
        background thread into a second directory so the GPU never
        waits on ffmpeg (ffmpeg is a subprocess — no GIL contention).

        Mattes are saved (and segments encoded) at matting
        resolution — upscaling to original resolution happens
        once at final assembly.

        Per-frame processors (RVM, MatAnyone2) are stepped frame
        by frame; chunk-level processors (SAM2Matting) receive
        the whole chunk directory via ``process_chunk``. For SBS
        with a chunk-level processor, ffmpeg splits the eyes
        into left/ and right/ subdirectories during extraction.

        Processor(s) are created once (from the first frame of
        the first active chunk) and reused across all chunks.
        Recurrent state / mask handoff carries across chunks.
        """
        fps = float(info["fps"])
        start_frame_0based = 0
        if config.start_frame > 0:
            start_frame_0based = config.start_frame - 1

        num_chunks = math.ceil(
            num_to_process / config.chunk_size
        )

        chunk_level = config.model_variant == "sam2matting"
        ext_fmt = "jpg" if chunk_level else "png"
        sbs_split = chunk_level and use_sbs

        # ffmpeg extracts directly at matting resolution, so the
        # per-frame PIL downscale/upscale is no longer needed.
        if scaler.active:
            tw, th = scaler.target_size
            if sbs_split:
                extract_size = (tw, th)  # per-eye outputs
            elif use_sbs:
                extract_size = (tw * 2, th)
            else:
                extract_size = (tw, th)
            self._emit(PipelineProgress(
                stage=(
                    f"Processing at {tw}x{th} "
                    f"for your GPU"
                ),
                stage_num=2,
                total_stages=total_stages,
            ))
        else:
            extract_size = None

        # Second frames dir lets extraction of chunk N+1 overlap
        # with matting of chunk N.
        prefetch_dir = frames_dir.with_name("frames_prefetch")
        prefetch_dir.mkdir(exist_ok=True)
        chunk_dirs = (frames_dir, prefetch_dir)

        def chunk_params(idx):
            offset = idx * config.chunk_size
            count = min(
                config.chunk_size, num_to_process - offset
            )
            start = start_frame_0based + offset
            ts = start / fps if fps > 0 else 0
            return ts, count

        # Processor(s) — created lazily from first active chunk
        processor = None
        proc_l = None
        proc_r = None

        seg_idx = resume_seg
        global_frame_idx = resume_frames

        pool = ThreadPoolExecutor(max_workers=1)
        pending = None
        try:
            for chunk_idx in range(num_chunks):
                chunk_offset = chunk_idx * config.chunk_size
                if chunk_offset < resume_frames:
                    continue

                if self._cancelled:
                    raise InterruptedError(
                        "Pipeline cancelled by user"
                    )

                ts, chunk_frames = chunk_params(chunk_idx)
                logger.info(
                    f"Chunk {chunk_idx + 1}/{num_chunks} "
                    f"({chunk_frames} frames)..."
                )
                self._emit(PipelineProgress(
                    stage=(
                        f"Extracting chunk "
                        f"{chunk_idx + 1}/{num_chunks}"
                    ),
                    stage_num=1,
                    total_stages=total_stages,
                    frame_num=global_frame_idx,
                    total_frames=num_to_process,
                ))

                if pending is not None:
                    # Chunk was prefetched during the previous
                    # chunk's matting — usually ready already.
                    frame_files = pending.result()
                    pending = None
                else:
                    frame_files = self._extract_chunk(
                        input_path,
                        chunk_dirs[chunk_idx % 2],
                        ts, chunk_frames,
                        scale_to=extract_size,
                        fmt=ext_fmt,
                        sbs_split=sbs_split,
                    )

                # Kick off extraction of the next chunk while
                # this chunk is matting.
                if chunk_idx + 1 < num_chunks:
                    next_ts, next_frames = chunk_params(
                        chunk_idx + 1
                    )
                    pending = pool.submit(
                        self._extract_chunk,
                        input_path,
                        chunk_dirs[(chunk_idx + 1) % 2],
                        next_ts, next_frames,
                        scale_to=extract_size,
                        quiet=True,
                        fmt=ext_fmt,
                        sbs_split=sbs_split,
                    )

                if not frame_files:
                    logger.warning(
                        f"Chunk {chunk_idx + 1} extracted "
                        f"0 frames, skipping"
                    )
                    continue

                # ── Create processor(s) on first chunk ──
                needs_first = (
                    config.model_variant
                    in ("matanyone2", "sam2matting")
                    or config.pov_mode
                )

                if use_sbs and proc_l is None:
                    if sbs_split:
                        # frame_files are the left-eye files;
                        # right eye shares the basename.
                        left0 = np.array(Image.open(
                            frame_files[0]
                        ).convert("RGB"))
                        right0 = np.array(Image.open(
                            frame_files[0].parent.parent
                            / "right" / frame_files[0].name
                        ).convert("RGB"))
                        logger.info(
                            "SBS: initialising left-eye "
                            "processor..."
                        )
                        self._proc_l = self._make_processor(
                            config, left0
                        )
                        logger.info(
                            "SBS: initialising right-eye "
                            "processor..."
                        )
                        self._proc_r = self._make_processor(
                            config, right0
                        )
                    else:
                        first_full = np.array(Image.open(
                            frame_files[0]
                        ).convert("RGB"))
                        self._init_sbs_processors(
                            config, first_full, needs_first,
                        )
                    proc_l = self._proc_l
                    proc_r = self._proc_r
                elif not use_sbs and processor is None:
                    first_seed = None
                    if needs_first:
                        # Frames are already at matting
                        # resolution — use directly.
                        first_seed = np.array(
                            Image.open(
                                frame_files[0]
                            ).convert("RGB")
                        )
                    processor = self._make_processor(
                        config, first_seed
                    )

                # ── Process this chunk's frames ──
                if chunk_level:
                    seg_frame = self._matte_chunk_level(
                        processor, proc_l, proc_r, use_sbs,
                        chunk_dirs[chunk_idx % 2],
                        frame_files, mattes_dir,
                        global_frame_idx, num_to_process,
                        estimated_disk_gb,
                    )
                    global_frame_idx += seg_frame
                else:
                    seg_frame = 0
                    for i, frame_file in enumerate(frame_files):
                        if self._cancelled:
                            raise InterruptedError(
                                "Pipeline cancelled by user"
                            )

                        if i % _DISK_CHECK_INTERVAL == 0:
                            self._check_disk_free(mattes_dir)

                        frame_arr = np.array(
                            Image.open(
                                frame_file
                            ).convert("RGB")
                        )

                        if use_sbs:
                            matte_arr = self._process_sbs_frame(
                                frame_arr, proc_l, proc_r,
                                parallel=(
                                    config.sbs_parallel_eyes
                                ),
                            )
                        else:
                            matte_arr = processor.process_frame(
                                frame_arr
                            )

                        seg_frame += 1
                        Image.fromarray(
                            matte_arr, mode="L"
                        ).save(
                            mattes_dir
                            / f"frame_{seg_frame:06d}.png"
                        )

                        try:
                            frame_file.unlink()
                        except OSError:
                            pass

                        global_frame_idx += 1
                        stage = (
                            "Matting SBS (L+R)"
                            if use_sbs
                            else "Generating mattes"
                        )
                        self._emit_matte_progress(
                            global_frame_idx - 1,
                            num_to_process,
                            frame_arr, matte_arr,
                            stage=stage,
                            estimated_disk_gb=estimated_disk_gb,
                        )
                        del frame_arr, matte_arr

                # Flush segment
                if seg_frame > 0:
                    self._flush_matte_segment(
                        mattes_dir, segments_dir, seg_idx,
                        fps_str=fps_str, crf=config.crf,
                    )
                    seg_idx += 1
                    global_done = global_frame_idx
                    self._save_checkpoint(
                        config, cfg_hash, input_path,
                        num_to_process, seg_idx,
                        global_done, segments_dir,
                    )

        except BaseException:
            # Make the background extractor's poll loop abort
            # quickly so pool shutdown doesn't block.
            self._cancelled = True
            raise
        finally:
            pool.shutdown(wait=True)
            if use_sbs:
                if proc_l is not None:
                    proc_l.cleanup()
                if proc_r is not None:
                    proc_r.cleanup()
            elif processor is not None:
                processor.cleanup()

    def _init_sbs_processors(
        self, config, first_frame, needs_first,
    ):
        """Create left/right eye processors for SBS mode.

        Args:
            first_frame: Full SBS frame array at matting
                resolution; split into per-eye seeds here.
        """
        logger.info("SBS mode: processing per-eye")
        left_f, right_f = split_frame(first_frame)

        if needs_first:
            left_seed = left_f
            right_seed = right_f
        else:
            left_seed = None
            right_seed = None

        logger.info(
            "SBS: initialising left-eye processor..."
        )
        self._proc_l = self._make_processor(
            config, left_seed
        )

        if getattr(self._proc_l, "supports_pair", False):
            # RVM-family: one shared model mattes both eyes in
            # a single batched forward pass (half the calls,
            # half the model VRAM).
            logger.info(
                "SBS: batched two-eye processing "
                "(single shared model)"
            )
            self._proc_r = self._proc_l
            return

        logger.info(
            "SBS: initialising right-eye processor..."
        )
        self._proc_r = self._make_processor(
            config, right_seed
        )

    def _matte_chunk_level(
        self, processor, proc_l, proc_r, use_sbs,
        chunk_dir, frame_files, mattes_dir,
        start_idx, total, estimated_disk_gb,
    ):
        """Run a chunk-level processor over one extracted chunk.

        Chunk-level processors (SAM2Matting) consume a whole
        frame directory and yield mattes as a generator. For
        SBS the two eye generators are stepped in lockstep and
        the mattes merged.

        Returns:
            Number of frames matted (matte PNGs written).
        """
        if use_sbs:
            left_dir = chunk_dir / "left"
            right_dir = chunk_dir / "right"
            gen = zip(
                proc_l.process_chunk(left_dir),
                proc_r.process_chunk(right_dir),
            )
            stage = "Matting SBS (L+R)"
        else:
            gen = (
                (m, None)
                for m in processor.process_chunk(chunk_dir)
            )
            stage = "Generating mattes"

        seg_frame = 0
        for i, (m_left, m_right) in enumerate(gen):
            if self._cancelled:
                raise InterruptedError(
                    "Pipeline cancelled by user"
                )
            if i % _DISK_CHECK_INTERVAL == 0:
                self._check_disk_free(mattes_dir)

            if m_right is not None:
                matte_arr = merge_mattes(m_left, m_right)
            else:
                matte_arr = m_left

            seg_frame += 1
            Image.fromarray(matte_arr, mode="L").save(
                mattes_dir / f"frame_{seg_frame:06d}.png"
            )

            idx = start_idx + i
            # Load the source frame only when the progress
            # emitter will actually use it (every 10th frame).
            if idx % 10 == 0 or idx == total - 1:
                src = None
                if i < len(frame_files):
                    src = np.array(Image.open(
                        frame_files[i]
                    ).convert("RGB"))
                    if m_right is not None:
                        right_f = (
                            frame_files[i].parent.parent
                            / "right" / frame_files[i].name
                        )
                        if right_f.exists():
                            src = np.concatenate(
                                [src, np.array(Image.open(
                                    right_f
                                ).convert("RGB"))],
                                axis=1,
                            )
                self._emit_matte_progress(
                    idx, total, src, matte_arr,
                    stage=stage,
                    estimated_disk_gb=estimated_disk_gb,
                )

        # Delete this chunk's source frames.
        for f in frame_files:
            try:
                f.unlink()
            except OSError:
                pass
            if use_sbs:
                try:
                    (f.parent.parent / "right"
                     / f.name).unlink()
                except OSError:
                    pass

        return seg_frame

    def _save_checkpoint(
        self, config, cfg_hash, input_path,
        total_frames, seg_idx, frames_done, segments_dir,
    ):
        """Persist resume state after a segment flush."""
        if not (config.auto_resume and cfg_hash):
            return
        ckpt = PipelineCheckpoint(
            input_path=str(input_path),
            input_hash=hash_file_head(input_path),
            config_hash=cfg_hash,
            total_frames=total_frames,
            chunk_size=config.chunk_size,
            completed_segments=seg_idx,
            completed_frames=frames_done,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        ckpt.save(segments_dir.parent)

    # ── Streaming pipeline ───────────────────────────────────

    def _run_stream_pipeline(
        self, config, info, input_path, segments_dir,
        num_to_process, total_stages,
        *, fps_str, use_sbs, scaler,
        resume_seg=0, resume_frames=0,
        cfg_hash="", estimated_disk_gb=0.0,
    ):
        """Stream frames through matting via rawvideo pipes.

        ffmpeg decodes (and downscales) the source into a raw
        RGB pipe; mattes stream back out through a raw grayscale
        pipe into the segment encoder. No frame images touch the
        disk, and decode runs ahead of the GPU via the reader's
        internal queue.

        Segments are encoded at matting resolution and cut every
        ``chunk_size`` frames so checkpoint/resume keeps working
        exactly as in the file-based path.
        """
        from vrautomatte.pipeline.framestream import (
            FrameStreamReader,
            SegmentStreamWriter,
        )

        fps = float(info["fps"])
        start_frame_0based = 0
        if config.start_frame > 0:
            start_frame_0based = config.start_frame - 1

        if scaler.active:
            tw, th = scaler.target_size
            out_w = tw * 2 if use_sbs else tw
            out_h = th
            self._emit(PipelineProgress(
                stage=(
                    f"Processing at {tw}x{th} "
                    f"for your GPU"
                ),
                stage_num=2,
                total_stages=total_stages,
            ))
        else:
            out_w, out_h = info["width"], info["height"]

        remaining = num_to_process - resume_frames
        start_ts = 0.0
        if fps > 0:
            start_ts = (
                start_frame_0based + resume_frames
            ) / fps

        self._emit(PipelineProgress(
            stage="Starting video stream",
            stage_num=1,
            total_stages=total_stages,
            frame_num=resume_frames,
            total_frames=num_to_process,
        ))
        reader = FrameStreamReader(
            input_path, (out_w, out_h),
            start_ts=start_ts,
            num_frames=remaining,
            scale=scaler.active,
        )

        needs_first = (
            config.model_variant == "matanyone2"
            or config.pov_mode
        )
        processor = None
        proc_l = None
        proc_r = None
        writer = None
        seg_idx = resume_seg
        global_frame_idx = resume_frames
        seg_frames = 0

        try:
            frame = reader.read()
            if frame is None:
                raise RuntimeError(
                    "No frames decoded from input — check "
                    "the file and frame range."
                )

            # ── Create processor(s) from the first frame ──
            if use_sbs:
                self._init_sbs_processors(
                    config, frame, needs_first,
                )
                proc_l = self._proc_l
                proc_r = self._proc_r
            else:
                processor = self._make_processor(
                    config, frame if needs_first else None
                )

            stage = (
                "Matting SBS (L+R)"
                if use_sbs else "Generating mattes"
            )

            def emit_matte(matte, src_frame):
                """Write one matte, cutting segments and
                checkpointing every chunk_size frames."""
                nonlocal writer, seg_frames
                nonlocal seg_idx, global_frame_idx
                if writer is None:
                    writer = SegmentStreamWriter(
                        segments_dir
                        / f"segment_{seg_idx:06d}.mp4",
                        (out_w, out_h),
                        fps_str, config.crf,
                    )
                writer.write(matte)
                seg_frames += 1
                global_frame_idx += 1
                self._emit_matte_progress(
                    global_frame_idx - 1,
                    num_to_process,
                    src_frame, matte,
                    stage=stage,
                    estimated_disk_gb=estimated_disk_gb,
                )
                if seg_frames >= config.chunk_size:
                    writer.close()
                    writer = None
                    seg_frames = 0
                    seg_idx += 1
                    self._save_checkpoint(
                        config, cfg_hash, input_path,
                        num_to_process, seg_idx,
                        global_frame_idx, segments_dir,
                    )

            # Half-rate matting: matte every Nth frame and
            # linearly interpolate the alpha in between. The
            # matte is temporally smooth, so lerp is visually
            # equivalent at stride 2 for 60fps content.
            stride = max(1, int(config.matte_stride))
            if stride > 1:
                logger.info(
                    f"Half-rate matting: every {stride}. "
                    f"frame matted, alpha interpolated"
                )
            last_matte = None
            deferred = 0
            frame_no = 0  # index within this run

            while frame is not None:
                if self._cancelled:
                    raise InterruptedError(
                        "Pipeline cancelled by user"
                    )
                if frame_no % _DISK_CHECK_INTERVAL == 0:
                    self._check_disk_free(segments_dir)

                do_matte = (
                    stride == 1
                    or last_matte is None
                    or frame_no % stride == 0
                )

                if do_matte:
                    if use_sbs:
                        matte_arr = self._process_sbs_frame(
                            frame, proc_l, proc_r,
                            parallel=config.sbs_parallel_eyes,
                        )
                    else:
                        matte_arr = (
                            processor.process_frame(frame)
                        )

                    if deferred:
                        lastf = last_matte.astype(np.float32)
                        curf = matte_arr.astype(np.float32)
                        for k in range(1, deferred + 1):
                            w = k / (deferred + 1)
                            interp = (
                                (1.0 - w) * lastf + w * curf
                            ).astype(np.uint8)
                            emit_matte(interp, None)
                        deferred = 0

                    emit_matte(matte_arr, frame)
                    last_matte = matte_arr
                else:
                    deferred += 1

                frame_no += 1
                frame = reader.read()

            # Trailing skipped frames at EOF: repeat the last
            # matte so every source frame has one.
            for _ in range(deferred):
                emit_matte(last_matte, None)

            # Final partial segment
            if writer is not None:
                writer.close()
                writer = None
                seg_idx += 1
                self._save_checkpoint(
                    config, cfg_hash, input_path,
                    num_to_process, seg_idx,
                    global_frame_idx, segments_dir,
                )

        except BaseException:
            self._cancelled = True
            if writer is not None:
                # Partial segment can't be trusted — remove it
                # so resume re-processes those frames.
                writer.abort()
                writer = None
            raise
        finally:
            reader.close()
            if use_sbs:
                if proc_l is not None:
                    proc_l.cleanup()
                if proc_r is not None:
                    proc_r.cleanup()
            elif processor is not None:
                processor.cleanup()

    _eye_streams = threading.local()

    def _process_sbs_frame(
        self, frame_arr, proc_l, proc_r, parallel=False,
    ):
        """Process one SBS frame through both eye processors.

        The frame is already at matting resolution — no per-eye
        scaling is needed. Three strategies, fastest first:
        - shared pair-capable processor (RVM): one batched
          forward pass for both eyes
        - ``parallel``: both eyes concurrently on worker threads
          with per-thread CUDA streams (two-instance models,
          i.e. MatAnyone2 — needs the VRAM headroom)
        - sequential fallback
        """
        left, right = split_frame(frame_arr)

        if proc_l is proc_r and getattr(
            proc_l, "supports_pair", False
        ):
            left_m, right_m = proc_l.process_frame_pair(
                left, right
            )
        elif parallel and proc_l is not proc_r:
            if self._eye_pool is None:
                self._eye_pool = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="eye",
                )
            f_l = self._eye_pool.submit(
                self._run_eye, proc_l, left
            )
            f_r = self._eye_pool.submit(
                self._run_eye, proc_r, right
            )
            left_m = f_l.result()
            right_m = f_r.result()
        else:
            left_m = proc_l.process_frame(left)
            right_m = proc_r.process_frame(right)

        matte = merge_mattes(left_m, right_m)
        del left, right, left_m, right_m
        return matte

    @classmethod
    def _run_eye(cls, proc, eye):
        """Run one eye on this worker thread's CUDA stream.

        Streams let the two eyes' GPU work overlap; the matte
        returns as numpy (the D2H copy synchronizes), so no
        cross-stream tensor sharing occurs.
        """
        try:
            import torch
            if torch.cuda.is_available():
                stream = getattr(
                    cls._eye_streams, "stream", None
                )
                if stream is None:
                    stream = torch.cuda.Stream()
                    cls._eye_streams.stream = stream
                with torch.cuda.stream(stream):
                    return proc.process_frame(eye)
        except ImportError:
            pass
        return proc.process_frame(eye)

    # ── Processor creation ───────────────────────────────────

    def _make_processor(self, config, first_frame):
        """Create a matting processor from config.

        Args:
            config: Pipeline configuration.
            first_frame: First video frame as uint8 RGB array.
                Required for matanyone2 and pov_mode; None
                otherwise.
        """
        needs_first_frame = (
            config.model_variant == "matanyone2"
            or config.pov_mode
        )
        if needs_first_frame and first_frame is not None:
            self._emit(PipelineProgress(
                stage="Generating first-frame mask",
                stage_num=2,
                total_stages=self._total_stages(),
            ))

        processor = create_processor(
            variant=config.model_variant,
            downsample_ratio=config.downsample_ratio,
            first_frame=(
                first_frame if needs_first_frame else None
            ),
            pov_mode=config.pov_mode,
            use_fp16=config.use_fp16,
            max_internal_size=config.ma2_internal_size,
            max_mem_frames=config.ma2_mem_frames,
            use_long_term=config.ma2_use_long_term,
            compile_model=config.ma2_compile_model,
            roi_matting=config.roi_matting,
            max_subjects=config.max_subjects,
        )

        if config.temporal_smoothing < 1.0:
            if getattr(processor, "chunk_level", False):
                logger.info(
                    "Temporal smoothing skipped — not "
                    "supported for chunk-level models"
                )
            else:
                processor = AlphaSmoother(
                    processor,
                    weight=config.temporal_smoothing,
                )
                logger.info(
                    f"Temporal smoothing enabled "
                    f"(weight={config.temporal_smoothing})"
                )

        return processor

    # ── Segment management ───────────────────────────────────

    def _flush_matte_segment(
        self, mattes_dir, segments_dir, seg_idx,
        fps_str, crf,
    ):
        """Encode PNGs into a segment video, then delete them.

        Uses libx264 with yuv420p for broad concat compatibility.
        Segments stay at matting resolution — upscaling to the
        original resolution happens once at final assembly.
        """
        segment_path = (
            segments_dir / f"segment_{seg_idx:06d}.mp4"
        )
        from vrautomatte.utils.ffmpeg import _encode_args
        base = [
            "ffmpeg", "-y",
            "-framerate", fps_str,
            "-i", str(mattes_dir / "frame_%06d.png"),
        ]
        tail = ["-pix_fmt", "yuv420p", str(segment_path)]
        devnull = dict(
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            subprocess.run(
                base + _encode_args("libx264", crf) + tail,
                **devnull,
            )
        except subprocess.CalledProcessError:
            # NVENC may fail when the GPU is busy with the
            # matting model — fall back to CPU encoding.
            logger.warning(
                "NVENC failed for segment encode, "
                "falling back to CPU libx264"
            )
            subprocess.run(
                base + ["-c:v", "libx264", "-crf", str(crf)]
                + tail,
                **devnull,
            )

        for png in mattes_dir.glob("frame_*.png"):
            try:
                png.unlink()
            except OSError:
                pass

        logger.debug(
            f"Flushed segment {seg_idx} -> "
            f"{segment_path.name}"
        )

    def _concat_matte_segments(
        self, segments_dir, output_path, fps_str, crf,
    ):
        """Concatenate segment videos via concat demuxer.

        Uses stream copy — no re-encode, fast regardless of
        total frame count.
        """
        segments = sorted(segments_dir.glob("segment_*.mp4"))
        if not segments:
            raise RuntimeError(
                "No matte segments found — matting "
                "stage may have failed."
            )

        if len(segments) == 1:
            shutil.copy2(segments[0], output_path)
            return

        concat_list = segments_dir / "concat_list.txt"
        with concat_list.open("w") as f:
            for seg in segments:
                safe = str(seg).replace(
                    "\\", "/"
                ).replace("'", "\'")
                f.write(f"file '{safe}'\n")

        subprocess.run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            "-c", "copy",
            str(output_path),
        ], check=True, stdin=subprocess.DEVNULL,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        concat_list.unlink(missing_ok=True)
        logger.info(
            f"Concatenated {len(segments)} segments "
            f"-> {output_path.name}"
        )

    # ── Progress ─────────────────────────────────────────────

    def _emit_matte_progress(
        self, i, total, source, matte,
        stage="Generating mattes",
        estimated_disk_gb=0.0,
    ):
        """Emit progress during matting (every 10 frames)."""
        if i % 10 != 0 and i != total - 1:
            return
        elapsed = time.monotonic() - self._matte_start_time
        fps = (i + 1) / elapsed if elapsed > 0 else 0
        remaining = total - (i + 1)
        eta = remaining / fps if fps > 0 else 0
        self._emit(PipelineProgress(
            stage=stage, stage_num=2,
            total_stages=self._total_stages(),
            frame_num=i + 1, total_frames=total,
            source_frame=source, matte_frame=matte,
            eta_sec=eta, fps=fps,
            estimated_disk_gb=estimated_disk_gb,
        ))

    # ── Audio / output ───────────────────────────────────────

    def _copy_with_audio(
        self, video_path, audio_source, output_path,
    ):
        """Copy video and mux audio from the original source."""
        try:
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_source),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                str(output_path),
            ], check=True,
               stdin=subprocess.DEVNULL,
               stdout=subprocess.DEVNULL,
               stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            logger.debug(
                "No audio track found, copying video only"
            )
            shutil.copy2(video_path, output_path)

    def _total_stages(self):
        """Calculate total stages based on config."""
        if self.config.output_format == OutputFormat.MATTE_ONLY:
            return 3  # extract+matte, assemble
        return 6  # + fisheye convert, red channel, pack

    # ── Disk management ──────────────────────────────────────

    @staticmethod
    def _estimate_disk_bytes(
        width, height, num_frames,
        total_frames=0, input_size=0,
        is_deovr=False, chunk_size=500,
        extract_w=0, extract_h=0,
    ):
        """Estimate peak temp disk under the chunked pipeline.

        Uses the actual input file size to estimate compressed
        video sizes instead of guessing compression ratios.

        Peak = per-chunk PNGs + proportional input file size
        (for intermediates like fisheye conversions / matte video).

        Frame PNGs are stored at extraction (matting) resolution
        when the scaler is active. Source PNGs are double-buffered
        (active chunk + prefetched next chunk); matte PNGs
        accumulate up to one chunk before each segment flush.
        """
        ew = extract_w or width
        eh = extract_h or height
        source_png = int(ew * eh * 3 * 0.5)
        matte_png = int(ew * eh * 0.5)
        per_frame = source_png * 2 + matte_png
        chunk_pngs = per_frame * min(num_frames, chunk_size)

        # Proportional input size for the processed range
        if total_frames > 0 and input_size > 0:
            frac = min(num_frames / total_frames, 1.0)
            proportional = int(input_size * frac)
        else:
            # Fallback: estimate ~6% of raw frame data
            proportional = int(
                width * height * 3 * num_frames * 0.06
            )

        # Intermediates scale with the proportional size:
        # segments + matte.mp4 ≈ 1× proportional (grayscale)
        # DeoVR adds fisheye_video + fisheye_matte ≈ 2×
        multiplier = 3 if is_deovr else 1
        intermediates = proportional * multiplier

        return chunk_pngs + intermediates

    @staticmethod
    def _check_disk_space(path, required):
        """Raise if drive has less than required + margin."""
        free = shutil.disk_usage(path).free
        needed = required + _MIN_FREE_BYTES
        if free < needed:
            free_gb = free / (1024 ** 3)
            need_gb = needed / (1024 ** 3)
            raise RuntimeError(
                f"Not enough disk space. "
                f"Available: {free_gb:.1f} GB, "
                f"estimated need: {need_gb:.1f} GB "
                f"(including 1 GB safety margin). "
                f"Free up space or reduce frame range."
            )

    @staticmethod
    def _check_disk_free(path):
        """Raise if free space drops below safety margin."""
        free = shutil.disk_usage(path).free
        if free < _MIN_FREE_BYTES:
            free_mb = free / (1024 ** 2)
            raise RuntimeError(
                f"Disk space critically low "
                f"({free_mb:.0f} MB remaining). "
                f"Processing stopped to prevent "
                f"filling the drive. Free up space "
                f"and retry."
            )
