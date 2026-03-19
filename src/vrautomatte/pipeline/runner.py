"""Pipeline orchestrator — chains all steps from input to output.

Steps:
1-2. Extract frames and generate mattes (chunked)
3. Reassemble matte frames into a video
4. (Optional) Convert equirectangular -> fisheye
5. (Optional) Pack alpha channel for DeoVR format
"""

import gc
import math
import os
import shutil
import subprocess
import tempfile
import time
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
    merge_frames,
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
    ) -> list[Path]:
        """Extract frames using fast keyframe seek.

        Uses ``-ss`` before ``-i`` for keyframe-based seeking.
        ~1-2 frame imprecision at chunk boundaries is acceptable
        for VR content.  Polls the output directory so the UI
        stays responsive and shows extraction progress.
        """
        for f in frames_dir.glob("frame_*.png"):
            try:
                f.unlink()
            except OSError:
                pass

        from vrautomatte.utils.ffmpeg import _hwaccel_args
        cmd = [
            "ffmpeg", "-y",
            *_hwaccel_args(),
            "-ss", f"{timestamp:.6f}",
            "-i", str(input_path),
            "-frames:v", str(num_frames),
            str(frames_dir / "frame_%06d.png"),
        ]

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
            try:
                count = len(os.listdir(frames_dir))
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

        return sorted(frames_dir.glob("frame_*.png"))

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
                for d in (frames_dir, mattes_dir):
                    for png in d.glob("*.png"):
                        try:
                            png.unlink()
                        except OSError:
                            pass

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
            )
            self._check_disk_space(tmp, estimated)
            est_gb = estimated / (1024 ** 3)
            logger.info(
                f"Estimated temp space: {est_gb:.1f} GB"
            )

            # ── Stages 1+2: Chunked extract + matte ──
            total_stages = self._total_stages()
            self._matte_start_time = time.monotonic()

            self._run_chunked_pipeline(
                config, info, input_path,
                frames_dir, mattes_dir, segments_dir,
                num_to_process, total_stages,
                fps_str=info["fps_str"],
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
            # Completed or non-resume: clean temp dir.
            # Incomplete + resume: leave dir for resume.
            if completed or not is_deterministic:
                shutil.rmtree(tmp, ignore_errors=True)

    # ── Chunked pipeline ─────────────────────────────────────

    def _run_chunked_pipeline(
        self, config, info, input_path,
        frames_dir, mattes_dir, segments_dir,
        num_to_process, total_stages,
        *, fps_str, resume_seg=0, resume_frames=0,
        cfg_hash="", estimated_disk_gb=0.0,
    ):
        """Run extraction and matting in interleaved chunks.

        For each chunk:
          1. Extract N frames via ffmpeg keyframe seek
          2. Matte each frame, flush segment, delete PNGs
          3. Save checkpoint for resume

        Processor(s) are created once (from the first frame of
        the first active chunk) and reused across all chunks.
        Recurrent state carries across chunk boundaries.
        """
        fps = float(info["fps"])
        start_frame_0based = 0
        if config.start_frame > 0:
            start_frame_0based = config.start_frame - 1

        num_chunks = math.ceil(
            num_to_process / config.chunk_size
        )
        use_sbs = config.is_sbs and detect_sbs(
            info["width"], info["height"]
        )

        # Create scaler (per-eye dims for SBS)
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
            self._emit(PipelineProgress(
                stage=(
                    f"Processing at {tw}x{th} "
                    f"for your GPU"
                ),
                stage_num=2,
                total_stages=total_stages,
            ))

        # Processor(s) — created lazily from first active chunk
        processor = None
        proc_l = None
        proc_r = None

        seg_idx = resume_seg
        global_frame_idx = resume_frames

        try:
            for chunk_idx in range(num_chunks):
                chunk_offset = chunk_idx * config.chunk_size
                if chunk_offset < resume_frames:
                    continue

                if self._cancelled:
                    raise InterruptedError(
                        "Pipeline cancelled by user"
                    )

                # Extract this chunk
                chunk_start = (
                    start_frame_0based + chunk_offset
                )
                chunk_frames = min(
                    config.chunk_size,
                    num_to_process - chunk_offset,
                )
                ts = (
                    chunk_start / fps if fps > 0 else 0
                )
                logger.info(
                    f"Extracting chunk "
                    f"{chunk_idx + 1}/{num_chunks} "
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

                frame_files = self._extract_chunk(
                    input_path, frames_dir,
                    ts, chunk_frames,
                )

                if not frame_files:
                    logger.warning(
                        f"Chunk {chunk_idx + 1} extracted "
                        f"0 frames, skipping"
                    )
                    continue

                # ── Create processor(s) on first chunk ──
                needs_first = (
                    config.model_variant == "matanyone2"
                    or config.pov_mode
                )

                if use_sbs and proc_l is None:
                    self._init_sbs_processors(
                        config, frame_files[0],
                        scaler, needs_first,
                    )
                    proc_l = self._proc_l
                    proc_r = self._proc_r
                elif not use_sbs and processor is None:
                    first_arr = np.array(
                        Image.open(
                            frame_files[0]
                        ).convert("RGB")
                    )
                    first_seed = (
                        scaler.downscale(first_arr)
                        if needs_first else None
                    )
                    del first_arr
                    processor = self._make_processor(
                        config, first_seed
                    )

                # ── Process this chunk's frames ──
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
                            frame_arr, proc_l, proc_r, scaler,
                        )
                    else:
                        scaled = scaler.downscale(frame_arr)
                        matte_arr = processor.process_frame(
                            scaled
                        )
                        matte_arr = scaler.upscale_matte(
                            matte_arr
                        )
                        del scaled

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

                    # Save checkpoint
                    if config.auto_resume and cfg_hash:
                        ckpt = PipelineCheckpoint(
                            input_path=str(input_path),
                            input_hash=hash_file_head(
                                input_path
                            ),
                            config_hash=cfg_hash,
                            total_frames=num_to_process,
                            chunk_size=config.chunk_size,
                            completed_segments=seg_idx,
                            completed_frames=global_frame_idx,
                            timestamp=time.strftime(
                                "%Y-%m-%dT%H:%M:%S"
                            ),
                        )
                        ckpt.save(segments_dir.parent)

                # Clear VRAM between chunks
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except ImportError:
                    pass
                gc.collect()

        finally:
            if use_sbs:
                if proc_l is not None:
                    proc_l.cleanup()
                if proc_r is not None:
                    proc_r.cleanup()
            elif processor is not None:
                processor.cleanup()

    def _init_sbs_processors(
        self, config, first_frame_path, scaler, needs_first,
    ):
        """Create left/right eye processors for SBS mode."""
        logger.info("SBS mode: processing per-eye")
        first_full = np.array(
            Image.open(first_frame_path).convert("RGB")
        )
        left_f, right_f = split_frame(first_full)
        del first_full

        if needs_first:
            left_seed = scaler.downscale(left_f)
            right_seed = scaler.downscale(right_f)
        else:
            left_seed = None
            right_seed = None
        del left_f, right_f

        logger.info(
            "SBS: initialising left-eye processor..."
        )
        self._proc_l = self._make_processor(
            config, left_seed
        )
        logger.info(
            "SBS: initialising right-eye processor..."
        )
        self._proc_r = self._make_processor(
            config, right_seed
        )

    @staticmethod
    def _process_sbs_frame(frame_arr, proc_l, proc_r, scaler):
        """Process one SBS frame through both eye processors."""
        left, right = split_frame(frame_arr)
        left = scaler.downscale(left)
        right = scaler.downscale(right)

        left_m = proc_l.process_frame(left)
        right_m = proc_r.process_frame(right)

        left_m = scaler.upscale_matte(left_m)
        right_m = scaler.upscale_matte(right_m)

        matte = merge_mattes(left_m, right_m)
        del left, right, left_m, right_m
        return matte

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
        )

        if config.temporal_smoothing < 1.0:
            processor = AlphaSmoother(
                processor, weight=config.temporal_smoothing
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
    ):
        """Estimate peak temp disk under the chunked pipeline.

        Uses the actual input file size to estimate compressed
        video sizes instead of guessing compression ratios.

        Peak = per-chunk PNGs + proportional input file size
        (for intermediates like fisheye conversions / matte video).
        """
        # Per-chunk source PNGs (deleted as matted, peak at
        # the start of each chunk before matting begins)
        source_png = int(width * height * 3 * 0.5)
        chunk_pngs = source_png * min(num_frames, chunk_size)

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
