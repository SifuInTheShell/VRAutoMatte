"""Pipeline orchestrator — chains all steps from input to output.

Steps:
1. Extract frames from input video
2. Generate AI alpha matte for each frame (via RVM)
3. Reassemble matte frames into a video
4. (Optional) Convert equirectangular → fisheye
5. (Optional) Pack alpha channel for DeoVR format
"""

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable

import numpy as np
from loguru import logger
from PIL import Image

from vrautomatte.pipeline.matte import create_processor
from vrautomatte.utils.ffmpeg import (
    check_ffmpeg,
    convert_to_fisheye,
    get_video_info,
    matte_to_red_channel,
    pack_alpha,
)
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
    model_variant: str = "mobilenetv3"   # mobilenetv3/resnet50/matanyone2
    downsample_ratio: float = 0.25       # RVM processing scale

    # Output settings
    output_format: OutputFormat = OutputFormat.MATTE_ONLY
    codec: str = "libx265"               # 'libx265' or 'libx264'
    crf: int = 18                        # Quality (lower = better)

    # VR-specific
    projection: ProjectionType = ProjectionType.EQUIRECTANGULAR
    fisheye_fov: int = 180
    fisheye_mask_path: str = ""          # Path to DeoVR mask8k.png

    # SBS processing
    is_sbs: bool = False                 # Side-by-side stereo (per-eye)

    # POV mode
    pov_mode: bool = False               # Remove POV body + background

    # Frame range (1-based, inclusive). 0 = unset (use all).
    start_frame: int = 0
    end_frame: int = 0

    # Custom temp directory (empty = system default).
    temp_dir: str = ""


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


class Pipeline:
    """Orchestrates the full video matting pipeline.

    Args:
        config: Pipeline configuration.
        on_progress: Callback for progress updates (for live preview).
    """

    def __init__(self, config: PipelineConfig,
                 on_progress: Callable[[PipelineProgress], None] | None = None):
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
            progress.elapsed_sec = time.monotonic() - self._start_time
            self.on_progress(progress)

    def _extract_frames_with_progress(
        self,
        cmd: list[str],
        frames_dir: Path,
        expected: int,
        total_stages: int,
    ) -> None:
        """Run ffmpeg extraction while reporting frame progress.

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
                    fps=fps,
                    eta_sec=eta,
                ))

        if process.returncode != 0:
            raise RuntimeError(
                "ffmpeg frame extraction failed "
                f"(exit code {process.returncode})"
            )

        # Emit final progress
        count = len(os.listdir(frames_dir))
        self._emit(PipelineProgress(
            stage="Extracting frames",
            stage_num=1,
            total_stages=total_stages,
            frame_num=count,
            total_frames=expected,
        ))

    def run(self) -> Path:
        """Execute the full pipeline.

        Returns:
            Path to the final output file.

        Raises:
            RuntimeError: If ffmpeg is not available or pipeline fails.
            InterruptedError: If cancelled by user.
        """
        if not check_ffmpeg():
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg and ensure "
                "it is on your PATH."
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
            f"{info['width']}x{info['height']} @ {info['fps']}fps, "
            f"{info['num_frames']} frames"
        )

        tmp_kwargs = {"prefix": "vrautomatte_"}
        if config.temp_dir:
            tmp_base = Path(config.temp_dir)
            tmp_base.mkdir(parents=True, exist_ok=True)
            tmp_kwargs["dir"] = str(tmp_base)

        with tempfile.TemporaryDirectory(**tmp_kwargs) as tmpdir:
            tmp = Path(tmpdir)
            frames_dir = tmp / "frames"
            mattes_dir = tmp / "mattes"
            frames_dir.mkdir()
            mattes_dir.mkdir()

            # ── Pre-flight disk space check ──
            num_to_process = info["num_frames"]
            if config.start_frame > 0 or config.end_frame > 0:
                s = max(config.start_frame, 1)
                e = config.end_frame or info["num_frames"]
                num_to_process = min(
                    e, info["num_frames"]
                ) - s + 1
            estimated = self._estimate_disk_bytes(
                info["width"], info["height"],
                num_to_process,
                is_deovr=(
                    config.output_format
                    == OutputFormat.DEOVR_ALPHA
                ),
            )
            self._check_disk_space(tmp, estimated)
            est_gb = estimated / (1024 ** 3)
            logger.info(
                f"Estimated temp space: {est_gb:.1f} GB"
            )

            # ── Stage 1: Extract frames ──
            total_stages = self._total_stages()
            self._emit(PipelineProgress(
                stage="Extracting frames", stage_num=1,
                total_stages=total_stages,
                total_frames=num_to_process,
            ))
            logger.info("Stage 1: Extracting frames...")

            # Build ffmpeg extract command with optional range
            extract_cmd = ["ffmpeg", "-y", "-i", str(input_path)]
            has_range = (
                config.start_frame > 0 or config.end_frame > 0
            )
            if has_range:
                start = max(config.start_frame - 1, 0)
                end = (
                    config.end_frame
                    if config.end_frame > 0
                    else info["num_frames"]
                )
                end = min(end, info["num_frames"])
                count = end - start
                # Select frame range via video filter.
                # -frames:v stops ffmpeg after outputting all
                # selected frames instead of decoding the entire
                # video (critical for large files).
                extract_cmd += [
                    "-vf",
                    f"select='between(n\\,{start}\\,{end - 1})'",
                    "-vsync", "vfr",
                    "-frames:v", str(count),
                ]
                logger.info(
                    f"Frame range: {start + 1}–{end} "
                    f"({count} frames)"
                )
            extract_cmd.append(
                str(frames_dir / "frame_%06d.png")
            )

            # Run ffmpeg and track extraction progress
            self._extract_frames_with_progress(
                extract_cmd, frames_dir, num_to_process,
                total_stages,
            )

            frame_files = sorted(frames_dir.glob("*.png"))
            total_frames = len(frame_files)
            logger.info(f"Extracted {total_frames} frames")

            # ── Stage 2: Generate mattes ──
            logger.info("Stage 2: Generating AI mattes...")
            self._matte_start_time = time.monotonic()

            # Determine if we should use SBS per-eye
            use_sbs = config.is_sbs and detect_sbs(
                info["width"], info["height"]
            )

            if use_sbs:
                self._run_sbs_matte_pass(
                    config, frame_files, mattes_dir,
                    total_frames,
                )
            else:
                self._run_matte_pass(
                    config, frame_files, mattes_dir,
                    total_frames,
                )

            # ── Stage 3: Reassemble matte video ──
            logger.info("Stage 3: Assembling matte video...")
            self._emit(PipelineProgress(
                stage="Assembling matte video", stage_num=3,
                total_stages=self._total_stages(),
            ))
            matte_video = tmp / "matte.mp4"
            subprocess.run([
                "ffmpeg", "-y",
                "-framerate", info["fps_str"],
                "-i", str(mattes_dir / "frame_%06d.png"),
                "-c:v", "libx264", "-crf", str(config.crf),
                "-pix_fmt", "yuv420p",
                str(matte_video),
            ], capture_output=True, check=True)

            if config.output_format == OutputFormat.MATTE_ONLY:
                # Copy matte and merge original audio into output
                self._copy_with_audio(
                    matte_video, input_path, output_path
                )
                logger.info(f"Done! Matte saved to: {output_path}")
                self._emit(PipelineProgress(
                    stage="Complete",
                    stage_num=self._total_stages(),
                    total_stages=self._total_stages(),
                ))
                return output_path

            # ── Stage 4: Convert to fisheye ──
            if config.projection == ProjectionType.EQUIRECTANGULAR:
                logger.info("Stage 4: Converting to fisheye...")
                self._emit(PipelineProgress(
                    stage="Converting to fisheye", stage_num=4,
                    total_stages=self._total_stages(),
                ))
                fisheye_video = tmp / "fisheye_video.mp4"
                fisheye_matte = tmp / "fisheye_matte.mp4"

                convert_to_fisheye(
                    input_path, config.fisheye_mask_path,
                    fisheye_video, config.fisheye_fov,
                    config.codec, config.crf,
                )
                convert_to_fisheye(
                    matte_video, config.fisheye_mask_path,
                    fisheye_matte, config.fisheye_fov,
                    "libx264", config.crf,
                )
            else:
                fisheye_video = input_path
                fisheye_matte = matte_video

            # ── Stage 5: Convert matte to red channel ──
            logger.info("Stage 5: Converting matte to red channel...")
            self._emit(PipelineProgress(
                stage="Packing alpha channel", stage_num=5,
                total_stages=self._total_stages(),
            ))
            red_matte = tmp / "matte_red.mp4"
            matte_to_red_channel(fisheye_matte, red_matte, config.crf)

            # ── Stage 6: Pack alpha ──
            logger.info("Stage 6: Packing video + alpha...")
            pack_alpha(
                fisheye_video, red_matte, output_path,
                config.codec, config.crf,
            )

            logger.info(f"Done! Alpha-packed video: {output_path}")
            self._emit(PipelineProgress(
                stage="Complete",
                stage_num=self._total_stages(),
                total_stages=self._total_stages(),
            ))
            return output_path

    def _make_processor(self, config, frame_files):
        """Create a processor, handling mask gen if needed."""
        first_frame = None
        needs_first_frame = (
            config.model_variant == "matanyone2"
            or config.pov_mode
        )
        if needs_first_frame:
            self._emit(PipelineProgress(
                stage="Generating first-frame mask",
                stage_num=2,
                total_stages=self._total_stages(),
            ))
            first_img = Image.open(
                frame_files[0]
            ).convert("RGB")
            first_frame = np.array(first_img)

        return create_processor(
            variant=config.model_variant,
            downsample_ratio=config.downsample_ratio,
            first_frame=first_frame,
            pov_mode=config.pov_mode,
        )

    def _run_matte_pass(
        self, config, frame_files, mattes_dir,
        total_frames,
    ):
        """Run matting on all frames (non-SBS path)."""
        processor = self._make_processor(
            config, frame_files
        )

        for i, frame_file in enumerate(frame_files):
            if self._cancelled:
                raise InterruptedError(
                    "Pipeline cancelled by user"
                )

            # Periodic disk space check
            if i % _DISK_CHECK_INTERVAL == 0:
                self._check_disk_free(mattes_dir)

            frame_img = Image.open(
                frame_file
            ).convert("RGB")
            frame_arr = np.array(frame_img)
            matte_arr = processor.process_frame(frame_arr)

            matte_img = Image.fromarray(matte_arr, mode="L")
            matte_img.save(mattes_dir / frame_file.name)

            self._emit_matte_progress(
                i, total_frames, frame_arr, matte_arr
            )

        processor.cleanup()

    def _run_sbs_matte_pass(
        self, config, frame_files, mattes_dir,
        total_frames,
    ):
        """Run per-eye matting on SBS stereo frames.

        Splits each frame, mattes left eye first (all frames),
        then right eye (all frames), merges results.
        """
        logger.info("SBS mode: processing per-eye")
        # Read all frames and split
        left_frames = []
        right_frames = []
        for ff in frame_files:
            img = np.array(
                Image.open(ff).convert("RGB")
            )
            left, right = split_frame(img)
            left_frames.append(left)
            right_frames.append(right)

        # --- Left eye pass ---
        logger.info("SBS: matting left eye...")
        needs_first = (
            config.model_variant == "matanyone2"
            or config.pov_mode
        )
        left_first = left_frames[0] if needs_first else None
        proc_l = create_processor(
            variant=config.model_variant,
            downsample_ratio=config.downsample_ratio,
            first_frame=left_first,
            pov_mode=config.pov_mode,
        )

        left_mattes = []
        for i, lf in enumerate(left_frames):
            if self._cancelled:
                raise InterruptedError(
                    "Pipeline cancelled by user"
                )
            if i % _DISK_CHECK_INTERVAL == 0:
                self._check_disk_free(mattes_dir)
            left_mattes.append(proc_l.process_frame(lf))
            self._emit_matte_progress(
                i, total_frames * 2,
                merge_frames(lf, right_frames[i]),
                None,
                stage="Matting left eye",
            )
        proc_l.cleanup()

        # --- Right eye pass ---
        logger.info("SBS: matting right eye...")
        right_first = (
            right_frames[0] if needs_first else None
        )
        proc_r = create_processor(
            variant=config.model_variant,
            downsample_ratio=config.downsample_ratio,
            first_frame=right_first,
            pov_mode=config.pov_mode,
        )

        right_mattes = []
        for i, rf in enumerate(right_frames):
            if self._cancelled:
                raise InterruptedError(
                    "Pipeline cancelled by user"
                )
            if i % _DISK_CHECK_INTERVAL == 0:
                self._check_disk_free(mattes_dir)
            right_mattes.append(proc_r.process_frame(rf))
            self._emit_matte_progress(
                total_frames + i, total_frames * 2,
                merge_frames(left_frames[i], rf),
                None,
                stage="Matting right eye",
            )
        proc_r.cleanup()

        # --- Merge and save ---
        logger.info("SBS: merging per-eye mattes...")
        for i, ff in enumerate(frame_files):
            merged = merge_mattes(
                left_mattes[i], right_mattes[i]
            )
            Image.fromarray(
                merged, mode="L"
            ).save(mattes_dir / ff.name)

    def _emit_matte_progress(
        self, i, total, source, matte,
        stage="Generating mattes",
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
        ))

    def _copy_with_audio(self, video_path: Path,
                         audio_source: Path, output_path: Path) -> None:
        """Copy video and mux audio from the original source.

        If the original has no audio track, just copies the video.
        """
        try:
            # Try muxing audio from original
            subprocess.run([
                "ffmpeg", "-y",
                "-i", str(video_path),
                "-i", str(audio_source),
                "-c:v", "copy", "-c:a", "aac",
                "-map", "0:v:0", "-map", "1:a:0?",
                "-shortest",
                str(output_path),
            ], capture_output=True, check=True)
        except subprocess.CalledProcessError:
            # No audio track — just copy video
            logger.debug("No audio track found, copying video only")
            shutil.copy2(video_path, output_path)

    def _total_stages(self) -> int:
        """Calculate total stages based on config."""
        if self.config.output_format == OutputFormat.MATTE_ONLY:
            return 3  # extract, matte, assemble
        return 6  # + fisheye convert, red channel, pack

    @staticmethod
    def _estimate_disk_bytes(
        width: int, height: int, num_frames: int,
        is_deovr: bool = False,
    ) -> int:
        """Estimate required temp disk space in bytes.

        PNG frames ≈ width × height × 3 × 0.5 (compression).
        We store source frames + matte frames + intermediate
        videos, so roughly 3× the frame data.

        Args:
            width: Video width.
            height: Video height.
            num_frames: Number of frames to process.
            is_deovr: DeoVR pipeline creates extra intermediates.

        Returns:
            Estimated bytes needed.
        """
        bytes_per_frame = int(width * height * 3 * 0.5)
        # source PNGs + matte PNGs
        frame_bytes = bytes_per_frame * num_frames * 2
        # intermediate video files (~10% of uncompressed)
        video_overhead = int(
            width * height * 3 * num_frames * 0.1
        )
        multiplier = 2 if is_deovr else 1
        return frame_bytes + (video_overhead * multiplier)

    @staticmethod
    def _check_disk_space(path: Path, required: int) -> None:
        """Raise if the drive has less than required + safety margin.

        Args:
            path: Any path on the target drive.
            required: Estimated bytes needed for the operation.

        Raises:
            RuntimeError: If insufficient disk space.
        """
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
    def _check_disk_free(path: Path) -> None:
        """Raise if free space drops below the safety margin.

        Called periodically during processing to prevent
        filling the drive.

        Args:
            path: Any path on the target drive.

        Raises:
            RuntimeError: If free space is critically low.
        """
        free = shutil.disk_usage(path).free
        if free < _MIN_FREE_BYTES:
            free_mb = free / (1024 ** 2)
            raise RuntimeError(
                f"Disk space critically low ({free_mb:.0f} MB "
                f"remaining). Processing stopped to prevent "
                f"filling the drive. Free up space and retry."
            )
