"""Pipeline orchestrator — chains all steps from input to output.

Steps:
1. Extract frames from input video
2. Generate AI alpha matte for each frame (via RVM)
3. Reassemble matte frames into a video
4. (Optional) Convert equirectangular → fisheye
5. (Optional) Pack alpha channel for DeoVR format
"""

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
    is_sbs: bool = True                  # Side-by-side stereo


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

        with tempfile.TemporaryDirectory(prefix="vrautomatte_") as tmpdir:
            tmp = Path(tmpdir)
            frames_dir = tmp / "frames"
            mattes_dir = tmp / "mattes"
            frames_dir.mkdir()
            mattes_dir.mkdir()

            # ── Stage 1: Extract frames ──
            self._emit(PipelineProgress(
                stage="Extracting frames", stage_num=1,
                total_stages=self._total_stages(),
            ))
            logger.info("Stage 1: Extracting frames...")
            subprocess.run([
                "ffmpeg", "-y", "-i", str(input_path),
                str(frames_dir / "frame_%06d.png"),
            ], capture_output=True, check=True)

            frame_files = sorted(frames_dir.glob("*.png"))
            total_frames = len(frame_files)
            logger.info(f"Extracted {total_frames} frames")

            # ── Stage 2: Generate mattes ──
            logger.info("Stage 2: Generating AI mattes...")
            self._matte_start_time = time.monotonic()

            # For MatAnyone 2, load the first frame for SAM2 mask
            first_frame = None
            if config.model_variant == "matanyone2":
                self._emit(PipelineProgress(
                    stage="Generating first-frame mask",
                    stage_num=2,
                    total_stages=self._total_stages(),
                ))
                first_img = Image.open(
                    frame_files[0]
                ).convert("RGB")
                first_frame = np.array(first_img)

            processor = create_processor(
                variant=config.model_variant,
                downsample_ratio=config.downsample_ratio,
                first_frame=first_frame,
            )

            for i, frame_file in enumerate(frame_files):
                if self._cancelled:
                    raise InterruptedError("Pipeline cancelled by user")

                frame_img = Image.open(frame_file).convert("RGB")
                frame_arr = np.array(frame_img)

                matte_arr = processor.process_frame(frame_arr)

                # Save matte
                matte_img = Image.fromarray(matte_arr, mode="L")
                matte_img.save(mattes_dir / frame_file.name)

                # Emit progress with preview and ETA
                if i % 10 == 0 or i == total_frames - 1:
                    elapsed = time.monotonic() - self._matte_start_time
                    fps = (i + 1) / elapsed if elapsed > 0 else 0
                    remaining = total_frames - (i + 1)
                    eta = remaining / fps if fps > 0 else 0

                    self._emit(PipelineProgress(
                        stage="Generating mattes",
                        stage_num=2,
                        total_stages=self._total_stages(),
                        frame_num=i + 1,
                        total_frames=total_frames,
                        source_frame=frame_arr,
                        matte_frame=matte_arr,
                        eta_sec=eta,
                        fps=fps,
                    ))

            processor.cleanup()  # free GPU memory

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
