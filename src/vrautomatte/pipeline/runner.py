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

    # ── MatAnyone 2 performance settings ──────────────────────────
    # These only apply when model_variant == "matanyone2".

    # Run model in FP16. Halves VRAM usage on CUDA. Default True.
    use_fp16: bool = True

    # Memory-bank encoding resolution (px, short side).
    # Reduces working-memory VRAM by ~95% at 480 vs full 4K.
    # -1 = full resolution. Default 480.
    ma2_internal_size: int = 480

    # Working-memory frame slots before long-term consolidation.
    # Lower = less VRAM, marginally less temporal smoothing.
    ma2_mem_frames: int = 3

    # XMem long-term memory potentiation. Required for videos
    # longer than a few minutes — prevents unbounded VRAM growth.
    ma2_use_long_term: bool = True

    # torch.compile with CUDA-graph-replay. Opt-in: adds ~30 s
    # first-frame compilation cost, saves ~15-30% per frame after.
    ma2_compile_model: bool = False

    # ── Disk management ───────────────────────────────────────────
    # Matte frames to accumulate before encoding a segment video and
    # deleting the PNGs. Caps the matte-PNG pool at chunk_size × PNG.
    # At 8K (5800×2900), each matte PNG ≈ 8 MB → 500 ≈ 4 GB peak.
    # Source PNGs are deleted immediately after processing regardless.
    chunk_size: int = 500


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
            segments_dir = tmp / "segments"
            frames_dir.mkdir()
            mattes_dir.mkdir()
            segments_dir.mkdir()

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
                chunk_size=config.chunk_size,
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
                    segments_dir, total_frames, info["fps_str"],
                )
            else:
                self._run_matte_pass(
                    config, frame_files, mattes_dir,
                    segments_dir, total_frames, info["fps_str"],
                )

            # ── Stage 3: Concatenate matte segments ──
            # Segment videos were encoded (and matte PNGs deleted)
            # during Stage 2. Source PNGs were also deleted per-frame.
            # This concat is fast — no re-encode, stream copy only.
            logger.info("Stage 3: Concatenating matte segments...")
            self._emit(PipelineProgress(
                stage="Assembling matte video", stage_num=3,
                total_stages=self._total_stages(),
            ))
            matte_video = tmp / "matte.mp4"
            self._concat_matte_segments(
                segments_dir, matte_video, info["fps_str"],
                config.crf,
            )

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

    def _make_processor(self, config, first_frame: np.ndarray | None):
        """Create a matting processor from config.

        Args:
            config: Pipeline configuration.
            first_frame: First video frame as uint8 RGB array.
                Required for matanyone2 and pov_mode; None otherwise.
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

        return create_processor(
            variant=config.model_variant,
            downsample_ratio=config.downsample_ratio,
            first_frame=first_frame if needs_first_frame else None,
            pov_mode=config.pov_mode,
            use_fp16=config.use_fp16,
            max_internal_size=config.ma2_internal_size,
            max_mem_frames=config.ma2_mem_frames,
            use_long_term=config.ma2_use_long_term,
            compile_model=config.ma2_compile_model,
        )

    def _run_matte_pass(
        self, config, frame_files, mattes_dir,
        segments_dir, total_frames, fps_str,
    ):
        """Run matting on all frames (non-SBS path).

        Source PNGs are deleted immediately after each frame is
        processed.  Matte PNGs are flushed to a segment video every
        ``config.chunk_size`` frames, then deleted — keeping the
        matte-PNG pool bounded regardless of video length.
        """
        first_frame = np.array(
            Image.open(frame_files[0]).convert("RGB")
        )
        processor = self._make_processor(config, first_frame)

        seg_idx = 0          # segment video counter
        seg_frame = 0        # frames written into current segment

        for i, frame_file in enumerate(frame_files):
            if self._cancelled:
                processor.cleanup()
                raise InterruptedError(
                    "Pipeline cancelled by user"
                )

            if i % _DISK_CHECK_INTERVAL == 0:
                self._check_disk_free(mattes_dir)

            frame_arr = np.array(
                Image.open(frame_file).convert("RGB")
            )
            matte_arr = processor.process_frame(frame_arr)

            # Save matte PNG with a local (per-segment) index so
            # ffmpeg's image2 demuxer sees a gapless sequence.
            seg_frame += 1
            Image.fromarray(matte_arr, mode="L").save(
                mattes_dir / f"frame_{seg_frame:06d}.png"
            )

            # Delete source PNG immediately — no longer needed.
            try:
                frame_file.unlink()
            except OSError:
                pass

            self._emit_matte_progress(
                i, total_frames, frame_arr, matte_arr
            )

            # Flush segment when chunk is full or on the last frame.
            if seg_frame >= config.chunk_size or i == total_frames - 1:
                self._flush_matte_segment(
                    mattes_dir, segments_dir, seg_idx,
                    fps_str=fps_str,
                    crf=config.crf,
                )
                seg_idx += 1
                seg_frame = 0

        processor.cleanup()

    def _run_sbs_matte_pass(
        self, config, frame_files, mattes_dir,
        segments_dir, total_frames, fps_str,
    ):
        """Run per-eye matting on SBS stereo frames.

        Processes both eyes together, one frame at a time:

            for each frame:
                split → left / right
                left_matte  = proc_l.process_frame(left)
                right_matte = proc_r.process_frame(right)
                merged = merge_mattes(left_matte, right_matte)
                save merged PNG → mattes_dir
                delete source PNG immediately
                discard all intermediate arrays

        Every ``config.chunk_size`` merged mattes, encode a segment
        video and delete the matte PNGs — keeping the pool bounded.

        Both InferenceCore instances live on the same GPU and take
        turns — no GPU contention because process_frame() is called
        sequentially, not concurrently.
        """
        logger.info("SBS mode: processing per-eye (interleaved)")

        needs_first = (
            config.model_variant == "matanyone2"
            or config.pov_mode
        )

        # Load only the first frame to seed both processors.
        first_full = np.array(
            Image.open(frame_files[0]).convert("RGB")
        )
        left_first, right_first = split_frame(first_full)
        del first_full

        logger.info("SBS: initialising left-eye processor...")
        proc_l = self._make_processor(
            config, left_first if needs_first else None
        )
        logger.info("SBS: initialising right-eye processor...")
        proc_r = self._make_processor(
            config, right_first if needs_first else None
        )
        del left_first, right_first

        seg_idx = 0
        seg_frame = 0

        # ── Per-frame interleaved loop ───────────────────────────
        for i, frame_file in enumerate(frame_files):
            if self._cancelled:
                proc_l.cleanup()
                proc_r.cleanup()
                raise InterruptedError(
                    "Pipeline cancelled by user"
                )

            if i % _DISK_CHECK_INTERVAL == 0:
                self._check_disk_free(mattes_dir)

            full = np.array(
                Image.open(frame_file).convert("RGB")
            )
            left, right = split_frame(full)

            left_matte  = proc_l.process_frame(left)
            right_matte = proc_r.process_frame(right)
            merged = merge_mattes(left_matte, right_matte)

            seg_frame += 1
            Image.fromarray(merged, mode="L").save(
                mattes_dir / f"frame_{seg_frame:06d}.png"
            )

            # Delete source PNG immediately.
            try:
                frame_file.unlink()
            except OSError:
                pass

            self._emit_matte_progress(
                i, total_frames, full, merged,
                stage="Matting SBS (L+R)",
            )

            del full, left, right, left_matte, right_matte, merged

            if seg_frame >= config.chunk_size or i == total_frames - 1:
                self._flush_matte_segment(
                    mattes_dir, segments_dir, seg_idx,
                    fps_str=fps_str,
                    crf=config.crf,
                )
                seg_idx += 1
                seg_frame = 0

        proc_l.cleanup()
        proc_r.cleanup()

    def _flush_matte_segment(
        self,
        mattes_dir: Path,
        segments_dir: Path,
        seg_idx: int,
        fps_str: str,
        crf: int,
    ) -> None:
        """Encode all PNGs in mattes_dir into a segment video, then delete them.

        Uses libx264 with yuv420p for broad compatibility with the
        concat step.  The encode is fast because the input frames are
        already at matte resolution (grayscale, single channel).

        Args:
            mattes_dir: Directory containing frame_%06d.png files.
            segments_dir: Where to write the segment .mp4 file.
            seg_idx: Segment number (used for the output filename).
            fps_str: Frame-rate string (e.g. "60000/1001") from ffprobe.
            crf: Quality setting for the segment encode.
        """
        segment_path = segments_dir / f"segment_{seg_idx:06d}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-framerate", fps_str,
            "-i", str(mattes_dir / "frame_%06d.png"),
            "-c:v", "libx264", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(segment_path),
        ], check=True, stdin=subprocess.DEVNULL,
           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Delete all PNGs — segment video is the canonical record now.
        for png in mattes_dir.glob("frame_*.png"):
            try:
                png.unlink()
            except OSError:
                pass

        logger.debug(
            f"Flushed segment {seg_idx} → {segment_path.name}"
        )

    def _concat_matte_segments(
        self,
        segments_dir: Path,
        output_path: Path,
        fps_str: str,
        crf: int,
    ) -> None:
        """Concatenate all segment videos into a single matte video.

        Uses the ffmpeg concat demuxer with stream-copy — no re-encode,
        so this step is fast regardless of total frame count.

        Falls back to a single-segment copy if only one segment exists.

        Args:
            segments_dir: Directory containing segment_NNNNNN.mp4 files.
            output_path: Destination matte video path.
            fps_str: Frame-rate string (passed through for single-file case).
            crf: Unused here but kept for signature consistency.
        """
        segments = sorted(segments_dir.glob("segment_*.mp4"))
        if not segments:
            raise RuntimeError(
                "No matte segments found — matting stage may have failed."
            )

        if len(segments) == 1:
            # Single segment: just rename/copy.
            shutil.copy2(segments[0], output_path)
            return

        # Write a concat list file that ffmpeg can read.
        concat_list = segments_dir / "concat_list.txt"
        with concat_list.open("w") as f:
            for seg in segments:
                # ffmpeg requires forward slashes and escaped single quotes.
                safe = str(seg).replace("\\", "/").replace("'", "\'")
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
            f"Concatenated {len(segments)} segments → {output_path.name}"
        )

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
        chunk_size: int = 500,
    ) -> int:
        """Estimate peak temp disk space in bytes under the chunked pipeline.

        Storage model:
        - Source PNGs: all extracted upfront, deleted one-by-one during
          the matte pass.  Peak = all num_frames × source_png_size.
        - Matte PNGs: at most chunk_size live on disk at any time before
          they are flushed to a segment video and deleted.
        - Segment videos: all segments accumulate until the concat step.
          Encoded H.264 at CRF 18 ≈ 5–8 % of raw uncompressed size.
        - DeoVR pipeline adds an extra fisheye video + red-channel matte.

        Args:
            width: Video width in pixels.
            height: Video height in pixels.
            num_frames: Total frames to process.
            is_deovr: True when producing DeoVR alpha-packed output.
            chunk_size: Matte frames per segment (from PipelineConfig).

        Returns:
            Estimated peak bytes needed.
        """
        # Source PNGs (RGB, compressed ≈ 50 % of raw).
        source_png_per_frame = int(width * height * 3 * 0.5)
        all_source_pngs = source_png_per_frame * num_frames

        # Matte PNGs (grayscale, compressed ≈ 50 % of raw).
        matte_png_per_frame = int(width * height * 1 * 0.5)
        peak_matte_pngs = matte_png_per_frame * chunk_size

        # Segment videos: all N segments exist simultaneously until
        # concat finishes.  H.264 CRF-18 ≈ 6 % of raw uncompressed.
        raw_matte_per_frame = width * height  # 1 byte grayscale
        all_segments = int(raw_matte_per_frame * num_frames * 0.06)

        # DeoVR adds a fisheye video + red-channel matte video.
        deovr_overhead = (
            int(width * height * 3 * num_frames * 0.06) * 2
            if is_deovr else 0
        )

        return (
            all_source_pngs
            + peak_matte_pngs
            + all_segments
            + deovr_overhead
        )

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
