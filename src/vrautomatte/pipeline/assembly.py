"""Chaptered background DeoVR assembly.

Assembly and matting use different hardware: matting saturates
the CUDA cores while assembly runs on NVDEC (decode), the CPU
(filter graph) and NVENC (encode). So instead of assembling the
whole video after matting finishes, completed matte segments are
assembled into final-encoded *chapter* files by a background
worker WHILE matting continues. At the end, the chapters are
concatenated with stream copy (instant) and the audio is muxed
in — assembly wall time hides almost entirely behind matting.

Chapters are cut on segment boundaries, so the frame accounting
stays exact even when a chunk delivered a frame more or less
(keyframe-seek imprecision in the file-based path): each
chapter's true frame count is probed from its matte concat and
the running offset accumulates from that.

Resume: finished chapter parts are written atomically
(tmp + rename) into the deterministic temp dir, so a resumed job
reuses them and only assembles what's missing.
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from loguru import logger

from vrautomatte.utils.ffmpeg import (
    _run_ffmpeg_logged,
    assemble_deovr,
    get_video_info,
)

# Target chapter length in frames (rounded to whole segments).
DEFAULT_CHAPTER_FRAMES = 6000


class ChapteredAssembler:
    """Assembles DeoVR chapters in the background during matting.

    Usage:
        1. Construct before matting starts.
        2. Call ``maybe_submit(completed_segments)`` after each
           segment flush (cheap; submits full chapters only).
        3. Call ``finalize(...)`` after matting — assembles any
           remaining segments, waits for the worker, concats the
           chapters and muxes audio into the final output.

    All heavy work runs on ONE worker thread so at most a single
    ffmpeg assembly job competes with matting at any time.
    """

    def __init__(
        self,
        source_path: Path,
        segments_dir: Path,
        work_dir: Path,
        *,
        is_equirect: bool,
        fov: int,
        mask_path,
        fps: float,
        start_frame_0based: int,
        crf: int,
        preset: str | None = None,
        chunk_size: int = 500,
        chapter_frames: int = DEFAULT_CHAPTER_FRAMES,
        cancel_check=None,
    ):
        self._source = Path(source_path)
        self._segments_dir = Path(segments_dir)
        self._work = Path(work_dir)
        self._parts_dir = self._work / "parts"
        self._work.mkdir(parents=True, exist_ok=True)
        self._parts_dir.mkdir(exist_ok=True)

        self._is_equirect = is_equirect
        self._fov = fov
        self._mask_path = mask_path
        self._fps = fps
        self._start0 = start_frame_0based
        self._crf = crf
        self._preset = preset
        self._cancel_check = cancel_check or (lambda: False)

        self._chapter_segments = max(
            1, round(chapter_frames / max(chunk_size, 1))
        )
        self._pool = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="assembly"
        )
        self._futures = []
        self._chapters_submitted = 0
        self._next_seg = 0
        # Frames covered by chapters processed so far — only
        # touched by the single worker thread (jobs run in
        # submission order).
        self._cum_frames = 0

        logger.info(
            f"Background assembly: chapters of "
            f"{self._chapter_segments} segment(s) "
            f"(~{self._chapter_segments * chunk_size} frames)"
        )

    # ── submission ──────────────────────────────────────────

    def maybe_submit(self, completed_segments: int) -> None:
        """Submit chapters fully covered by flushed segments."""
        while (
            self._next_seg + self._chapter_segments
            <= completed_segments
        ):
            self._submit(self._chapter_segments)

    def _submit(self, n_segments: int) -> None:
        idx = self._chapters_submitted
        first = self._next_seg
        segs = [
            self._segments_dir / f"segment_{i:06d}.mp4"
            for i in range(first, first + n_segments)
        ]
        self._futures.append(
            self._pool.submit(
                self._assemble_chapter, idx, segs
            )
        )
        self._chapters_submitted += 1
        self._next_seg += n_segments

    # ── worker ──────────────────────────────────────────────

    def _concat_mattes(
        self, idx: int, seg_paths: list
    ) -> Path:
        """Stream-copy the chapter's matte segments into one file."""
        matte_ch = self._work / f"matte_ch_{idx:06d}.mp4"
        concat_list = self._work / f"matte_ch_{idx:06d}.txt"
        with concat_list.open("w") as f:
            for seg in seg_paths:
                safe = str(seg).replace("\\", "/")
                f.write(f"file '{safe}'\n")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c", "copy",
                str(matte_ch),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        concat_list.unlink(missing_ok=True)
        return matte_ch

    def _assemble_chapter(
        self, idx: int, seg_paths: list
    ) -> None:
        if self._cancel_check():
            raise InterruptedError("Assembly cancelled")

        part = self._parts_dir / f"part_{idx:06d}.mp4"

        if part.exists():
            # Left over from a resumed run — trust it (written
            # via atomic rename) and just advance the offset.
            n = get_video_info(part)["num_frames"]
            self._cum_frames += n
            logger.info(
                f"Chapter {idx}: reusing existing part "
                f"({n} frames)"
            )
            return

        matte_ch = self._concat_mattes(idx, seg_paths)
        try:
            n = get_video_info(matte_ch)["num_frames"]
            ss = (
                (self._start0 + self._cum_frames) / self._fps
                if self._fps > 0 else 0.0
            )
            dur = n / self._fps if self._fps > 0 else None

            tmp_part = part.with_name(part.name + ".tmp.mp4")
            t0 = time.monotonic()
            assemble_deovr(
                self._source, matte_ch, tmp_part,
                is_equirect=self._is_equirect,
                fov=self._fov,
                mask_path=self._mask_path,
                ss_sec=ss,
                dur_sec=dur,
                crf=self._crf,
                preset=self._preset,
                total_frames=n,
                include_audio=False,
                cancel_check=self._cancel_check,
                label=f"chapter-{idx:03d}",
            )
            tmp_part.rename(part)
            self._cum_frames += n
            logger.info(
                f"Chapter {idx} assembled: {n} frames in "
                f"{time.monotonic() - t0:.0f}s (background)"
            )
        finally:
            matte_ch.unlink(missing_ok=True)

    # ── finalization ────────────────────────────────────────

    def finalize(
        self,
        total_segments: int,
        output_path: Path,
        total_frames: int = 0,
    ) -> None:
        """Assemble the tail, wait, concat chapters + mux audio."""
        if self._next_seg < total_segments:
            self._submit(total_segments - self._next_seg)

        # Wait for the worker; propagate the first failure.
        for fut in self._futures:
            fut.result()
        self._pool.shutdown(wait=True)

        parts = sorted(self._parts_dir.glob("part_*.mp4"))
        if not parts:
            raise RuntimeError(
                "No assembled chapters found — assembly "
                "stage failed."
            )

        concat_list = self._work / "parts.txt"
        with concat_list.open("w") as f:
            for p in parts:
                safe = str(p).replace("\\", "/")
                f.write(f"file '{safe}'\n")

        ss0 = (
            self._start0 / self._fps if self._fps > 0 else 0
        )
        # -shortest ends the mux at the video's end (the video
        # length is exact from the chapter concat); the audio
        # from the source is seeked to the processed range and
        # trimmed by it. Frame-exact, no rounding risk.
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(concat_list),
            *(["-ss", f"{ss0:.4f}"] if ss0 > 0 else []),
            "-i", str(self._source),
            "-map", "0:v:0", "-c:v", "copy",
            "-map", "1:a:0?", "-c:a", "copy",
            "-shortest",
            str(output_path),
        ]
        _run_ffmpeg_logged(
            cmd, "final-mux", total_frames=total_frames,
            cancel_check=self._cancel_check,
        )
        concat_list.unlink(missing_ok=True)
        logger.info(
            f"Final output: {len(parts)} chapters, "
            f"{self._cum_frames} frames -> {output_path}"
        )

    def abort(self) -> None:
        """Stop the worker (cancel flag stops in-flight ffmpeg)."""
        self._pool.shutdown(wait=True)