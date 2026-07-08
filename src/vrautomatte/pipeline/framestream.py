"""Raw-video streaming between ffmpeg and the matting loop.

Replaces the PNG round-trip for per-frame matting models:

    ffmpeg (decode + scale) --rgb24 pipe--> matting --gray pipe-->
    ffmpeg (segment encode)

No frame PNGs are written to disk, no PNG codec work happens at
all, and decode runs ahead of matting via a bounded frame queue.

Windows pipe safety: the historical deadlocks in this project came
from ffmpeg's *stderr* progress stream (``\\r``-terminated,
unbounded) filling a 64 KB pipe buffer. Here stderr is DEVNULL and
stdout carries fixed-size binary frames read with ``readinto`` —
the "block until exactly n bytes" behaviour of Windows pipe reads
is exactly what a fixed-size frame reader wants.
"""

import queue
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from loguru import logger

from vrautomatte.utils.ffmpeg import _hwaccel_args, stream_encode_args


class FrameStreamReader:
    """Stream RGB frames from an ffmpeg rawvideo pipe.

    A background thread reads fixed-size frames from ffmpeg's
    stdout into a bounded queue, so video decode runs ahead of
    (and overlaps with) GPU matting.

    Args:
        input_path: Source video file.
        out_size: (width, height) of the delivered frames. When
            ``scale`` is True, ffmpeg downscales to this size;
            otherwise it must equal the native video size.
        start_ts: Seek position in seconds (keyframe seek).
        num_frames: Number of frames to deliver.
        scale: Apply an ffmpeg lanczos scale filter to out_size.
        queue_depth: Frames buffered ahead of the consumer.
    """

    def __init__(
        self,
        input_path: Path,
        out_size: tuple[int, int],
        *,
        start_ts: float = 0.0,
        num_frames: int = 0,
        scale: bool = False,
        queue_depth: int = 8,
    ):
        self._w, self._h = out_size
        self._frame_bytes = self._w * self._h * 3
        self._queue: queue.Queue = queue.Queue(maxsize=queue_depth)
        self._closed = False

        cmd = [
            "ffmpeg", "-nostdin", "-v", "error",
            *_hwaccel_args(),
            "-ss", f"{start_ts:.6f}",
            "-i", str(input_path),
            "-frames:v", str(num_frames),
        ]
        if scale:
            cmd += [
                "-vf",
                f"scale={self._w}:{self._h}:flags=lanczos",
            ]
        cmd += ["-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]

        logger.debug(
            f"FrameStreamReader: {self._w}x{self._h}, "
            f"ss={start_ts:.2f}s, n={num_frames}"
        )
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._thread = threading.Thread(
            target=self._read_loop, daemon=True
        )
        self._thread.start()

    def _read_loop(self) -> None:
        stdout = self._proc.stdout
        while not self._closed:
            buf = bytearray(self._frame_bytes)
            view = memoryview(buf)
            got = 0
            while got < self._frame_bytes:
                n = stdout.readinto(view[got:])
                if not n:
                    break
                got += n
            if got < self._frame_bytes:
                break  # EOF (or process killed)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(
                self._h, self._w, 3
            )
            while not self._closed:
                try:
                    self._queue.put(arr, timeout=0.25)
                    break
                except queue.Full:
                    continue
        # Sentinel — consumer sees end-of-stream.
        while not self._closed:
            try:
                self._queue.put(None, timeout=0.25)
                break
            except queue.Full:
                continue

    def read(self) -> np.ndarray | None:
        """Return the next frame, or None at end of stream."""
        item = self._queue.get()
        return item

    def close(self) -> None:
        """Stop the reader and terminate ffmpeg."""
        self._closed = True
        try:
            self._proc.kill()
        except OSError:
            pass
        # Unblock the reader thread if it's waiting on a full
        # queue, then join.
        while self._thread.is_alive():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                time.sleep(0.05)
        self._proc.wait()


class SegmentStreamWriter:
    """Encode grayscale mattes into a segment video via a pipe.

    Mattes are written at matting resolution — upscaling to the
    original resolution happens once at final assembly, not per
    segment. The encoder is chosen up-front (NVENC probe) because
    a pipe feed cannot be retried after a mid-stream failure.

    Args:
        segment_path: Output .mp4 path.
        size: (width, height) of the incoming mattes.
        fps_str: Framerate string (e.g. "60000/1001").
        crf: Encode quality.
    """

    def __init__(
        self,
        segment_path: Path,
        size: tuple[int, int],
        fps_str: str,
        crf: int,
    ):
        self._path = Path(segment_path)
        w, h = size
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{w}x{h}",
            "-framerate", fps_str,
            "-i", "pipe:0",
            *stream_encode_args(crf),
            "-pix_fmt", "yuv420p",
            str(self._path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.frames_written = 0
        # Feeder thread: serializing + piping a 4 MB matte
        # blocks the matting loop for milliseconds per frame
        # (16% of it in a live profile) and encoder finalize
        # stalls it for seconds. The bounded queue absorbs
        # bursts and overlaps pipe I/O with matting.
        self._q: queue.Queue = queue.Queue(maxsize=8)
        self._feed_error: BaseException | None = None
        self._feeder = threading.Thread(
            target=self._feed_loop,
            name="segment-feeder",
            daemon=True,
        )
        self._feeder.start()

    def _feed_loop(self) -> None:
        while True:
            matte = self._q.get()
            if matte is None:
                return
            try:
                self._proc.stdin.write(matte.tobytes())
            except BaseException as exc:  # noqa: BLE001
                self._feed_error = exc
                return

    def write(self, matte: np.ndarray) -> None:
        """Queue one grayscale matte frame for encoding."""
        if self._feed_error is not None:
            raise RuntimeError(
                f"Segment encoder feed failed for "
                f"{self._path.name}"
            ) from self._feed_error
        self._q.put(matte)
        self.frames_written += 1

    def close(self) -> None:
        """Finish the segment and verify the encode succeeded."""
        self._q.put(None)
        self._feeder.join()
        try:
            self._proc.stdin.close()
        except OSError:
            pass
        ret = self._proc.wait()
        if self._feed_error is not None:
            raise RuntimeError(
                f"Segment encoder feed failed for "
                f"{self._path.name}"
            ) from self._feed_error
        if ret != 0:
            raise RuntimeError(
                f"Segment encoder failed (exit {ret}) for "
                f"{self._path.name}"
            )
        logger.debug(
            f"Segment closed: {self._path.name} "
            f"({self.frames_written} frames)"
        )

    def abort(self) -> None:
        """Kill the encoder and remove the partial segment."""
        self._feed_error = self._feed_error or RuntimeError(
            "aborted"
        )
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        try:
            self._proc.kill()
            self._proc.wait()
        except OSError:
            pass
        self._path.unlink(missing_ok=True)
