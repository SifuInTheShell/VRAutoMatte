"""FFmpeg wrapper utilities for video processing."""

import functools
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from loguru import logger


# ── NVENC detection ──────────────────────────────────────────

@functools.lru_cache(maxsize=1)
def _detect_nvenc() -> dict:
    """Detect available NVENC encoders.

    Returns dict with keys ``hevc``, ``h264``, ``av1``,
    each True/False.  Cached after first call.
    """
    result = {"hevc": False, "h264": False, "av1": False}
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        text = out.stdout
        if "hevc_nvenc" in text:
            result["hevc"] = True
        if "h264_nvenc" in text:
            result["h264"] = True
        if "av1_nvenc" in text:
            result["av1"] = True
    except Exception:
        pass
    avail = [k for k, v in result.items() if v]
    if avail:
        logger.info(f"NVENC available: {', '.join(avail)}")
    else:
        logger.info("NVENC not available, using CPU encoding")
    return result


def _encode_args(codec: str, crf: int) -> list:
    """Return ffmpeg encoding args, preferring NVENC when available.

    Falls back to CPU ``libx265`` / ``libx264`` if NVENC is not
    detected.
    """
    nvenc = _detect_nvenc()

    if codec == "libsvtav1" and nvenc["av1"]:
        return [
            "-c:v", "av1_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", str(crf + 5),
            "-b:v", "0",
        ]
    if codec == "libx265" and nvenc["hevc"]:
        return [
            "-c:v", "hevc_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", str(crf + 3),
            "-b:v", "0",
        ]
    if codec == "libx264" and nvenc["h264"]:
        return [
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-rc", "vbr",
            "-cq", str(crf + 2),
            "-b:v", "0",
        ]
    # CPU fallback
    return ["-c:v", codec, "-crf", str(crf)]


def _encode_args_cpu(codec: str, crf: int) -> list:
    """Return CPU-only encoding args (no NVENC)."""
    return ["-c:v", codec, "-crf", str(crf)]


def _hwaccel_args() -> list:
    """Return hardware-accelerated decode args if NVENC is present."""
    nvenc = _detect_nvenc()
    if nvenc["hevc"] or nvenc["h264"]:
        return ["-hwaccel", "auto"]
    return []


def _run_ffmpeg_logged(
    cmd: list,
    label: str,
    total_frames: int = 0,
) -> None:
    """Run an ffmpeg command with progress logging.

    Uses ``-progress pipe:1`` so ffmpeg writes structured
    ``key=value\\n`` progress to stdout (safe on Windows —
    unlike stderr which uses ``\\r``).  Logs frame count
    every ~5 seconds.

    Args:
        cmd: The ffmpeg command list (will be modified in place
            to add ``-progress pipe:1``).
        label: Human-readable label for log messages.
        total_frames: If known, used for percentage display.
    """
    # Inject -progress before the output path (last arg)
    run_cmd = cmd[:-1] + ["-progress", "pipe:1"] + cmd[-1:]

    logger.info(f"[{label}] starting ffmpeg")
    logger.debug(f"[{label}] cmd: {' '.join(str(c) for c in run_cmd)}")
    t0 = time.monotonic()

    proc = subprocess.Popen(
        run_cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    last_log = t0
    last_frame = 0
    try:
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("frame="):
                try:
                    last_frame = int(line.split("=", 1)[1])
                except ValueError:
                    pass
            if line == "progress=continue" or line == "progress=end":
                now = time.monotonic()
                if now - last_log >= 5.0 or line == "progress=end":
                    elapsed = now - t0
                    if total_frames > 0 and last_frame > 0:
                        pct = last_frame / total_frames * 100
                        logger.info(
                            f"[{label}] frame {last_frame}"
                            f"/{total_frames} "
                            f"({pct:.0f}%) "
                            f"elapsed {elapsed:.0f}s"
                        )
                    elif last_frame > 0:
                        logger.info(
                            f"[{label}] frame {last_frame} "
                            f"elapsed {elapsed:.0f}s"
                        )
                    last_log = now
    finally:
        proc.wait()

    elapsed = time.monotonic() - t0
    if proc.returncode != 0:
        # If an NVENC encoder was used, retry with CPU fallback
        _NVENC_CODECS = {
            "hevc_nvenc": "libx265",
            "h264_nvenc": "libx264",
            "av1_nvenc": "libsvtav1",
        }
        nvenc_used = any(
            x in run_cmd for x in _NVENC_CODECS
        )
        if nvenc_used:
            logger.warning(
                f"[{label}] NVENC failed (code "
                f"{proc.returncode}), retrying with CPU"
            )
            cpu_cmd = []
            skip_next = False
            for i, tok in enumerate(cmd):
                if skip_next:
                    skip_next = False
                    continue
                if tok in _NVENC_CODECS:
                    cpu_cmd.append(_NVENC_CODECS[tok])
                    cpu_cmd.extend(["-crf", "18"])
                elif tok in ("-preset", "-rc", "-cq", "-b:v"):
                    skip_next = True
                    continue
                elif tok == "-hwaccel":
                    skip_next = True
                    continue
                elif tok == "vbr":
                    continue
                else:
                    cpu_cmd.append(tok)
            _run_ffmpeg_logged(
                cpu_cmd, f"{label}-cpu", total_frames
            )
            return

        logger.error(
            f"[{label}] ffmpeg exited with code "
            f"{proc.returncode} after {elapsed:.0f}s"
        )
        raise subprocess.CalledProcessError(
            proc.returncode, run_cmd
        )
    logger.info(
        f"[{label}] done — {last_frame} frames "
        f"in {elapsed:.0f}s"
    )


def check_ffmpeg() -> bool:
    """Check if ffmpeg and ffprobe are available on PATH."""
    return (
        shutil.which("ffmpeg") is not None
        and shutil.which("ffprobe") is not None
    )


def get_video_info(path: str | Path) -> dict:
    """Probe a video file and return metadata.

    Returns:
        Dict with keys: width, height, fps, duration, codec, num_frames.
    """
    path = str(path)
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)

    video_stream = None
    for s in data.get("streams", []):
        if s.get("codec_type") == "video":
            video_stream = s
            break

    if not video_stream:
        raise ValueError(f"No video stream found in {path}")

    # Parse frame rate (e.g. "30000/1001" or "30/1")
    fps_str = video_stream.get("r_frame_rate", "30/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den)

    duration = float(data.get("format", {}).get("duration", 0))
    num_frames = int(video_stream.get("nb_frames", int(fps * duration)))

    return {
        "width": int(video_stream["width"]),
        "height": int(video_stream["height"]),
        "fps": round(fps, 3),
        "fps_str": fps_str,
        "duration": round(duration, 2),
        "num_frames": num_frames,
        "codec": video_stream.get("codec_name", "unknown"),
    }


def has_audio(path: str | Path) -> bool:
    """Check whether a video file contains an audio stream.

    Args:
        path: Path to the video file.

    Returns:
        True if the file has at least one audio stream.
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return bool(result.stdout.strip())


def extract_frame(video_path: str | Path, frame_num: int,
                  output_path: str | Path) -> Path:
    """Extract a single frame from a video as PNG.

    Args:
        video_path: Path to the input video.
        frame_num: 0-based frame index to extract.
        output_path: Where to save the PNG.

    Returns:
        Path to the saved frame.
    """
    output_path = Path(output_path)
    info = get_video_info(video_path)
    timestamp = frame_num / info["fps"]

    cmd = [
        "ffmpeg", "-y",
        *_hwaccel_args(),
        "-ss", f"{timestamp:.4f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        str(output_path),
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def split_sbs(video_path: str | Path, left_path: str | Path,
              right_path: str | Path, crf: int = 18) -> None:
    """Split a side-by-side stereo video into left and right eye videos.

    Args:
        video_path: Path to the SBS input video.
        left_path: Output path for the left eye.
        right_path: Output path for the right eye.
        crf: Constant rate factor for encoding quality (lower = better).
    """
    enc = _encode_args("libx264", crf)
    cmd = [
        "ffmpeg", "-y",
        *_hwaccel_args(),
        "-i", str(video_path),
        "-filter_complex",
        "[0:v]crop=iw/2:ih:0:0[L];[0:v]crop=iw/2:ih:iw/2:0[R]",
        "-map", "[L]", *enc, str(left_path),
        "-map", "[R]", *enc, str(right_path),
    ]
    total_frames = 0
    try:
        info = get_video_info(video_path)
        total_frames = info.get("num_frames", 0)
    except Exception:
        pass
    _run_ffmpeg_logged(
        cmd, "split-sbs", total_frames=total_frames
    )


def stack_sbs(left_path: str | Path, right_path: str | Path,
              output_path: str | Path, crf: int = 18) -> None:
    """Recombine left and right videos into side-by-side.

    Args:
        left_path: Left eye video.
        right_path: Right eye video.
        output_path: Output SBS video path.
        crf: Encoding quality.
    """
    cmd = [
        "ffmpeg", "-y",
        *_hwaccel_args(),
        "-i", str(left_path), "-i", str(right_path),
        "-filter_complex", "[0:v][1:v]hstack[out]",
        "-map", "[out]",
        *_encode_args("libx264", crf),
        str(output_path),
    ]
    total_frames = 0
    try:
        info = get_video_info(left_path)
        total_frames = info.get("num_frames", 0)
    except Exception:
        pass
    _run_ffmpeg_logged(
        cmd, "stack-sbs", total_frames=total_frames
    )


def convert_to_fisheye(
    video_path: str | Path,
    mask_path: str | Path | None,
    output_path: str | Path,
    fov: int = 180,
    codec: str = "libx265",
    crf: int = 18,
) -> None:
    """Convert equirectangular SBS video to fisheye projection.

    Uses DeoVR's documented FFmpeg filter chain.  When *mask_path*
    is provided the DeoVR circular mask is overlaid (use for the
    source video).  When ``None`` the mask step is skipped (use for
    the matte so alpha data extends beyond the fisheye circles).
    """
    video_path = Path(video_path)
    use_mask = mask_path is not None

    if not video_path.exists():
        raise FileNotFoundError(
            f"Input video not found: {video_path}"
        )
    if use_mask:
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise FileNotFoundError(
                f"Fisheye mask not found: {mask_path}"
            )

    vid_mb = video_path.stat().st_size / 1_048_576
    logger.info(
        f"convert_to_fisheye: input={video_path.name} "
        f"({vid_mb:.0f} MB), FOV={fov}, codec={codec}"
    )

    # Get frame count for progress reporting
    total_frames = 0
    try:
        info = get_video_info(video_path)
        total_frames = info.get("num_frames", 0)
    except Exception:
        pass

    v360 = (
        f"v360=hequirect:fisheye:"
        f"iv_fov=180:ih_fov=180:v_fov={fov}:h_fov={fov}"
    )
    if use_mask:
        filter_complex = (
            f"[0:v]split=2[L][R];"
            f"[L]crop=iw/2:ih:0:0[Lc];[R]crop=iw/2:ih:iw/2:0[Rc];"
            f"[Lc]{v360}[Lf];[Rc]{v360}[Rf];"
            f"[Lf][Rf]hstack[stk];"
            f"[1:v][stk]scale2ref[msk][base];"
            f"[base][msk]overlay=0:0:shortest=1"
        )
    else:
        filter_complex = (
            f"[0:v]split=2[L][R];"
            f"[L]crop=iw/2:ih:0:0[Lc];[R]crop=iw/2:ih:iw/2:0[Rc];"
            f"[Lc]{v360}[Lf];[Rc]{v360}[Rf];"
            f"[Lf][Rf]hstack"
        )
    cmd = [
        "ffmpeg", "-y",
        *_hwaccel_args(),
        "-i", str(video_path),
        *(
            ["-loop", "1", "-i", str(mask_path)]
            if use_mask else []
        ),
        "-filter_complex", filter_complex,
        *_encode_args(codec, crf), "-c:a", "copy",
        str(output_path),
    ]
    _run_ffmpeg_logged(
        cmd, "fisheye", total_frames=total_frames
    )


def apply_fisheye_mask(
    video_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    codec: str = "libx265",
    crf: int = 18,
) -> None:
    """Apply the circular fisheye mask to an already-fisheye video.

    Blacks out pixels outside the fisheye circles without
    re-projecting. Used when the source is already in fisheye
    projection but needs clean circular boundaries.
    """
    total_frames = 0
    try:
        info = get_video_info(video_path)
        total_frames = info.get("num_frames", 0)
    except Exception:
        pass

    filter_complex = (
        "[0:v]format=yuv420p[vid];"
        "[1:v][vid]scale2ref[msk][base];"
        "[base][msk]overlay=0:0:shortest=1,format=yuv420p"
    )
    cmd = [
        "ffmpeg", "-y",
        *_hwaccel_args(),
        "-i", str(video_path),
        "-loop", "1", "-i", str(mask_path),
        "-filter_complex", filter_complex,
        *_encode_args(codec, crf), "-c:a", "copy",
        str(output_path),
    ]
    _run_ffmpeg_logged(
        cmd, "fisheye-mask", total_frames=total_frames
    )


def matte_to_red_channel(input_path: str | Path,
                         output_path: str | Path,
                         crf: int = 18) -> None:
    """Convert a grayscale matte video to red-channel-only format."""
    total_frames = 0
    try:
        info = get_video_info(input_path)
        total_frames = info.get("num_frames", 0)
    except Exception:
        pass

    cmd = [
        "ffmpeg", "-y",
        *_hwaccel_args(),
        "-i", str(input_path),
        "-filter_complex",
        "format=rgb24,geq=r='r(X,Y)':g='0':b='0',format=yuv420p",
        *_encode_args("libx265", crf),
        str(output_path),
    ]
    _run_ffmpeg_logged(
        cmd, "red-channel", total_frames=total_frames
    )


def pack_alpha(
    video_path: str | Path,
    matte_path: str | Path,
    output_path: str | Path,
    codec: str = "libsvtav1",
    crf: int = 18,
) -> None:
    """Pack alpha matte into fisheye video for DeoVR passthrough.

    Follows the DeoVR alpha packing spec:
    1. Scale matte to 40% of the video resolution.
    2. Convert to red-channel-only, apply colorkey.
    3. Split into 6 segments.
    4. Overlay each segment into the dark corner areas
       around the fisheye circles.

    Args:
        video_path: The fisheye video (SBS with dark borders).
        matte_path: Grayscale matte video (white = person).
        output_path: Final output video.
        codec: Video codec (default AV1).
        crf: Encoding quality.
    """
    info = get_video_info(video_path)
    total_frames = info.get("num_frames", 0)
    W = info["width"]
    H = info["height"]

    # ── DeoVR segment geometry (40% scale) ────────────────
    w_out = W * 4 // 10
    h_out = H * 4 // 10
    hw = w_out // 2          # half-width of 40% matte
    hh = h_out // 2          # half-height of 40% matte
    qw = w_out // 4          # quarter-width

    # Crop regions on the 40% matte
    # (x, y, w, h) for each of 6 segments
    crops = [
        (0,  0,  hw, hh),                # s1: left-top
        (0,  hh, hw, hh),                # s2: left-bottom
        (hw, 0,  qw, hh),                # s3: mid-right top
        (w_out - qw, 0, qw, hh),         # s4: far-right top
        (hw, hh, hw, hh),                # s5: right bottom
        (w_out - qw, hh, qw, hh),        # s6: far-right bot
    ]

    # Overlay positions on the full video frame
    x_ctr = W // 2 - hw // 2             # centred in left half
    y_bot = H - hh                        # bottom edge
    x_far = W - qw                        # right edge
    positions = [
        (x_ctr, y_bot),    # s1 → bottom-center
        (x_ctr, 0),        # s2 → top-center
        (x_far, y_bot),    # s3 → bottom-right
        (0,     y_bot),    # s4 → bottom-left
        (x_far, 0),        # s5 → top-right
        (0,     0),         # s6 → top-left
    ]

    # ── Build ffmpeg filter chain ─────────────────────────
    # Scale matte to 40%, make red-only, colorkey black→transparent
    prep = (
        f"[1:v]scale={w_out}:{h_out},"
        f"format=rgba,"
        f"colorchannelmixer="
        f"rr=1:rg=0:rb=0:ra=0:"
        f"gr=0:gg=0:gb=0:ga=0:"
        f"br=0:bg=0:bb=0:ba=0:"
        f"ar=0:ag=0:ab=0:aa=1,"
        f"colorkey=color=black:similarity=0.1,"
        f"split=6"
    )
    seg_labels = "[o1][o2][o3][o4][o5][o6]"
    parts = [f"{prep}{seg_labels}"]

    for i, (cx, cy, cw, ch) in enumerate(crops):
        parts.append(
            f"[o{i+1}]crop={cw}:{ch}:{cx}:{cy}[s{i+1}]"
        )

    prev = "[0:v]"
    for i, (px, py) in enumerate(positions):
        out = f"[a{i}]" if i < 5 else ",format=yuv420p[out]"
        parts.append(
            f"{prev}[s{i+1}]overlay={px}:{py}{out}"
        )
        if i < 5:
            prev = f"[a{i}]"

    filter_complex = ";".join(parts)

    cmd = [
        "ffmpeg", "-y",
        *_hwaccel_args(),
        "-i", str(video_path), "-i", str(matte_path),
        "-filter_complex", filter_complex,
        "-map", "[out]", "-map", "0:a?",
        *_encode_args(codec, crf), "-c:a", "copy",
        str(output_path),
    ]
    _run_ffmpeg_logged(
        cmd, "pack-alpha", total_frames=total_frames
    )
