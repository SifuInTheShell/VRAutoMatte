"""FFmpeg wrapper utilities for video processing."""

import json
import shutil
import subprocess
from pathlib import Path

from loguru import logger


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
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-filter_complex",
        "[0:v]crop=iw/2:ih:0:0[L];[0:v]crop=iw/2:ih:iw/2:0[R]",
        "-map", "[L]", "-c:v", "libx264", "-crf", str(crf), str(left_path),
        "-map", "[R]", "-c:v", "libx264", "-crf", str(crf), str(right_path),
    ]
    logger.info(f"Splitting SBS: {video_path}")
    subprocess.run(cmd, capture_output=True, check=True)


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
        "-i", str(left_path), "-i", str(right_path),
        "-filter_complex", "[0:v][1:v]hstack[out]",
        "-map", "[out]",
        "-c:v", "libx264", "-crf", str(crf),
        str(output_path),
    ]
    logger.info(f"Stacking SBS: {output_path}")
    subprocess.run(cmd, capture_output=True, check=True)


def convert_to_fisheye(
    video_path: str | Path,
    mask_path: str | Path,
    output_path: str | Path,
    fov: int = 180,
    codec: str = "libx265",
    crf: int = 18,
) -> None:
    """Convert equirectangular SBS video to fisheye projection.

    Uses DeoVR's documented FFmpeg filter chain.

    Args:
        video_path: Input equirectangular SBS video.
        mask_path: DeoVR fisheye mask PNG (e.g. mask8k.png).
        output_path: Output fisheye SBS video.
        fov: Field of view for fisheye (typically 180-200).
        codec: Video codec (libx265 recommended).
        crf: Encoding quality.
    """
    v360 = (
        f"v360=hequirect:fisheye:"
        f"iv_fov=180:ih_fov=180:v_fov={fov}:h_fov={fov}"
    )
    filter_complex = (
        f"[0:v]split=2[L][R];"
        f"[L]crop=iw/2:ih:0:0[Lc];[R]crop=iw/2:ih:iw/2:0[Rc];"
        f"[Lc]{v360}[Lf];[Rc]{v360}[Rf];"
        f"[Lf][Rf]hstack[stk];[stk][1:v]overlay=0:0"
    )
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path), "-i", str(mask_path),
        "-filter_complex", filter_complex,
        "-c:v", codec, "-crf", str(crf), "-c:a", "copy",
        str(output_path),
    ]
    logger.info(f"Converting to fisheye (FOV={fov}): {output_path}")
    subprocess.run(cmd, capture_output=True, check=True)


def matte_to_red_channel(input_path: str | Path,
                         output_path: str | Path,
                         crf: int = 18) -> None:
    """Convert a grayscale matte video to red-channel-only format.

    DeoVR's alpha packing uses only the red channel of the matte.

    Args:
        input_path: Grayscale matte video.
        output_path: Red-channel-only matte video.
        crf: Encoding quality.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-filter_complex",
        "format=rgb24,geq=r='r(X,Y)':g='0':b='0'",
        "-c:v", "libx264", "-crf", str(crf),
        str(output_path),
    ]
    logger.info(f"Converting matte to red channel: {output_path}")
    subprocess.run(cmd, capture_output=True, check=True)


def pack_alpha(
    video_path: str | Path,
    matte_path: str | Path,
    output_path: str | Path,
    codec: str = "libx265",
    crf: int = 18,
) -> None:
    """Pack video and alpha matte vertically for DeoVR alpha passthrough.

    The output has the video on top and the red-channel matte on bottom,
    stacked vertically. The filename must contain '_ALPHA' for DeoVR
    to recognize it.

    Args:
        video_path: The fisheye-converted video.
        matte_path: The red-channel matte video.
        output_path: Final packed output video.
        codec: Video codec.
        crf: Encoding quality.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path), "-i", str(matte_path),
        "-filter_complex", "[0:v][1:v]vstack[out]",
        "-map", "[out]", "-map", "0:a?",
        "-c:v", codec, "-crf", str(crf), "-c:a", "copy",
        str(output_path),
    ]
    logger.info(f"Packing alpha: {output_path}")
    subprocess.run(cmd, capture_output=True, check=True)
