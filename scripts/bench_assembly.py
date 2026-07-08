"""Assembly stage-isolation benchmark.

Times each layer of the fused DeoVR assembly on real content so
you can see where the wall time actually goes:

    1. decode only            (NVDEC/CPU decode ceiling)
    2. decode + v360          (adds the CPU fisheye projection)
    3. full filter graph      (adds the DeoVR alpha pack)
    4. full graph + encode    (adds NVENC/CPU final encode)

Usage:
    uv run python scripts/bench_assembly.py VIDEO [--frames 900]
        [--fov 180] [--preset p2] [--matte MATTE.mp4]

Without --matte, a synthetic mid-gray matte (lavfi) stands in —
fine for performance, since the pack graph's cost doesn't depend
on matte content.

Reading the result:
    - (2) - (1) = v360 cost -> if dominant, GPU projection
      (remap_opencl) is the next optimization.
    - (4) - (3) = encoder cost -> if dominant, drop the NVENC
      preset (Final Encode: Fast) or lower CRF expectations.
    - (1) dominant -> assembly is decode-bound; only NVDEC-
      resident filtering would help.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from vrautomatte.utils.ffmpeg import (  # noqa: E402
    _alpha_pack_graph_parts,
    _encode_args,
    _fisheye_graph_parts,
    _hwaccel_args,
    get_video_info,
)


def run_timed(label, cmd, frames):
    t0 = time.monotonic()
    r = subprocess.run(
        cmd, stdin=subprocess.DEVNULL,
        capture_output=True,
    )
    dt = time.monotonic() - t0
    if r.returncode != 0:
        tail = r.stderr.decode(errors="replace")[-400:]
        print(f"{label:<28} FAILED\n{tail}")
        return None
    fps = frames / dt if dt > 0 else 0
    print(f"{label:<28} {dt:7.1f}s   {fps:7.1f} fps")
    return dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--frames", type=int, default=900)
    ap.add_argument("--fov", type=int, default=180)
    ap.add_argument("--preset", default="p2")
    ap.add_argument("--matte", default=None)
    ap.add_argument(
        "--ss", type=float, default=60.0,
        help="seek offset in seconds (skip intro)",
    )
    args = ap.parse_args()

    info = get_video_info(args.video)
    W, H = info["width"], info["height"]
    fps = info["fps"]
    dur = args.frames / fps
    n = args.frames
    print(
        f"Input: {W}x{H} @ {fps} fps, testing {n} frames "
        f"({dur:.1f}s) from t={args.ss:.0f}s\n"
    )

    base_in = [
        "ffmpeg", "-y", "-v", "error",
        *_hwaccel_args(),
        "-ss", f"{args.ss:.3f}",
        "-i", args.video,
    ]
    if args.matte:
        matte_in = ["-i", args.matte]
    else:
        matte_in = [
            "-f", "lavfi",
            "-i",
            f"color=gray:size={W // 2}x{H // 2}:rate={fps}",
        ]
    frames_cap = ["-frames:v", str(n)]

    v360_parts = ";".join(
        _fisheye_graph_parts("[0:v]", "[vid]", args.fov, "F")
    )
    full_parts = ";".join(
        _fisheye_graph_parts("[0:v]", "[vid]", args.fov, "F")
        + _fisheye_graph_parts(
            "[1:v]", "[mat]", args.fov, "M"
        )
        + _alpha_pack_graph_parts(W, H, "[vid]", "[mat]")
    )

    t1 = run_timed(
        "1. decode only",
        base_in + frames_cap + ["-f", "null", "-"],
        n,
    )
    t2 = run_timed(
        "2. decode + v360",
        base_in + [
            "-filter_complex", v360_parts,
            "-map", "[vid]", *frames_cap,
            "-f", "null", "-",
        ],
        n,
    )
    t3 = run_timed(
        "3. full graph (no encode)",
        base_in + matte_in + [
            "-filter_complex", full_parts,
            "-map", "[out]", *frames_cap,
            "-f", "null", "-",
        ],
        n,
    )
    t4 = run_timed(
        "4. full graph + encode",
        base_in + matte_in + [
            "-filter_complex", full_parts,
            "-map", "[out]", *frames_cap,
            *_encode_args("libsvtav1", 18, args.preset),
            "-f", "null", "-",
        ],
        n,
    )

    if all(x is not None for x in (t1, t2, t3, t4)):
        print(
            f"\nBreakdown: decode {t1:.1f}s | "
            f"v360 +{t2 - t1:.1f}s | "
            f"pack +{t3 - t2:.1f}s | "
            f"encode +{t4 - t3:.1f}s"
        )
        worst = max(
            (t1, "decode -> NVDEC-resident filtering"),
            (t2 - t1, "v360 -> GPU remap (remap_opencl)"),
            (t3 - t2, "pack graph -> filter threads"),
            (t4 - t3, "encode -> faster NVENC preset"),
        )
        print(f"Dominant cost: {worst[1]}")


if __name__ == "__main__":
    main()
