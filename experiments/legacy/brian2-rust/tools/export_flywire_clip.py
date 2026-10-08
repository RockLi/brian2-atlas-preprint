#!/usr/bin/env python3
"""Export a game recording to an MP4 with a continuous, timestamp-aligned track."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="WebM saved by Record 8 rounds")
    parser.add_argument("output", type=Path, help="New .mp4 path (never overwritten)")
    parser.add_argument("--gain-db", type=float, default=12, help="Effect gain (default: 12 dB)")
    args = parser.parse_args()
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        parser.error("ffmpeg and ffprobe are required")
    if not args.source.is_file():
        parser.error("source must be an existing recording")
    if args.output.suffix.lower() != ".mp4" or args.output.exists():
        parser.error("output must be a new .mp4 file")
    if not -60 <= args.gain_db <= 24:
        parser.error("gain must be between -60 and 24 dB")
    # MediaRecorder WebM files often have no duration metadata. Bound padding by
    # the actual video packets instead of relying on -shortest with sparse audio.
    packets = json.loads(subprocess.check_output([
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets",
        "-show_entries", "packet=pts_time,duration_time", "-of", "json", str(args.source),
    ]))["packets"]
    duration = max((float(p["pts_time"]) + float(p.get("duration_time", 1 / 30))
                    for p in packets if "pts_time" in p), default=0)
    if not math.isfinite(duration) or duration <= 0:
        parser.error("recording has no valid video duration")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Older recordings contain no audio packets while the Web Audio graph is
    # idle. Fill those timestamp gaps with silence before AAC encoding; shifting
    # the entire audio track or concatenating samples cannot fix per-round gaps.
    audio_filter = (
        "aresample=async=1:first_pts=0,"
        f"volume={args.gain_db}dB,alimiter=limit=0.9:level=false:latency=true,apad"
    )
    # Render audio separately so sparse legacy packets cannot stall FFmpeg's
    # interleaved audio/video filter scheduling while silence is inserted.
    with tempfile.TemporaryDirectory(prefix="flylab-audio-") as temporary:
        wav = str(Path(temporary) / "continuous.wav")
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
            "-i", str(args.source), "-map", "0:a:0", "-vn", "-af", audio_filter,
            "-t", str(duration), "-c:a", "pcm_s16le", wav,
        ], check=True)
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-nostdin", "-n",
            "-i", str(args.source), "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
            "-r", "30", "-c:a", "aac", "-b:a", "160k", "-t", str(duration),
            "-movflags", "+faststart", str(args.output),
        ], check=True)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
