from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

from .util import run_command


def ffprobe(path: Path, ffprobe_binary: str = "ffprobe") -> dict:
    completed = run_command(
        [
            ffprobe_binary,
            "-v",
            "error",
            "-show_streams",
            "-show_format",
            "-of",
            "json",
            str(path),
        ],
        capture=True,
    )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"ffprobe returned invalid JSON for {path.name}") from error


def audio_info(path: Path, ffprobe_binary: str = "ffprobe") -> dict[str, object]:
    if not path.is_file():
        raise ValueError(f"source is not a regular file: {path}")
    if not os.access(path, os.R_OK):
        raise ValueError(f"source is not readable: {path}")
    probe = ffprobe(path, ffprobe_binary)
    streams = probe.get("streams", [])
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not audio_streams:
        raise ValueError(f"source has no decodable audio stream: {path}")
    raw_duration = probe.get("format", {}).get("duration") or audio_streams[0].get("duration")
    try:
        duration = float(raw_duration)
    except (TypeError, ValueError) as error:
        raise ValueError(f"source has no finite duration: {path}") from error
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f"source has invalid duration: {duration}")
    stream = audio_streams[0]
    return {
        "duration_seconds": duration,
        "codec": stream.get("codec_name"),
        "sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        "channels": int(stream["channels"]) if stream.get("channels") else None,
    }


def make_analysis_wav(
    source: Path,
    destination: Path,
    *,
    ffmpeg_binary: str = "ffmpeg",
    ffprobe_binary: str = "ffprobe",
    logger: Callable[[str], None] | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.{os.getpid()}.tmp.wav")
    temporary.unlink(missing_ok=True)
    try:
        run_command(
            [
                ffmpeg_binary,
                "-nostdin",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            logger=logger,
        )
        info = audio_info(temporary, ffprobe_binary=ffprobe_binary)
        if info["sample_rate"] != 16000 or info["channels"] != 1:
            raise RuntimeError("analysis WAV validation failed")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def validate_video(
    path: Path,
    *,
    ffprobe_binary: str,
    width: int,
    height: int,
    fps: int,
    expected_duration: float,
    require_audio: bool,
) -> dict[str, object]:
    probe = ffprobe(path, ffprobe_binary)
    streams = probe.get("streams", [])
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or (require_audio and len(audios) != 1) or (not require_audio and audios):
        raise RuntimeError(f"unexpected streams in {path.name}: video={len(videos)}, audio={len(audios)}")
    video = videos[0]
    if video.get("codec_name") != "h264":
        raise RuntimeError(f"expected H.264 video, found {video.get('codec_name')}")
    if int(video.get("width", 0)) != width or int(video.get("height", 0)) != height:
        raise RuntimeError("video dimensions do not match the configured renderer")
    rate = video.get("avg_frame_rate", "0/1")
    numerator, denominator = (float(part) for part in rate.split("/", 1))
    measured_fps = numerator / denominator if denominator else 0.0
    if not math.isfinite(measured_fps) or abs(measured_fps - fps) > 0.02:
        raise RuntimeError(f"video frame rate mismatch: {measured_fps} != {fps}")
    if require_audio and audios[0].get("codec_name") != "aac":
        raise RuntimeError(f"expected AAC audio, found {audios[0].get('codec_name')}")
    if require_audio and int(audios[0].get("sample_rate", 0)) != 48000:
        raise RuntimeError(f"expected 48 kHz AAC audio, found {audios[0].get('sample_rate')}")
    raw_duration = probe.get("format", {}).get("duration") or video.get("duration")
    duration = float(raw_duration)
    if not math.isfinite(duration) or duration <= 0 or not math.isfinite(expected_duration) or expected_duration <= 0:
        raise RuntimeError("video duration is not finite and positive")
    tolerance = max(0.25, min(2.0, expected_duration * 0.002))
    if abs(duration - expected_duration) > tolerance:
        raise RuntimeError(f"duration mismatch: expected {expected_duration:.3f}s, found {duration:.3f}s")
    return {"duration_seconds": duration, "video_codec": "h264", "audio_codec": "aac" if audios else None}
