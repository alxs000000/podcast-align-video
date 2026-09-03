from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

from .config import RendererConfig
from .media import validate_video
from .util import run_command


def normalized_regions(regions: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    result: list[tuple[float, float]] = []
    for start, end in sorted(regions):
        start = max(0.0, min(duration, float(start)))
        end = max(0.0, min(duration, float(end)))
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def silence_candidates(speech_regions: list[tuple[float, float]], duration: float, threshold: float) -> list[tuple[float, float]]:
    speech = normalized_regions(speech_regions, duration)
    candidates = []
    cursor = 0.0
    for start, end in speech:
        if start - cursor >= threshold:
            candidates.append((cursor, start))
        cursor = max(cursor, end)
    if duration - cursor >= threshold:
        candidates.append((cursor, duration))
    return candidates


def subtract_protected(
    candidates: list[tuple[float, float]],
    protected: list[tuple[float, float]],
    threshold: float,
) -> list[tuple[float, float]]:
    protection = normalized_regions(protected, max((end for _, end in protected), default=0.0))
    output: list[tuple[float, float]] = []
    for candidate_start, candidate_end in candidates:
        pieces = [(candidate_start, candidate_end)]
        for protect_start, protect_end in protection:
            next_pieces = []
            for start, end in pieces:
                if protect_end <= start or protect_start >= end:
                    next_pieces.append((start, end))
                    continue
                if protect_start > start:
                    next_pieces.append((start, min(end, protect_start)))
                if protect_end < end:
                    next_pieces.append((max(start, protect_end), end))
            pieces = next_pieces
        output.extend((start, end) for start, end in pieces if end - start >= threshold)
    return output


def kept_regions(cuts: list[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    result = []
    cursor = 0.0
    for start, end in normalized_regions(cuts, duration):
        if start > cursor:
            result.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < duration:
        result.append((cursor, duration))
    return [(start, end) for start, end in result if end - start > 0.001]


def create_speech_cut(
    *,
    full_video: Path,
    output: Path,
    alignment: dict,
    vad_file: Path,
    threshold: float,
    renderer: RendererConfig,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    logger: Callable[[str], None],
    source_audio: Path | None = None,
) -> dict[str, object]:
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("speech-cut threshold must be positive")
    vad = json.loads(vad_file.read_text(encoding="utf-8"))
    duration = float(alignment["metadata"]["audio_duration_seconds"])
    speech = [(float(item[0]), float(item[1])) for item in vad.get("regions", [])]
    protected = [(float(word["focus_start"]), float(word["focus_end"])) for word in alignment["words"]]
    raw_candidates = silence_candidates(speech, duration, threshold)
    cuts = subtract_protected(raw_candidates, protected, threshold)
    cuts = normalized_regions(cuts, duration)
    removed = sum(end - start for start, end in cuts)
    if not cuts or removed <= 0.001:
        output.unlink(missing_ok=True)
        return {"status": "no_cuts", "threshold_seconds": threshold, "cut_count": 0, "removed_seconds": 0.0}
    kept = kept_regions(cuts, duration)
    if not kept:
        raise RuntimeError("speech-cut would remove the entire video")
    expected_duration = sum(end - start for start, end in kept)
    expression = "+".join(f"between(t,{start:.6f},{end:.6f})" for start, end in kept)
    filter_graph = (
        f"[0:v:0]select='{expression}',setpts=N/FRAME_RATE/TB[v];"
        f"[1:a:0]aselect='{expression}',asetpts=N/SR/TB[a]"
    )
    audio_input = source_audio if source_audio is not None else full_video
    temporary = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    try:
        codec = ["-c:v", "libx264", "-preset", renderer.preset, "-crf", str(renderer.crf)]
        if renderer.encoder == "h264_nvenc":
            codec = ["-c:v", "h264_nvenc", "-preset", renderer.preset, "-cq", str(renderer.crf), "-b:v", "0"]
        run_command(
            [
                ffmpeg_binary,
                "-nostdin",
                "-y",
                "-hide_banner",
                "-loglevel",
                "warning",
                "-i",
                str(full_video),
                "-i",
                str(audio_input),
                "-filter_complex",
                filter_graph,
                "-map",
                "[v]",
                "-map",
                "[a]",
                *codec,
                "-pix_fmt",
                renderer.pixel_format,
                "-r",
                str(renderer.fps),
                "-c:a",
                "aac",
                "-b:a",
                renderer.audio_bitrate,
                "-ar",
                "48000",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            logger=logger,
        )
        validate_video(
            temporary,
            ffprobe_binary=ffprobe_binary,
            width=renderer.width,
            height=renderer.height,
            fps=renderer.fps,
            expected_duration=expected_duration,
            require_audio=True,
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "status": "created",
        "threshold_seconds": threshold,
        "cut_count": len(cuts),
        "removed_seconds": round(removed, 6),
        "output_duration_seconds": round(expected_duration, 6),
    }
