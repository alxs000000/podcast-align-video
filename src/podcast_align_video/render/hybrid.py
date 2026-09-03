from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Callable

from ..config import RendererConfig
from ..media import validate_video
from ..util import atomic_write_json, atomic_write_text, run_command, sha256_file


LAYOUT_SCHEMA_VERSION = 1
RENDER_SCHEMA_VERSION = 2


def measure_layouts(
    alignment: dict,
    output_dir: Path,
    *,
    width: int,
    height: int,
    browser_cache: Path,
    layout_fingerprint: str,
    logger: Callable[[str], None],
) -> list[dict]:
    from playwright.sync_api import sync_playwright

    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "layout-meta.json"
    jsonl_path = output_dir / "layout.jsonl"
    expected_meta = {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "fingerprint": layout_fingerprint,
        "width": width,
        "height": height,
        "sentence_count": len(alignment["sentences"]),
        "word_count": len(alignment["words"]),
    }
    existing: dict[int, dict] = {}
    if meta_path.is_file() and jsonl_path.is_file():
        try:
            if json.loads(meta_path.read_text(encoding="utf-8")) == expected_meta:
                with jsonl_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            item = json.loads(line)
                            _validate_layout_record(item, alignment)
                            existing[int(item["sentence_index"])] = item
                        except Exception:
                            # A killed process may leave only its final JSONL line
                            # incomplete. Earlier lines remain valid checkpoints.
                            break
        except Exception:
            existing = {}
    atomic_write_json(meta_path, expected_meta)
    # Normalize away a possibly truncated final JSONL line before appending.
    # Every previously validated sentence remains available for resume.
    atomic_write_text(
        jsonl_path,
        "".join(
            json.dumps(existing[index], ensure_ascii=False, separators=(",", ":")) + "\n"
            for index in sorted(existing)
        ),
    )
    logger(f"layout checkpoint: {len(existing)}/{len(alignment['sentences'])} sentences reusable")

    assets = Path(__file__).with_name("assets")
    page_url = (assets / "index.html").resolve().as_uri()
    browser_cache.mkdir(parents=True, exist_ok=True)
    old_browser_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_cache)
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=1)
            page.goto(page_url, wait_until="load")
            page.evaluate("document.fonts.ready")
            for sentence_index, sentence in enumerate(alignment["sentences"]):
                if sentence_index in existing:
                    continue
                words = alignment["words"][sentence["word_start"]:sentence["word_end"]]
                texts = [str(word["text"]) for word in words]
                measured = page.evaluate("words => window.podcastAlignVideoMeasure(words)", texts)
                record = {"sentence_index": sentence_index, "words": measured}
                _validate_layout_record(record, alignment)
                with jsonl_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                existing[sentence_index] = record
                if len(existing) == 1 or len(existing) % 25 == 0 or len(existing) == len(alignment["sentences"]):
                    logger(f"measured {len(existing)}/{len(alignment['sentences'])} subtitle layouts")
            browser.close()
    finally:
        if old_browser_path is None:
            os.environ.pop("PLAYWRIGHT_BROWSERS_PATH", None)
        else:
            os.environ["PLAYWRIGHT_BROWSERS_PATH"] = old_browser_path
    return [existing[index] for index in range(len(alignment["sentences"]))]


def _validate_layout_record(record: dict, alignment: dict) -> None:
    index = int(record["sentence_index"])
    sentence = alignment["sentences"][index]
    source_words = alignment["words"][sentence["word_start"]:sentence["word_end"]]
    measured = record.get("words")
    if not isinstance(measured, list) or len(measured) != len(source_words):
        raise ValueError(f"layout word count mismatch for sentence {index}")
    for source, rect in zip(source_words, measured):
        if rect.get("text") != source.get("text"):
            raise ValueError(f"layout text mismatch for sentence {index}")
        for key in ("x", "y", "width", "height", "textX", "textY", "textWidth", "textHeight", "fontSize"):
            value = float(rect[key])
            if not math.isfinite(value) or (key in {"width", "height", "fontSize"} and value <= 0):
                raise ValueError(f"invalid layout metric {key} for sentence {index}")


def ass_time(seconds: float) -> str:
    safe = max(0.0, seconds)
    hours = int(safe // 3600)
    minutes = int((safe % 3600) // 60)
    whole = int(safe % 60)
    centiseconds = min(99, int((safe - math.floor(safe)) * 100))
    return f"{hours}:{minutes:02d}:{whole:02d}.{centiseconds:02d}"


def ass_escape(text: object) -> str:
    return str(text).replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}").replace("\n", " ")


def ass_header(width: int, height: int) -> str:
    return f"""[Script Info]
ScriptType: v4.00+
Collisions: Normal
PlayResX: {width}
PlayResY: {height}
Timer: 100.0000
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Geist,76,&HF6F2E9,&HF6F2E9,&H55000000,&HFF000000,-1,0,0,0,100,100,0,0,1,2,2,5,0,0,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""


def dialogue(layer: int, start: float, end: float, x: float, y: float, size: float, text: str, active: bool = False) -> str:
    colors = "\\1c&H170725&\\3c&H67C8FF&\\bord1\\shad0" if active else "\\1c&HF6F2E9&\\3c&H55000000&\\bord2\\shad2"
    return (
        f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},Default,,0,0,0,,"
        f"{{\\fnGeist\\b1\\an5\\pos({x:.2f},{y:.2f})\\fs{max(16, round(size))}{colors}}}{ass_escape(text)}"
    )


def rounded_rect_path(rect: dict) -> str:
    x = float(rect["x"])
    y = float(rect["y"])
    width = float(rect["width"])
    height = float(rect["height"])
    right, bottom = x + width, y + height
    radius = min(28.0, height * 0.26, width * 0.18)
    return " ".join(
        [
            f"m {x + radius:.2f} {y:.2f}",
            f"l {right - radius:.2f} {y:.2f}",
            f"b {right:.2f} {y:.2f} {right:.2f} {y:.2f} {right:.2f} {y + radius:.2f}",
            f"l {right:.2f} {bottom - radius:.2f}",
            f"b {right:.2f} {bottom:.2f} {right:.2f} {bottom:.2f} {right - radius:.2f} {bottom:.2f}",
            f"l {x + radius:.2f} {bottom:.2f}",
            f"b {x:.2f} {bottom:.2f} {x:.2f} {bottom:.2f} {x:.2f} {bottom - radius:.2f}",
            f"l {x:.2f} {y + radius:.2f}",
            f"b {x:.2f} {y:.2f} {x:.2f} {y:.2f} {x + radius:.2f} {y:.2f}",
        ]
    )


def box_dialogue(layer: int, start: float, end: float, rect: dict, blur: bool = False) -> str:
    tags = (
        "\\p1\\pos(0,0)\\an7\\1c&H67C8FF&\\alpha&H82&\\blur8\\bord0\\shad0"
        if blur
        else "\\p1\\pos(0,0)\\an7\\1c&H67C8FF&\\3c&H4397C9&\\bord3\\shad0"
    )
    return (
        f"Dialogue: {layer},{ass_time(start)},{ass_time(end)},Default,,0,0,0,,"
        f"{{{tags}}}{rounded_rect_path(rect)}{{\\p0}}"
    )


def build_ass(alignment: dict, layouts: list[dict], start: float, end: float, width: int, height: int) -> str:
    rows: list[str] = []
    for sentence, layout in zip(alignment["sentences"], layouts):
        base_start = max(start, float(sentence["start"]))
        base_end = min(end, float(sentence["end"]))
        if base_end <= base_start:
            continue
        source_words = alignment["words"][sentence["word_start"]:sentence["word_end"]]
        for word, rect in zip(source_words, layout["words"]):
            text_x = float(rect.get("textX", rect["x"]))
            text_y = float(rect.get("textY", rect["y"]))
            text_width = float(rect.get("textWidth", rect["width"]))
            text_height = float(rect.get("textHeight", rect["height"]))
            x, y = text_x + text_width / 2, text_y + text_height / 2
            focus_start = max(base_start, start, float(word["focus_start"]))
            focus_end = min(base_end, end, float(word["focus_end"]))
            if focus_end > focus_start:
                if focus_start > base_start:
                    rows.append(dialogue(0, base_start - start, focus_start - start, x, y, rect["fontSize"], word["text"]))
                if base_end > focus_end:
                    rows.append(dialogue(0, focus_end - start, base_end - start, x, y, rect["fontSize"], word["text"]))
                rows.append(box_dialogue(1, focus_start - start, focus_end - start, rect, blur=True))
                rows.append(box_dialogue(2, focus_start - start, focus_end - start, rect))
                rows.append(dialogue(3, focus_start - start, focus_end - start, x, y, rect["fontSize"], word["text"], active=True))
            else:
                rows.append(dialogue(0, base_start - start, base_end - start, x, y, rect["fontSize"], word["text"]))
    return "\ufeff" + ass_header(width, height) + "\n".join(rows) + "\n"


def render_video(
    *,
    alignment: dict,
    audio: Path,
    output_dir: Path,
    output: Path,
    config: RendererConfig,
    ffmpeg_binary: str,
    ffprobe_binary: str,
    browser_cache: Path,
    fingerprint: str,
    logger: Callable[[str], None],
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    duration = float(alignment["metadata"]["audio_duration_seconds"])
    layouts = measure_layouts(
        alignment,
        output_dir,
        width=config.width,
        height=config.height,
        browser_cache=browser_cache,
        layout_fingerprint=fingerprint,
        logger=logger,
    )
    segment_dir = output_dir / "segments"
    segment_dir.mkdir(exist_ok=True)
    segments: list[Path] = []
    start = 0.0
    segment_index = 0
    while start < duration - 1e-6:
        end = min(duration, start + config.segment_seconds)
        segment = segment_dir / f"segment-{segment_index:04d}.mp4"
        ass = segment_dir / f"segment-{segment_index:04d}.ass"
        segment_meta = segment_dir / f"segment-{segment_index:04d}.json"
        expected = {
            "schema_version": RENDER_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "index": segment_index,
            "start": round(start, 6),
            "end": round(end, 6),
            "encoder": config.encoder,
            "preset": config.preset,
            "crf": config.crf,
        }
        reusable = False
        if segment.is_file() and segment_meta.is_file():
            try:
                saved = json.loads(segment_meta.read_text(encoding="utf-8"))
                if all(saved.get(key) == value for key, value in expected.items()) and saved.get("sha256") == sha256_file(segment):
                    validate_video(
                        segment,
                        ffprobe_binary=ffprobe_binary,
                        width=config.width,
                        height=config.height,
                        fps=config.fps,
                        expected_duration=end - start,
                        require_audio=False,
                    )
                    reusable = True
            except Exception:
                reusable = False
        if reusable:
            logger(f"reusing rendered segment {segment_index + 1}")
        else:
            atomic_write_text(ass, build_ass(alignment, layouts, start, end, config.width, config.height))
            temporary = segment.with_name(f".{segment.stem}.{os.getpid()}.tmp.mp4")
            temporary.unlink(missing_ok=True)
            try:
                codec = _codec_arguments(config)
                run_command(
                    [
                        ffmpeg_binary,
                        "-nostdin",
                        "-y",
                        "-hide_banner",
                        "-loglevel",
                        "warning",
                        "-f",
                        "lavfi",
                        "-i",
                        f"color=c=0x090b0c:s={config.width}x{config.height}:r={config.fps}:d={end - start}",
                        "-vf",
                        _ass_filter(ass, Path(__file__).with_name("assets") / "fonts"),
                        "-an",
                        *codec,
                        "-pix_fmt",
                        config.pixel_format,
                        "-r",
                        str(config.fps),
                        "-movflags",
                        "+faststart",
                        str(temporary),
                    ],
                    logger=logger,
                )
                validate_video(
                    temporary,
                    ffprobe_binary=ffprobe_binary,
                    width=config.width,
                    height=config.height,
                    fps=config.fps,
                    expected_duration=end - start,
                    require_audio=False,
                )
                os.replace(temporary, segment)
                atomic_write_json(segment_meta, {**expected, "sha256": sha256_file(segment)})
                logger(f"rendered segment {segment_index + 1} ({start:.1f}-{end:.1f}s)")
            finally:
                temporary.unlink(missing_ok=True)
        segments.append(segment)
        segment_index += 1
        start = end

    concat_list = output_dir / "segments.txt"
    atomic_write_text(concat_list, "".join(f"file '{_concat_escape(path.resolve())}'\n" for path in segments))
    joined = output_dir / "video-only.mp4"
    temporary_joined = output_dir / f".video-only.{os.getpid()}.tmp.mp4"
    temporary_joined.unlink(missing_ok=True)
    run_command(
        [
            ffmpeg_binary,
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(temporary_joined),
        ],
        logger=logger,
    )
    validate_video(
        temporary_joined,
        ffprobe_binary=ffprobe_binary,
        width=config.width,
        height=config.height,
        fps=config.fps,
        expected_duration=duration,
        require_audio=False,
    )
    os.replace(temporary_joined, joined)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.stem}.{os.getpid()}.tmp.mp4")
    temporary_output.unlink(missing_ok=True)
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
                str(joined),
                "-i",
                str(audio),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                config.audio_bitrate,
                "-ar",
                "48000",
                "-t",
                f"{duration:.6f}",
                "-movflags",
                "+faststart",
                str(temporary_output),
            ],
            logger=logger,
        )
        validation = validate_video(
            temporary_output,
            ffprobe_binary=ffprobe_binary,
            width=config.width,
            height=config.height,
            fps=config.fps,
            expected_duration=duration,
            require_audio=True,
        )
        os.replace(temporary_output, output)
    finally:
        temporary_output.unlink(missing_ok=True)
        temporary_joined.unlink(missing_ok=True)
    return {"segment_count": len(segments), **validation}


def _codec_arguments(config: RendererConfig) -> list[str]:
    if config.encoder == "libx264":
        return ["-c:v", "libx264", "-preset", config.preset, "-crf", str(config.crf)]
    if config.encoder == "h264_nvenc":
        return ["-c:v", config.encoder, "-preset", config.preset, "-cq", str(config.crf), "-b:v", "0"]
    raise ValueError("renderer.encoder must be libx264 or h264_nvenc")


def _ass_filter(ass: Path, fonts: Path) -> str:
    def escaped(path: Path) -> str:
        return str(path.resolve()).replace("\\", "/").replace(":", "\\:").replace("'", "\\'")

    return f"ass=filename='{escaped(ass)}':fontsdir='{escaped(fonts)}'"


def _concat_escape(path: Path) -> str:
    return str(path).replace("'", "'\\''")
