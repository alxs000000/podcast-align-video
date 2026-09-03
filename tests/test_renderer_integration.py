from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from podcast_align_video.config import RendererConfig, load_config
from podcast_align_video.media import validate_video
from podcast_align_video.render.hybrid import measure_layouts, render_video


@pytest.mark.skipif(not os.environ.get("PAV_RUN_RENDER_INTEGRATION"), reason="set PAV_RUN_RENDER_INTEGRATION=1")
def test_real_playwright_ass_ffmpeg_render(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is unavailable")
    filters = subprocess.run(
        ["ffmpeg", "-hide_banner", "-filters"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=True,
    ).stdout
    if not any(line.split()[1:2] == ["ass"] for line in filters.splitlines() if len(line.split()) >= 2):
        pytest.skip("this FFmpeg build has no libass filter")
    browser_override = os.environ.get("PLAYWRIGHT_BROWSERS_PATH")
    browser_cache = Path(browser_override) if browser_override else load_config(None).browser_cache
    if not browser_cache.is_dir():
        pytest.skip("Playwright Chromium is unavailable; run scripts/setup.sh")
    audio = tmp_path / "source.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=2.4:sample_rate=48000",
            str(audio),
        ],
        check=True,
    )
    alignment = {
        "schema_version": 1,
        "metadata": {"audio_duration_seconds": 2.4},
        "words": [
            {"index": 0, "text": "A", "focus_start": 0.0, "focus_end": 0.4},
            {"index": 1, "text": "faithful", "focus_start": 0.4, "focus_end": 0.8},
            {"index": 2, "text": "hybrid", "focus_start": 0.8, "focus_end": 1.2},
            {"index": 3, "text": "subtitle", "focus_start": 1.2, "focus_end": 1.7},
            {"index": 4, "text": "video.", "focus_start": 1.7, "focus_end": 2.3},
        ],
        "sentences": [
            {"index": 0, "word_start": 0, "word_end": 3, "start": 0.0, "end": 1.2},
            {"index": 1, "word_start": 3, "word_end": 5, "start": 1.2, "end": 2.3},
        ],
        "chunks": [{"index": 0, "word_start": 0, "word_end": 5}],
    }
    config = RendererConfig(segment_seconds=1.2)
    work = tmp_path / "render"
    output = work / "video.mp4"
    report = render_video(
        alignment=alignment,
        audio=audio,
        output_dir=work,
        output=output,
        config=config,
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        browser_cache=browser_cache,
        fingerprint="integration",
        logger=lambda _: None,
    )
    assert report["segment_count"] == 2
    validate_video(
        output,
        ffprobe_binary="ffprobe",
        width=1920,
        height=1080,
        fps=30,
        expected_duration=2.4,
        require_audio=True,
    )
    layouts = [json.loads(line) for line in (work / "layout.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [len(item["words"]) for item in layouts] == [3, 2]
    assert all(16 <= word["fontSize"] <= 76 for item in layouts for word in item["words"])
    segment_hashes = {
        path.name: path.read_bytes()
        for path in sorted((work / "segments").glob("segment-*.mp4"))
    }
    (work / "layout.jsonl").write_text(
        json.dumps(layouts[0], separators=(",", ":")) + "\n{\"sentence_index\":",
        encoding="utf-8",
    )
    messages: list[str] = []
    resumed = render_video(
        alignment=alignment,
        audio=audio,
        output_dir=work,
        output=output,
        config=config,
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        browser_cache=browser_cache,
        fingerprint="integration",
        logger=messages.append,
    )
    assert resumed["segment_count"] == 2
    assert any("layout checkpoint: 1/2" in message for message in messages)
    assert sum("reusing rendered segment" in message for message in messages) == 2
    assert segment_hashes == {
        path.name: path.read_bytes()
        for path in sorted((work / "segments").glob("segment-*.mp4"))
    }

    snapshot_words = (
        "Reliable tools turn long conversations into clear, readable videos "
        "without wasting an afternoon."
    ).split()
    snapshot = measure_layouts(
        {
            "words": [{"text": text} for text in snapshot_words],
            "sentences": [{"word_start": 0, "word_end": len(snapshot_words)}],
        },
        tmp_path / "layout-snapshot",
        width=1920,
        height=1080,
        browser_cache=browser_cache,
        layout_fingerprint="representative-v1",
        logger=lambda _: None,
    )[0]["words"]
    assert [round(word["y"], 1) for word in snapshot] == [
        349.7, 349.7, 349.7, 349.7, 349.7,
        449.3, 449.3, 449.3, 449.3,
        549.0, 549.0, 549.0, 549.0,
    ]
    assert [snapshot[index]["x"] for index in (0, 4, 5, 9, 12)] == pytest.approx(
        [335.671875, 1104.515625, 525.609375, 464.125, 1101.0],
        abs=0.25,
    )
    assert [snapshot[index]["width"] for index in (0, 4, 12)] == pytest.approx(
        [279.90625, 479.796875, 354.859375],
        abs=0.25,
    )
    assert {round(word["fontSize"], 3) for word in snapshot} == {75.531}
    assert {word["fontWeight"] for word in snapshot} == {590}
