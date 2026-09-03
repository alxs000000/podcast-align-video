from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from podcast_align_video.config import RendererConfig
from podcast_align_video.speech_cut import create_speech_cut


@pytest.mark.skipif(not os.environ.get("PAV_RUN_RENDER_INTEGRATION"), reason="set PAV_RUN_RENDER_INTEGRATION=1")
def test_real_ffmpeg_speech_cut(tmp_path: Path) -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg is unavailable")
    original_audio = tmp_path / "original.wav"
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
            "sine=frequency=440:duration=12:sample_rate=48000",
            "-c:a",
            "pcm_s16le",
            str(original_audio),
        ],
        check=True,
    )
    source = tmp_path / "video.mp4"
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
            "color=c=black:s=320x180:r=30:d=12",
            "-i",
            str(original_audio),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ],
        check=True,
    )
    vad = tmp_path / "vad.json"
    vad.write_text(json.dumps({"regions": [[0, 4], [10, 12]]}), encoding="utf-8")
    alignment = {
        "metadata": {"audio_duration_seconds": 12.0},
        "words": [
            {"focus_start": 0.5, "focus_end": 1.0},
            {"focus_start": 10.5, "focus_end": 11.0},
        ],
    }
    output = tmp_path / "cut.mp4"
    result = create_speech_cut(
        full_video=source,
        source_audio=original_audio,
        output=output,
        alignment=alignment,
        vad_file=vad,
        threshold=5.0,
        renderer=RendererConfig(width=320, height=180),
        ffmpeg_binary="ffmpeg",
        ffprobe_binary="ffprobe",
        logger=lambda _: None,
    )
    assert result["status"] == "created"
    assert result["cut_count"] == 1
    assert result["removed_seconds"] == pytest.approx(6.0)
    assert output.is_file()
