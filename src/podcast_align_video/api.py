from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import RunConfig, load_config
from .pipeline import execute


@dataclass(frozen=True)
class RunResult:
    job_id: str
    output_directory: Path
    full_video: Path
    speech_cut_video: Path | None
    source_audio: Path
    transcript: Path
    word_timings: Path
    manifest: Path
    log: Path
    warnings: tuple[str, ...]


def run(
    source: str | Path,
    output_dir: Path | None = None,
    config: RunConfig | Path | None = None,
) -> RunResult:
    """Run the synchronous v0.1 pipeline for one local file or public YouTube video."""
    result = execute(source, output_dir=output_dir, config=load_config(config))
    return RunResult(
        job_id=result.job_id,
        output_directory=result.output_dir,
        full_video=result.full_video,
        speech_cut_video=result.speech_cut_video,
        source_audio=result.source_audio,
        transcript=result.transcript,
        word_timings=result.word_timings,
        manifest=result.manifest,
        log=result.log,
        warnings=result.warnings,
    )
