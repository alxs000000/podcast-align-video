from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Callable

from .checkpoints import stage_fingerprint, valid_checkpoint, write_checkpoint
from .config import RunConfig
from .doctor import require_ready
from .manifest import Manifest
from .media import audio_info, make_analysis_wav, validate_video
from .render import render_video
from .sources import prepare_source
from .speech_cut import create_speech_cut
from .util import (
    RunLogger,
    artifact_record,
    atomic_copy,
    atomic_write_json,
    safe_exception_text,
    sha256_file,
    slugify,
    stable_hash,
)
from .validate import validate_word_timings


PIPELINE_SCHEMA_VERSION = 1


@dataclasses.dataclass(frozen=True)
class PipelineResult:
    job_id: str
    output_dir: Path
    full_video: Path
    speech_cut_video: Path | None
    source_audio: Path
    transcript: Path
    word_timings: Path
    manifest: Path
    log: Path
    warnings: tuple[str, ...]


def execute(
    source: str | Path,
    *,
    output_dir: Path | None,
    config: RunConfig,
) -> PipelineResult:
    require_ready(config)
    prepared = prepare_source(source, config.data_root, config.runtime.ffprobe)
    try:
        fingerprint = stable_hash(
            {
                "pipeline_schema_version": PIPELINE_SCHEMA_VERSION,
                "source_sha256": prepared.sha256,
                "config": config.fingerprint_dict(),
            }
        )
        job_id = f"{slugify(prepared.title)}-{fingerprint[:12]}"
        destination = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (Path.cwd() / "outputs" / job_id).resolve()
        )
        _validate_destination(destination, fingerprint)
        destination.mkdir(parents=True, exist_ok=True)
        manifest_path = destination / "run-manifest.json"
        source_metadata = {**prepared.metadata, "sha256": prepared.sha256}
        manifest = Manifest.open(
            manifest_path,
            job_id=job_id,
            fingerprint=fingerprint,
            source=source_metadata,
            config=config.fingerprint_dict(),
        )
        log_path = destination / "run.log"
        logger = RunLogger(log_path)
        logger(f"job {job_id} started")

        job_root = config.data_root / "jobs" / job_id
        work = job_root / "work"
        work.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            job_root / "job.json",
            {"schema_version": 1, "job_id": job_id, "fingerprint": fingerprint, "status": "running"},
        )

        public_source = destination / f"source{prepared.extension}"
        if not public_source.is_file() or sha256_file(public_source) != prepared.sha256:
            atomic_copy(prepared.path, public_source)
        if sha256_file(public_source) != prepared.sha256:
            raise RuntimeError("source copy failed byte-for-byte validation")

        analysis_dir = work / "analysis"
        analysis_wav = analysis_dir / "analysis.wav"
        analysis_fp = stage_fingerprint(
            "analysis",
            schema_version=1,
            inputs={"source": prepared.sha256},
            settings={"sample_rate": 16000, "channels": 1, "codec": "pcm_s16le"},
        )

        def run_analysis() -> tuple[list[Path], dict[str, object]]:
            make_analysis_wav(
                public_source,
                analysis_wav,
                ffmpeg_binary=config.runtime.ffmpeg,
                ffprobe_binary=config.runtime.ffprobe,
                logger=logger,
            )
            info = audio_info(analysis_wav, config.runtime.ffprobe)
            return [analysis_wav], info

        _stage(
            "analysis",
            analysis_dir,
            analysis_fp,
            manifest,
            logger,
            run_analysis,
            validator=lambda _: _validate_analysis(analysis_wav, config),
        )

        stt_dir = work / "stt"
        stt_fp = stage_fingerprint(
            "stt",
            schema_version=1,
            inputs={"analysis": sha256_file(analysis_wav)},
            settings={
                **dataclasses.asdict(config.stt),
                "device": config.runtime.device,
                "language": config.language,
            },
            models={"cohere": config.cohere.public_dict(), "silero_vad": "6.2.1"},
        )

        def run_stt() -> tuple[list[Path], dict[str, object]]:
            worker = Path(__file__).with_name("workers") / "cohere_worker.py"
            args = [
                str(config.cohere_python),
                str(worker),
                "--audio",
                str(analysis_wav),
                "--output-dir",
                str(stt_dir),
                "--model",
                str(config.cohere.resolved_path(config.data_root)),
                "--revision",
                config.cohere.revision,
                "--device",
                config.runtime.device,
                "--window-seconds",
                str(config.stt.window_seconds),
                "--overlap-seconds",
                str(config.stt.overlap_seconds),
                "--batch-size",
                str(config.stt.batch_size),
                "--vad-chunk-seconds",
                str(config.stt.vad_chunk_seconds),
                "--vad-padding-seconds",
                str(config.stt.vad_padding_seconds),
                "--vad-merge-gap-seconds",
                str(config.stt.vad_merge_gap_seconds),
                "--quarantine-max-seconds",
                str(config.stt.quarantine_max_seconds),
            ]
            from .util import run_command

            run_command(args, logger=logger, env={"PYTHONUNBUFFERED": "1"})
            report = _validate_stt(stt_dir)
            return _all_stage_files(stt_dir), report

        _stage(
            "stt",
            stt_dir,
            stt_fp,
            manifest,
            logger,
            run_stt,
            validator=lambda _: _validate_stt(stt_dir),
        )

        qwen_dir = work / "qwen"
        qwen_fp = stage_fingerprint(
            "qwen_alignment",
            schema_version=10,
            inputs={
                "analysis": sha256_file(analysis_wav),
                "transcript": sha256_file(stt_dir / "sentences.txt"),
                "cohere_windows": sha256_file(stt_dir / "stt-windows.json"),
            },
            settings={
                "max_batch_items": config.alignment.max_batch_items,
                "max_padded_seconds": config.alignment.max_padded_seconds,
                "device": config.runtime.device,
                "language": config.language,
            },
            models={
                "qwen_asr": config.qwen_asr.public_dict(),
                "qwen_aligner": config.qwen_aligner.public_dict(),
            },
        )

        def run_qwen() -> tuple[list[Path], dict[str, object]]:
            worker = Path(__file__).with_name("workers") / "qwen_worker.py"
            from .util import run_command

            run_command(
                [
                    str(config.qwen_python),
                    str(worker),
                    "--audio",
                    str(analysis_wav),
                    "--transcript",
                    str(stt_dir / "sentences.txt"),
                    "--chunks-dir",
                    str(stt_dir / "chunks"),
                    "--cohere-checkpoint",
                    str(stt_dir / "stt-windows.json"),
                    "--output-dir",
                    str(qwen_dir),
                    "--asr-model",
                    str(config.qwen_asr.resolved_path(config.data_root)),
                    "--asr-revision",
                    config.qwen_asr.revision,
                    "--aligner-model",
                    str(config.qwen_aligner.resolved_path(config.data_root)),
                    "--aligner-revision",
                    config.qwen_aligner.revision,
                    "--device",
                    config.runtime.device,
                    "--max-batch-items",
                    str(config.alignment.max_batch_items),
                    "--max-padded-seconds",
                    str(config.alignment.max_padded_seconds),
                ],
                logger=logger,
                env={"PYTHONUNBUFFERED": "1"},
            )
            report = _validate_qwen(qwen_dir)
            return _all_stage_files(qwen_dir), report

        _stage(
            "qwen_alignment",
            qwen_dir,
            qwen_fp,
            manifest,
            logger,
            run_qwen,
            validator=lambda _: _validate_qwen(qwen_dir),
        )

        mfa_dir = work / "mfa"
        mfa_fp = stage_fingerprint(
            "mfa_refinement",
            schema_version=1,
            inputs={
                "qwen_timings": sha256_file(qwen_dir / "qwen_word_timings.json"),
                "mfa_manifest": sha256_file(qwen_dir / "mfa_manifest.json"),
            },
            settings=dataclasses.asdict(config.alignment),
            models={"mfa": "3.4.1"},
        )

        def run_mfa() -> tuple[list[Path], dict[str, object]]:
            from .util import run_command

            textgrids = mfa_dir / "textgrids"
            textgrids.mkdir(parents=True, exist_ok=True)
            mfa_environment = {**config.mfa_subprocess_environment(), "PYTHONUNBUFFERED": "1"}
            run_command(
                [
                    str(config.mfa_binary),
                    "align",
                    str(qwen_dir / "mfa_corpus"),
                    config.alignment.mfa_dictionary,
                    config.alignment.mfa_acoustic_model,
                    str(textgrids),
                    "--g2p_model_path",
                    config.alignment.mfa_g2p_model,
                    "--fine_tune",
                    "--single_speaker",
                    "--num_jobs",
                    str(config.alignment.mfa_jobs),
                    "--temporary_directory",
                    str(mfa_dir / "temporary"),
                    "--clean",
                    "--overwrite",
                ],
                logger=logger,
                env=mfa_environment,
            )
            merge_worker = Path(__file__).with_name("workers") / "mfa_merge.py"
            run_command(
                [
                    str(config.mfa_python),
                    str(merge_worker),
                    "--output-dir",
                    str(mfa_dir),
                    "--qwen-dir",
                    str(qwen_dir),
                    "--textgrids-dir",
                    str(textgrids),
                    "--allow-qwen-fallback",
                    "--mfa-version",
                    "3.4.1",
                    "--mfa-acoustic-model",
                    config.alignment.mfa_acoustic_model,
                    "--mfa-dictionary",
                    config.alignment.mfa_dictionary,
                    "--mfa-g2p-model",
                    config.alignment.mfa_g2p_model,
                ],
                logger=logger,
                env=mfa_environment,
            )
            alignment = validate_word_timings(mfa_dir / "word_timings.json")
            return _all_stage_files(mfa_dir), _alignment_summary(alignment)

        _stage(
            "mfa_refinement",
            mfa_dir,
            mfa_fp,
            manifest,
            logger,
            run_mfa,
            validator=lambda _: validate_word_timings(mfa_dir / "word_timings.json"),
        )
        alignment = validate_word_timings(mfa_dir / "word_timings.json")
        manifest.value["alignment"] = _alignment_summary(alignment)
        if alignment["metadata"].get("mfa_coverage_warning"):
            manifest.warning(str(alignment["metadata"]["mfa_coverage_warning"]))
        manifest.save()

        render_dir = work / "render"
        render_fp = stage_fingerprint(
            "hybrid_render",
            schema_version=2,
            inputs={
                "source": prepared.sha256,
                "word_timings": sha256_file(mfa_dir / "word_timings.json"),
            },
            settings=dataclasses.asdict(config.renderer),
            models={"font": "Geist-590/OFL-1.1"},
        )

        def validate_render(_: Path) -> dict[str, object]:
            return validate_video(
                render_dir / "video.mp4",
                ffprobe_binary=config.runtime.ffprobe,
                width=config.renderer.width,
                height=config.renderer.height,
                fps=config.renderer.fps,
                expected_duration=float(alignment["metadata"]["audio_duration_seconds"]),
                require_audio=True,
            )

        def run_render() -> tuple[list[Path], dict[str, object]]:
            report = render_video(
                alignment=alignment,
                audio=public_source,
                output_dir=render_dir,
                output=render_dir / "video.mp4",
                config=config.renderer,
                ffmpeg_binary=config.runtime.ffmpeg,
                ffprobe_binary=config.runtime.ffprobe,
                browser_cache=config.browser_cache,
                fingerprint=render_fp,
                logger=logger,
            )
            return [render_dir / "video.mp4", render_dir / "layout-meta.json", render_dir / "layout.jsonl"], report

        _stage(
            "hybrid_render",
            render_dir,
            render_fp,
            manifest,
            logger,
            run_render,
            validator=validate_render,
        )

        full_video = destination / "video.mp4"
        transcript = destination / "transcript.txt"
        word_timings = destination / "word-timings.json"
        atomic_copy(render_dir / "video.mp4", full_video)
        atomic_copy(stt_dir / "transcript.txt", transcript)
        atomic_copy(mfa_dir / "word_timings.json", word_timings)
        validate_video(
            full_video,
            ffprobe_binary=config.runtime.ffprobe,
            width=config.renderer.width,
            height=config.renderer.height,
            fps=config.renderer.fps,
            expected_duration=float(alignment["metadata"]["audio_duration_seconds"]),
            require_audio=True,
        )
        validate_word_timings(word_timings)

        speech_dir = work / "speech-cut"
        speech_fp = stage_fingerprint(
            "speech_cut",
            schema_version=1,
            inputs={
                "video": sha256_file(render_dir / "video.mp4"),
                "vad": sha256_file(stt_dir / "vad-regions.json"),
                "word_timings": sha256_file(mfa_dir / "word_timings.json"),
            },
            settings={"threshold_seconds": config.speech_cut.threshold_seconds},
        )
        speech_result: dict[str, object]
        speech_cut_video: Path | None = None

        def validate_speech_checkpoint(directory: Path) -> dict[str, object]:
            result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
            status = result.get("status")
            if status == "no_cuts":
                if result.get("cut_count") != 0 or float(result.get("removed_seconds", -1)) != 0:
                    raise RuntimeError("invalid no-cuts checkpoint")
                if (directory / "video-speech-cut.mp4").exists():
                    raise RuntimeError("no-cuts checkpoint unexpectedly contains a video")
                return result
            if status != "created" or int(result.get("cut_count", 0)) < 1:
                raise RuntimeError("invalid speech-cut checkpoint status")
            expected_duration = float(result.get("output_duration_seconds", 0))
            validate_video(
                directory / "video-speech-cut.mp4",
                ffprobe_binary=config.runtime.ffprobe,
                width=config.renderer.width,
                height=config.renderer.height,
                fps=config.renderer.fps,
                expected_duration=expected_duration,
                require_audio=True,
            )
            return result

        try:
            checkpoint = valid_checkpoint(
                speech_dir,
                speech_fp,
                extra_validator=validate_speech_checkpoint,
            )
            if checkpoint:
                speech_result = json.loads((speech_dir / "result.json").read_text(encoding="utf-8"))
                logger("reusing speech-cut checkpoint")
                manifest.finish_stage("speech_cut", checkpoint, reused=True)
            else:
                started = manifest.start_stage("speech_cut", speech_fp)
                speech_dir.mkdir(parents=True, exist_ok=True)
                internal_cut = speech_dir / "video-speech-cut.mp4"
                speech_result = create_speech_cut(
                    full_video=render_dir / "video.mp4",
                    source_audio=public_source,
                    output=internal_cut,
                    alignment=alignment,
                    vad_file=stt_dir / "vad-regions.json",
                    threshold=config.speech_cut.threshold_seconds,
                    renderer=config.renderer,
                    ffmpeg_binary=config.runtime.ffmpeg,
                    ffprobe_binary=config.runtime.ffprobe,
                    logger=logger,
                )
                atomic_write_json(speech_dir / "result.json", speech_result)
                validate_speech_checkpoint(speech_dir)
                artifacts = [speech_dir / "result.json"]
                if speech_result["status"] == "created":
                    artifacts.append(internal_cut)
                checkpoint = write_checkpoint(
                    speech_dir,
                    speech_fp,
                    artifacts,
                    started_monotonic=started,
                    details=speech_result,
                )
                manifest.finish_stage("speech_cut", checkpoint)
            if speech_result["status"] == "created":
                speech_cut_video = destination / "video-speech-cut.mp4"
                atomic_copy(speech_dir / "video-speech-cut.mp4", speech_cut_video)
            else:
                (destination / "video-speech-cut.mp4").unlink(missing_ok=True)
        except Exception as error:
            safe_error = _safe_error_message(error, config.data_root, destination)
            speech_result = {
                "status": "failed",
                "threshold_seconds": config.speech_cut.threshold_seconds,
                "error": safe_error,
            }
            manifest.fail_stage("speech_cut", error)
            # A speech-cut failure is explicitly non-fatal once the canonical
            # full video has passed validation.
            manifest.value["status"] = "running"
            manifest.value["stages"]["speech_cut"]["error"] = safe_error
            manifest.warning(f"speech-cut failed; full video is valid: {safe_error}")
            (destination / "video-speech-cut.mp4").unlink(missing_ok=True)

        logger("all required artifacts validated; finalizing manifest")
        artifacts: dict[str, object] = {
            "video": artifact_record(full_video),
            "source": artifact_record(public_source),
            "transcript": artifact_record(transcript),
            "word_timings": artifact_record(word_timings),
            "run_log": artifact_record(log_path),
        }
        if speech_cut_video is not None:
            artifacts["speech_cut_video"] = artifact_record(speech_cut_video)
        manifest.complete(artifacts, speech_result)
        atomic_copy(manifest_path, job_root / "run-manifest.json")
        atomic_write_json(
            job_root / "job.json",
            {"schema_version": 1, "job_id": job_id, "fingerprint": fingerprint, "status": "complete"},
        )
        return PipelineResult(
            job_id=job_id,
            output_dir=destination,
            full_video=full_video,
            speech_cut_video=speech_cut_video,
            source_audio=public_source,
            transcript=transcript,
            word_timings=word_timings,
            manifest=manifest_path,
            log=log_path,
            warnings=tuple(manifest.value["warnings"]),
        )
    except Exception as error:
        if "manifest" in locals():
            manifest.value["status"] = "failed"
            manifest.value["error"] = _safe_error_message(
                error,
                config.data_root,
                destination if "destination" in locals() else config.data_root,
            )
            manifest.save()
        if "job_root" in locals():
            atomic_write_json(
                job_root / "job.json",
                {
                    "schema_version": 1,
                    "job_id": job_id,
                    "fingerprint": fingerprint,
                    "status": "failed",
                },
            )
        raise
    finally:
        prepared.cleanup()


def _stage(
    name: str,
    directory: Path,
    fingerprint: str,
    manifest: Manifest,
    logger: Callable[[str], None],
    runner: Callable[[], tuple[list[Path], dict[str, object]]],
    *,
    validator: Callable[[Path], object] | None = None,
) -> dict:
    checkpoint = valid_checkpoint(directory, fingerprint, extra_validator=validator)
    if checkpoint:
        logger(f"reusing {name} checkpoint")
        manifest.finish_stage(name, checkpoint, reused=True)
        return checkpoint
    started = manifest.start_stage(name, fingerprint)
    directory.mkdir(parents=True, exist_ok=True)
    try:
        artifacts, details = runner()
        if validator is not None:
            validator(directory)
        checkpoint = write_checkpoint(
            directory,
            fingerprint,
            artifacts,
            started_monotonic=started,
            details=details,
        )
        manifest.finish_stage(name, checkpoint)
        return checkpoint
    except Exception as error:
        manifest.fail_stage(name, error)
        raise


def _validate_destination(destination: Path, fingerprint: str) -> None:
    if not destination.exists() or not any(destination.iterdir()):
        return
    manifest = destination / "run-manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"output directory is not empty and has no run manifest: {destination}")
    try:
        existing = json.loads(manifest.read_text(encoding="utf-8"))
    except Exception as error:
        raise RuntimeError(f"output directory contains an unreadable run manifest: {destination}") from error
    if existing.get("fingerprint") != fingerprint:
        raise RuntimeError("output directory belongs to a different input, config, or model revision; refusing to overwrite")


def _validate_analysis(path: Path, config: RunConfig) -> dict[str, object]:
    info = audio_info(path, config.runtime.ffprobe)
    if info["sample_rate"] != 16000 or info["channels"] != 1:
        raise RuntimeError("analysis checkpoint is not 16 kHz mono")
    return info


def _validate_stt(directory: Path) -> dict[str, object]:
    result = json.loads((directory / "stt-result.json").read_text(encoding="utf-8"))
    windows = json.loads((directory / "stt-windows.json").read_text(encoding="utf-8"))
    vad = json.loads((directory / "vad-regions.json").read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or result.get("quality", {}).get("ok") is not True:
        raise RuntimeError("STT result failed schema or quality validation")
    schedule = windows.get("schedule", [])
    entries = windows.get("windows", {})
    if not schedule or any(str(item[0]) not in entries for item in schedule):
        raise RuntimeError("STT window checkpoint is incomplete")
    if not vad.get("regions"):
        raise RuntimeError("VAD checkpoint contains no speech regions")
    if not (directory / "transcript.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("transcript is empty")
    if not (directory / "sentences.txt").read_text(encoding="utf-8").strip():
        raise RuntimeError("sentence transcript is empty")
    return {
        "word_count": result["quality"]["word_count"],
        "window_count": result["window_count"],
        "speech_seconds": result["speech_seconds"],
    }


def _validate_qwen(directory: Path) -> dict[str, object]:
    qwen = json.loads((directory / "qwen_word_timings.json").read_text(encoding="utf-8"))
    manifest = json.loads((directory / "mfa_manifest.json").read_text(encoding="utf-8"))
    if qwen.get("schema_version") != 1 or not qwen.get("words"):
        raise RuntimeError("Qwen word timing output is invalid")
    if not manifest:
        raise RuntimeError("MFA corpus manifest is empty")
    corpus = directory / "mfa_corpus" / "speaker1"
    for utterance in manifest:
        identifier = utterance["utterance_id"]
        if not (corpus / f"{identifier}.wav").is_file() or not (corpus / f"{identifier}.lab").is_file():
            raise RuntimeError(f"MFA corpus is incomplete at {identifier}")
    return {
        "word_count": len(qwen["words"]),
        "segment_count": qwen["metadata"].get("alignment_segment_count"),
        "alignment_stats": qwen["metadata"].get("alignment_stats"),
    }


def _alignment_summary(alignment: dict) -> dict[str, object]:
    metadata = alignment["metadata"]
    return {
        "word_count": len(alignment["words"]),
        "sentence_count": len(alignment["sentences"]),
        "mfa_word_coverage": metadata.get("mfa_word_coverage"),
        "mfa_effective_words": metadata.get("mfa_effective_words"),
        "missing_textgrid_count": len(metadata.get("missing_textgrids", [])),
        "invalid_textgrid_count": len(metadata.get("invalid_textgrids", [])),
        "source_counts": metadata.get("source_counts", {}),
        "boundary_delta_median_seconds": metadata.get("median_qwen_mfa_boundary_delta_seconds"),
        "boundary_delta_p95_seconds": metadata.get("p95_qwen_mfa_boundary_delta_seconds"),
    }


def _all_stage_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file()
        and path.name != "checkpoint.json"
        and not path.name.startswith(".")
        and not path.name.endswith(".tmp")
        and "temporary" not in path.relative_to(directory).parts
    )


def _safe_error_message(error: BaseException, *private_roots: Path) -> str:
    return safe_exception_text(error, *private_roots)
