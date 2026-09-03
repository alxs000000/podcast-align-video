from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

from .api import run
from .config import RunConfig, load_config
from .doctor import format_checks, run_checks
from .models import fetch_models
from .util import directory_size, ensure_job_id, human_bytes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="podcast-align-video")
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("models", help="manage pinned model files")
    model_commands = models.add_subparsers(dest="models_command", required=True)
    fetch = model_commands.add_parser("fetch", help="fetch pinned Hugging Face and MFA models")
    fetch.add_argument("--config", type=Path)

    doctor = commands.add_parser("doctor", help="verify system tools, environments, models, and GPU")
    doctor.add_argument("--config", type=Path)

    run_parser = commands.add_parser("run", help="create aligned hybrid video artifacts")
    run_parser.add_argument("source", help="local audio path or one public YouTube video URL")
    run_parser.add_argument("--output-dir", type=Path)
    run_parser.add_argument("--config", type=Path)
    run_parser.add_argument("--silence-threshold", type=_positive_float)
    run_parser.add_argument("--device")

    clean = commands.add_parser("clean", help="remove retained work for one completed job")
    clean.add_argument("job_id")
    clean.add_argument("--config", type=Path)
    clean.add_argument("--yes", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        if args.command == "models":
            fetch_models(load_config(args.config))
            return
        if args.command == "doctor":
            checks = run_checks(load_config(args.config))
            print(format_checks(checks))
            if any(check.required and not check.ok for check in checks):
                raise SystemExit(1)
            return
        if args.command == "run":
            config = load_config(args.config).with_overrides(
                device=args.device,
                silence_threshold=args.silence_threshold,
            )
            result = run(args.source, output_dir=args.output_dir, config=config)
            print(
                json.dumps(
                    {
                        "job_id": result.job_id,
                        "output_directory": str(result.output_directory),
                        "video": str(result.full_video),
                        "speech_cut_video": str(result.speech_cut_video) if result.speech_cut_video else None,
                        "source": str(result.source_audio),
                        "transcript": str(result.transcript),
                        "word_timings": str(result.word_timings),
                        "manifest": str(result.manifest),
                        "log": str(result.log),
                        "warnings": result.warnings,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if args.command == "clean":
            _clean(args.job_id, load_config(args.config), confirmed=args.yes)
            return
        raise AssertionError("unhandled command")
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except SystemExit:
        raise
    except Exception as error:
        raise SystemExit(f"error: {error}") from error


def _clean(job_id: str, config: RunConfig, *, confirmed: bool) -> None:
    job_id = ensure_job_id(job_id)
    jobs_root = (config.data_root / "jobs").resolve()
    job_root = (jobs_root / job_id).resolve()
    if not job_root.is_relative_to(jobs_root) or not job_root.is_dir():
        raise ValueError(f"unknown job: {job_id}")
    job_state = job_root / "job.json"
    run_manifest = job_root / "run-manifest.json"
    if not job_state.is_file() or not run_manifest.is_file():
        raise RuntimeError("clean only accepts jobs with durable completion metadata")
    state = json.loads(job_state.read_text(encoding="utf-8"))
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    if state.get("status") != "complete" or manifest.get("status") != "complete":
        raise RuntimeError("clean only removes work from completed jobs")
    work = job_root / "work"
    size = directory_size(work)
    print(f"Job: {job_id}")
    print(f"Work to remove: {human_bytes(size)} ({work})")
    print("Published artifacts and model files will not be removed.")
    if not confirmed:
        print("Dry run only. Add --yes to remove this completed job's work directory.")
        return
    if work.is_dir():
        shutil.rmtree(work)
    print(f"Removed {human_bytes(size)} of retained work. This removal is not recoverable by the tool.")


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive number")
    return parsed
