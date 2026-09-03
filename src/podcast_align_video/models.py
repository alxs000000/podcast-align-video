from __future__ import annotations

import json
from pathlib import Path

from huggingface_hub import snapshot_download

from .config import RunConfig
from .util import atomic_write_json, run_command, utc_now


MODEL_MARKER = ".podcast-align-video-model.json"
WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth"}


def fetch_models(config: RunConfig) -> None:
    specs = [config.cohere, config.qwen_asr, config.qwen_aligner]
    print("Models to fetch:")
    for spec in specs:
        print(f"- {spec.id}")
        print(f"  revision: {spec.revision}")
        print(f"  license/model card: {spec.license_url}")
        print(f"  approximate size: {spec.size_gib:.1f} GiB")
    print("Cohere access is gated. This command does not accept terms or grant access on your behalf.")
    print("Use your own approved Hugging Face token (for example via HF_TOKEN) before continuing.")

    for spec in specs:
        destination = spec.resolved_path(config.data_root)
        if spec.path is not None:
            if not _model_directory_complete(destination, spec.id):
                raise RuntimeError(f"configured local model path is incomplete: {destination}")
            print(f"Using configured local model: {spec.id} -> {destination}")
            continue
        if _model_revision_matches(destination, spec.id, spec.revision):
            print(f"Using previously fetched model: {spec.id}@{spec.revision}")
            continue
        destination.mkdir(parents=True, exist_ok=True)
        print(f"Fetching {spec.id}@{spec.revision} ...", flush=True)
        try:
            snapshot_download(
                repo_id=spec.id,
                revision=spec.revision,
                local_dir=destination,
                local_dir_use_symlinks=False,
            )
        except Exception as error:
            if spec is config.cohere:
                raise RuntimeError(
                    "Cohere download failed. Confirm that you accepted the model terms and that your own "
                    "Hugging Face token can access the gated repository."
                ) from error
            raise
        inventory = _model_inventory(destination)
        if not _model_directory_complete(destination, spec.id, inventory):
            raise RuntimeError(f"downloaded model snapshot is incomplete: {spec.id}@{spec.revision}")
        atomic_write_json(
            destination / MODEL_MARKER,
            {
                "schema_version": 1,
                "id": spec.id,
                "revision": spec.revision,
                "fetched_at": utc_now(),
                "files": inventory,
            },
        )

    if not config.mfa_binary.is_file():
        raise RuntimeError("MFA environment is missing; run scripts/setup.sh before models fetch")
    for kind, model in (
        ("dictionary", config.alignment.mfa_dictionary),
        ("acoustic", config.alignment.mfa_acoustic_model),
        ("g2p", config.alignment.mfa_g2p_model),
    ):
        print(f"Fetching MFA {kind} model {model} ...", flush=True)
        run_command(
            [str(config.mfa_binary), "model", "download", kind, model],
            env=config.mfa_subprocess_environment(),
        )


def _model_revision_matches(destination: Path, model_id: str, revision: str) -> bool:
    try:
        marker = json.loads((destination / MODEL_MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return (
        marker.get("schema_version") == 1
        and marker.get("id") == model_id
        and marker.get("revision") == revision
        and _model_directory_complete(destination, model_id, marker.get("files"))
    )


def _model_inventory(destination: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(destination.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(destination)
        if (
            relative.name == MODEL_MARKER
            or relative.suffix == ".pyc"
            or ".cache" in relative.parts
            or "__pycache__" in relative.parts
        ):
            continue
        records.append({"path": relative.as_posix(), "size_bytes": path.stat().st_size})
    return records


def _model_directory_complete(
    destination: Path,
    model_id: str,
    expected_inventory: object = None,
) -> bool:
    try:
        config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    if not isinstance(config, dict):
        return False
    weights = [
        path
        for path in destination.rglob("*")
        if path.is_file() and path.suffix.casefold() in WEIGHT_SUFFIXES and path.stat().st_size > 0
    ]
    if not weights:
        return False
    lowered = model_id.casefold()
    if "cohere" in lowered:
        required = ["preprocessor_config.json", "tokenizer_config.json"]
        alternatives = [("tokenizer.json", "tokenizer.model")]
    else:
        required = ["preprocessor_config.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
        alternatives = []
    if any(not (destination / name).is_file() for name in required):
        return False
    if any(not any((destination / name).is_file() for name in names) for names in alternatives):
        return False
    for index_path in destination.glob("*.index.json"):
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
            shards = set(index.get("weight_map", {}).values())
        except (OSError, ValueError, TypeError, AttributeError):
            return False
        if not shards or any(not (destination / str(shard)).is_file() for shard in shards):
            return False
    if expected_inventory is None:
        return True
    if not isinstance(expected_inventory, list) or not expected_inventory:
        return False
    root = destination.resolve()
    for record in expected_inventory:
        if not isinstance(record, dict):
            return False
        relative = record.get("path")
        size = record.get("size_bytes")
        if not isinstance(relative, str) or not isinstance(size, int) or size < 0:
            return False
        path = (destination / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != size:
            return False
    return True
