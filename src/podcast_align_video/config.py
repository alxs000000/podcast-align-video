from __future__ import annotations

import dataclasses
import math
import os
import tomllib
from pathlib import Path
from typing import Any, Mapping


COHERE_REVISION = "b1eacc2686a3d08ceaae5f24a88b1d519620bc09"
QWEN_ASR_REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
QWEN_ALIGNER_REVISION = "c7cbfc2048c462b0d63a45797104fc9db3ad62b7"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    id: str
    revision: str
    license_url: str
    size_gib: float
    path: Path | None = None

    def resolved_path(self, data_root: Path) -> Path:
        if self.path is not None:
            return self.path.expanduser().resolve()
        return data_root / "models" / self.id.replace("/", "--")

    def public_dict(self) -> dict[str, object]:
        return {"id": self.id, "revision": self.revision}


@dataclasses.dataclass(frozen=True)
class RuntimeConfig:
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    device: str = "cuda:0"


@dataclasses.dataclass(frozen=True)
class STTConfig:
    window_seconds: float = 28.0
    overlap_seconds: float = 4.0
    batch_size: int = 2
    vad_chunk_seconds: float = 120.0
    vad_padding_seconds: float = 0.35
    vad_merge_gap_seconds: float = 0.9
    quarantine_max_seconds: float = 12.0


@dataclasses.dataclass(frozen=True)
class AlignmentConfig:
    max_batch_items: int = 4
    max_padded_seconds: float = 80.0
    mfa_acoustic_model: str = "english_mfa"
    mfa_dictionary: str = "english_mfa"
    mfa_g2p_model: str = "english_us_mfa"
    mfa_jobs: int = 2


@dataclasses.dataclass(frozen=True)
class RendererConfig:
    width: int = 1920
    height: int = 1080
    fps: int = 30
    segment_seconds: float = 1800.0
    encoder: str = "libx264"
    preset: str = "veryfast"
    crf: int = 20
    pixel_format: str = "yuv420p"
    audio_bitrate: str = "192k"


@dataclasses.dataclass(frozen=True)
class SpeechCutConfig:
    threshold_seconds: float = 5.0


@dataclasses.dataclass(frozen=True)
class RunConfig:
    schema_version: int = 1
    language: str = "en"
    data_root: Path = Path("~/.local/share/podcast-align-video")
    runtime: RuntimeConfig = dataclasses.field(default_factory=RuntimeConfig)
    cohere: ModelSpec = dataclasses.field(
        default_factory=lambda: ModelSpec(
            "CohereLabs/cohere-transcribe-03-2026",
            COHERE_REVISION,
            "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026",
            3.9,
        )
    )
    qwen_asr: ModelSpec = dataclasses.field(
        default_factory=lambda: ModelSpec(
            "Qwen/Qwen3-ASR-0.6B",
            QWEN_ASR_REVISION,
            "https://huggingface.co/Qwen/Qwen3-ASR-0.6B",
            1.8,
        )
    )
    qwen_aligner: ModelSpec = dataclasses.field(
        default_factory=lambda: ModelSpec(
            "Qwen/Qwen3-ForcedAligner-0.6B",
            QWEN_ALIGNER_REVISION,
            "https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B",
            1.8,
        )
    )
    stt: STTConfig = dataclasses.field(default_factory=STTConfig)
    alignment: AlignmentConfig = dataclasses.field(default_factory=AlignmentConfig)
    renderer: RendererConfig = dataclasses.field(default_factory=RendererConfig)
    speech_cut: SpeechCutConfig = dataclasses.field(default_factory=SpeechCutConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_root", self.data_root.expanduser().resolve())
        if self.schema_version != 1:
            raise ValueError("config schema_version must be 1")
        if self.language != "en":
            raise ValueError("v0.1 supports English only (language must be 'en')")
        device_parts = self.runtime.device.split(":", 1)
        if len(device_parts) != 2 or device_parts[0] != "cuda" or not device_parts[1].isdigit():
            raise ValueError("v0.1 requires an explicit CUDA device such as cuda:0")
        if not self.runtime.ffmpeg.strip() or not self.runtime.ffprobe.strip():
            raise ValueError("ffmpeg and ffprobe commands must not be empty")
        if not math.isfinite(self.speech_cut.threshold_seconds) or self.speech_cut.threshold_seconds <= 0:
            raise ValueError("speech-cut threshold must be a positive number")
        if (
            not math.isfinite(self.stt.window_seconds)
            or not math.isfinite(self.stt.overlap_seconds)
            or self.stt.window_seconds <= 0
            or not 0 <= self.stt.overlap_seconds < self.stt.window_seconds
        ):
            raise ValueError("STT overlap must be non-negative and smaller than the window")
        if (
            self.stt.batch_size < 1
            or not math.isfinite(self.stt.vad_chunk_seconds)
            or self.stt.vad_chunk_seconds <= 0
        ):
            raise ValueError("STT batch size and VAD chunk duration must be positive")
        nonnegative_stt = (
            self.stt.vad_padding_seconds,
            self.stt.vad_merge_gap_seconds,
            self.stt.quarantine_max_seconds,
        )
        if any(not math.isfinite(value) or value < 0 for value in nonnegative_stt):
            raise ValueError("STT VAD padding, merge gap, and quarantine duration must be non-negative")
        if (
            self.alignment.max_batch_items < 1
            or not math.isfinite(self.alignment.max_padded_seconds)
            or self.alignment.max_padded_seconds <= 0
            or self.alignment.mfa_jobs < 1
        ):
            raise ValueError("alignment batch limits and MFA job count must be positive")
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (self.renderer.width, self.renderer.height, self.renderer.fps)
        ):
            raise ValueError("renderer dimensions and fps must be positive")
        if not math.isfinite(self.renderer.segment_seconds) or self.renderer.segment_seconds <= 0:
            raise ValueError("renderer segment_seconds must be positive")
        if self.renderer.encoder not in {"libx264", "h264_nvenc"}:
            raise ValueError("renderer encoder must be libx264 or h264_nvenc")
        if not 0 <= self.renderer.crf <= 51:
            raise ValueError("renderer CRF/CQ must be between 0 and 51")
        if not self.renderer.preset.strip() or not self.renderer.pixel_format.strip() or not self.renderer.audio_bitrate.strip():
            raise ValueError("renderer preset, pixel format, and audio bitrate must not be empty")
        for spec in (self.cohere, self.qwen_asr, self.qwen_aligner):
            if not spec.id.strip() or not spec.revision.strip() or not spec.license_url.startswith("https://"):
                raise ValueError("model ID, immutable revision, and HTTPS license URL are required")
            if not math.isfinite(spec.size_gib) or spec.size_gib <= 0:
                raise ValueError("model size must be positive")

    @property
    def cohere_python(self) -> Path:
        return self.data_root / "envs" / "cohere" / "bin" / "python"

    @property
    def qwen_python(self) -> Path:
        return self.data_root / "envs" / "qwen" / "bin" / "python"

    @property
    def mfa_binary(self) -> Path:
        return self.mfa_environment / "bin" / "mfa"

    @property
    def mfa_python(self) -> Path:
        return self.mfa_environment / "bin" / "python"

    @property
    def mfa_environment(self) -> Path:
        location_file = self.data_root / "envs" / "mfa.location"
        try:
            configured = Path(location_file.read_text(encoding="utf-8").strip()).expanduser()
        except OSError:
            configured = self.data_root / "envs" / "mfa"
        if not configured.is_absolute():
            raise ValueError(f"MFA environment location must be absolute: {location_file}")
        return configured.resolve()

    @property
    def mfa_root(self) -> Path:
        return self.data_root / "mfa-data"

    def mfa_subprocess_environment(self) -> dict[str, str]:
        executable_path = str(self.mfa_environment / "bin")
        inherited_path = os.environ.get("PATH", "")
        return {
            "MFA_ROOT_DIR": str(self.mfa_root),
            "PATH": f"{executable_path}{os.pathsep}{inherited_path}" if inherited_path else executable_path,
        }

    @property
    def browser_cache(self) -> Path:
        location_file = self.data_root / "browser-cache.location"
        try:
            configured = Path(location_file.read_text(encoding="utf-8").strip()).expanduser()
        except OSError:
            configured = self.data_root / "browser-cache"
        if not configured.is_absolute():
            raise ValueError(f"browser cache location must be absolute: {location_file}")
        return configured.resolve()

    @classmethod
    def from_toml(cls, path: Path) -> "RunConfig":
        with path.expanduser().open("rb") as handle:
            raw = tomllib.load(handle)
        return cls.from_mapping(raw, base_dir=path.expanduser().resolve().parent)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], base_dir: Path | None = None) -> "RunConfig":
        known_top_level = {
            "schema_version", "language", "data_root", "runtime", "models",
            "stt", "alignment", "renderer", "speech_cut",
        }
        unknown_top_level = set(raw) - known_top_level
        if unknown_top_level:
            raise ValueError(f"unknown config keys: {', '.join(sorted(unknown_top_level))}")
        model_tables = raw.get("models", {})
        if not isinstance(model_tables, Mapping):
            raise ValueError("models configuration must be a table")
        unknown_models = set(model_tables) - {"cohere", "qwen_asr", "qwen_aligner"}
        if unknown_models:
            raise ValueError(f"unknown model keys: {', '.join(sorted(unknown_models))}")
        defaults = cls()

        def path_value(value: object) -> Path | None:
            if value in (None, ""):
                return None
            expanded = Path(os.path.expandvars(str(value))).expanduser()
            if not expanded.is_absolute() and base_dir is not None:
                expanded = base_dir / expanded
            return expanded.resolve()

        def model(name: str, original: ModelSpec) -> ModelSpec:
            raw_values = model_tables.get(name, {})
            if not isinstance(raw_values, Mapping):
                raise ValueError(f"models.{name} configuration must be a table")
            allowed = {"id", "revision", "license_url", "size_gib", "path"}
            unknown = set(raw_values) - allowed
            if unknown:
                raise ValueError(f"unknown models.{name} keys: {', '.join(sorted(unknown))}")
            values = dict(raw_values)
            return ModelSpec(
                id=str(values.get("id", original.id)),
                revision=str(values.get("revision", original.revision)),
                license_url=str(values.get("license_url", original.license_url)),
                size_gib=float(values.get("size_gib", original.size_gib)),
                path=path_value(values.get("path", original.path)),
            )

        runtime = _dataclass_from_mapping(RuntimeConfig, defaults.runtime, raw.get("runtime", {}))
        stt = _dataclass_from_mapping(STTConfig, defaults.stt, raw.get("stt", {}))
        alignment = _dataclass_from_mapping(AlignmentConfig, defaults.alignment, raw.get("alignment", {}))
        renderer = _dataclass_from_mapping(RendererConfig, defaults.renderer, raw.get("renderer", {}))
        speech_cut = _dataclass_from_mapping(SpeechCutConfig, defaults.speech_cut, raw.get("speech_cut", {}))
        data_root_value = raw.get("data_root", defaults.data_root)
        data_root = path_value(data_root_value)
        assert data_root is not None
        return cls(
            schema_version=int(raw.get("schema_version", defaults.schema_version)),
            language=str(raw.get("language", defaults.language)),
            data_root=data_root,
            runtime=runtime,
            cohere=model("cohere", defaults.cohere),
            qwen_asr=model("qwen_asr", defaults.qwen_asr),
            qwen_aligner=model("qwen_aligner", defaults.qwen_aligner),
            stt=stt,
            alignment=alignment,
            renderer=renderer,
            speech_cut=speech_cut,
        )

    def with_overrides(self, *, device: str | None = None, silence_threshold: float | None = None) -> "RunConfig":
        runtime = dataclasses.replace(self.runtime, device=device) if device else self.runtime
        speech_cut = (
            dataclasses.replace(self.speech_cut, threshold_seconds=float(silence_threshold))
            if silence_threshold is not None
            else self.speech_cut
        )
        return dataclasses.replace(self, runtime=runtime, speech_cut=speech_cut)

    def fingerprint_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "language": self.language,
            # Executable paths are deliberately omitted because manifests are
            # share-safe. The selected CUDA device is output-relevant and safe.
            "runtime": {"device": self.runtime.device},
            "models": {
                "cohere": self.cohere.public_dict(),
                "qwen_asr": self.qwen_asr.public_dict(),
                "qwen_aligner": self.qwen_aligner.public_dict(),
                "mfa": {
                    "version": "3.4.1",
                    "acoustic": self.alignment.mfa_acoustic_model,
                    "dictionary": self.alignment.mfa_dictionary,
                    "g2p": self.alignment.mfa_g2p_model,
                },
                "silero_vad": "6.2.1",
            },
            "stt": dataclasses.asdict(self.stt),
            "alignment": dataclasses.asdict(self.alignment),
            "renderer": dataclasses.asdict(self.renderer),
            "speech_cut": dataclasses.asdict(self.speech_cut),
        }


def _dataclass_from_mapping(cls: type[Any], original: Any, values: object) -> Any:
    if not isinstance(values, Mapping):
        raise ValueError(f"{cls.__name__} configuration must be a table")
    known = {field.name for field in dataclasses.fields(cls)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {', '.join(sorted(unknown))}")
    merged = {field.name: getattr(original, field.name) for field in dataclasses.fields(cls)}
    merged.update(values)
    return cls(**merged)


def load_config(config: RunConfig | Path | None) -> RunConfig:
    if isinstance(config, RunConfig):
        return config
    if config is not None:
        return RunConfig.from_toml(Path(config))
    env_path = os.environ.get("PODCAST_ALIGN_VIDEO_CONFIG")
    if env_path:
        return RunConfig.from_toml(Path(env_path))
    return RunConfig()
