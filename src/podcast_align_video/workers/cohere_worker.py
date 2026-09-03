from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


SCHEMA_VERSION = 1
WINDOW_CHECKPOINT_SCHEMA = 1
VAD_CHECKPOINT_SCHEMA = 1


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def log(message: str) -> None:
    print(f"[cohere] {message}", flush=True)


def quality_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[^\W_]+(?:'[^\W_]+)?", text.casefold(), flags=re.UNICODE)
        if token
    ]


def repeated_ngram_stats(words: list[str]) -> dict[str, Any]:
    max_consecutive = 1
    max_ngram = ""
    max_count = 0
    for size in (3, 4, 5, 6):
        positions: dict[tuple[str, ...], set[int]] = {}
        for index in range(max(0, len(words) - size + 1)):
            positions.setdefault(tuple(words[index:index + size]), set()).add(index)
        for gram, starts in positions.items():
            for start in starts:
                count = 1
                while start + count * size in starts:
                    count += 1
                if count > max_consecutive:
                    max_consecutive = count
                    max_ngram = " ".join(gram)
                if count >= 2:
                    max_count = max(max_count, count)
    counts = Counter(tuple(words[index:index + 4]) for index in range(max(0, len(words) - 3)))
    dominant_gram, dominant_count = counts.most_common(1)[0] if counts else ((), 0)
    return {
        "max_consecutive_repeats": max_consecutive,
        "max_consecutive_ngram": max_ngram,
        "dominant_4gram": " ".join(dominant_gram),
        "dominant_4gram_count": dominant_count,
        "dominant_4gram_fraction": dominant_count / max(1, len(words)),
        "max_repeat_count": max_count,
    }


def transcript_quality_report(text: str, duration_seconds: float | None = None) -> dict[str, Any]:
    words = quality_tokens(text)
    repeats = repeated_ngram_stats(words)
    unique_word_ratio = len(set(words)) / max(1, len(words))
    reasons: list[str] = []
    if len(words) < 3:
        reasons.append("too_few_words")
    if len(words) >= 40 and (
        repeats["max_consecutive_repeats"] >= 8
        or (repeats["max_consecutive_repeats"] >= 4 and unique_word_ratio < 0.08)
    ):
        reasons.append("consecutive_ngram_loop")
    if len(words) >= 80 and repeats["dominant_4gram_count"] >= max(12, int(len(words) * 0.02)):
        reasons.append("dominant_ngram_loop")
    if len(words) >= 100 and unique_word_ratio < 0.08:
        reasons.append("low_vocabulary")
    words_per_second = len(words) / duration_seconds if duration_seconds and duration_seconds > 0 else None
    if words_per_second is not None and words_per_second > 5.5:
        reasons.append("implausibly_fast")
    return {
        "ok": not reasons,
        "word_count": len(words),
        "unique_word_ratio": unique_word_ratio,
        "words_per_second": words_per_second,
        "repetition": repeats,
        "reasons": reasons,
    }


def normalized_join_word(word: str) -> str:
    return re.sub(r"[^\w']", "", word.casefold(), flags=re.UNICODE)


def merge_overlapping_transcripts(left: str, right: str, max_words: int = 48) -> str:
    left_words = left.split()
    right_words = right.split()
    left_norm = [normalized_join_word(word) for word in left_words]
    right_norm = [normalized_join_word(word) for word in right_words]
    overlap = 0
    for size in range(min(max_words, len(left_words), len(right_words)), 1, -1):
        if left_norm[-size:] == right_norm[:size]:
            overlap = size
            break
    if overlap:
        right_words = right_words[overlap:]
    return " ".join(part for part in (" ".join(left_words), " ".join(right_words)) if part)


def full_audio_schedule(duration: float, window: float, overlap: float) -> list[tuple[int, float, float]]:
    result = []
    start = 0.0
    step = window - overlap
    while start < duration:
        end = min(duration, start + window)
        result.append((len(result), round(start, 3), round(end, 3)))
        if end >= duration:
            break
        start += step
    return result


def merge_speech_regions(
    regions: list[tuple[float, float]],
    duration: float,
    padding: float,
    merge_gap: float,
) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(regions):
        start = max(0.0, min(duration, float(start)))
        end = max(0.0, min(duration, float(end)))
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + merge_gap:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return [
        (round(max(0.0, start - padding), 3), round(min(duration, end + padding), 3))
        for start, end in merged
    ]


def speech_schedule(regions: list[tuple[float, float]], window: float, overlap: float) -> list[tuple[int, float, float]]:
    result = []
    for region_start, region_end in regions:
        start = region_start
        while start < region_end:
            end = min(region_end, start + window)
            result.append((len(result), round(start, 3), round(end, 3)))
            if end >= region_end:
                break
            start = end - overlap
    return result


def detect_speech_regions(audio: Path, output: Path, args: argparse.Namespace, audio_hash: str) -> list[tuple[float, float]]:
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad

    checkpoint = output / "vad-regions.json"
    settings = {
        "schema_version": VAD_CHECKPOINT_SCHEMA,
        "algorithm": "silero-vad-6.2.1",
        "audio_sha256": audio_hash,
        "chunk_seconds": args.vad_chunk_seconds,
        "padding_seconds": args.vad_padding_seconds,
        "merge_gap_seconds": args.vad_merge_gap_seconds,
    }
    if checkpoint.is_file():
        try:
            old = json.loads(checkpoint.read_text(encoding="utf-8"))
            if all(old.get(key) == value for key, value in settings.items()) and isinstance(old.get("regions"), list):
                regions = [(float(item[0]), float(item[1])) for item in old["regions"]]
                log(f"reusing {len(regions)} validated VAD regions")
                return regions
        except Exception:
            log("ignoring unreadable VAD checkpoint")

    info = sf.info(str(audio))
    sample_rate = int(info.samplerate)
    if sample_rate != 16000 or info.channels != 1:
        raise RuntimeError("Cohere worker requires the pipeline's 16 kHz mono analysis WAV")
    duration = float(info.duration)
    chunk_frames = max(sample_rate, int(round(args.vad_chunk_seconds * sample_rate)))
    model = load_silero_vad(onnx=False)
    model.to("cpu").eval()
    raw: list[tuple[float, float]] = []
    with sf.SoundFile(str(audio)) as handle:
        chunk_start = 0
        while chunk_start < len(handle):
            handle.seek(chunk_start)
            waveform = handle.read(min(chunk_frames, len(handle) - chunk_start), dtype="float32")
            if len(waveform) == 0:
                break
            timestamps = get_speech_timestamps(
                torch.from_numpy(waveform),
                model,
                sampling_rate=sample_rate,
                min_speech_duration_ms=250,
                min_silence_duration_ms=300,
                speech_pad_ms=150,
                return_seconds=False,
            )
            raw.extend(
                (
                    (chunk_start + int(item["start"])) / sample_rate,
                    (chunk_start + int(item["end"])) / sample_rate,
                )
                for item in timestamps
            )
            chunk_start += len(waveform)
    regions = merge_speech_regions(raw, duration, args.vad_padding_seconds, args.vad_merge_gap_seconds)
    if not regions:
        raise RuntimeError("Silero VAD found no speech in the source audio")
    speech_seconds = sum(end - start for start, end in regions)
    write_json(
        checkpoint,
        {
            **settings,
            "duration_seconds": duration,
            "speech_seconds": round(speech_seconds, 6),
            "regions": [[start, end] for start, end in regions],
        },
    )
    log(f"VAD complete: {len(regions)} regions, {speech_seconds:.1f}s candidate speech")
    del model
    gc.collect()
    return regions


def iter_windows(audio: Path, schedule: list[tuple[int, float, float]]):
    import soundfile as sf

    with sf.SoundFile(str(audio)) as handle:
        sample_rate = int(handle.samplerate)
        total = len(handle)
        for index, start, end in schedule:
            start_frame = max(0, min(total, int(round(start * sample_rate))))
            end_frame = max(start_frame, min(total, int(round(end * sample_rate))))
            handle.seek(start_frame)
            waveform = handle.read(end_frame - start_frame, dtype="float32")
            if len(waveform):
                actual_start = start_frame / sample_rate
                yield index, actual_start, actual_start + len(waveform) / sample_rate, waveform, sample_rate


def load_model(model_path: Path, device: str):
    import torch
    from transformers import AutoProcessor, CohereAsrForConditionalGeneration
    from transformers.utils import logging as transformers_logging

    transformers_logging.disable_progress_bar()

    if not model_path.is_dir():
        raise RuntimeError(f"Cohere model is missing: {model_path}")
    processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
    model = CohereAsrForConditionalGeneration.from_pretrained(
        str(model_path),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, processor


def transcribe_arrays(model: Any, processor: Any, arrays: list[Any], sample_rate: int, device: str) -> list[str]:
    import torch

    if not arrays:
        return []
    inputs = processor(
        audio=arrays,
        language="en",
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    moved = {}
    for key, value in inputs.items():
        if not hasattr(value, "to"):
            moved[key] = value
        elif key == "input_features":
            moved[key] = value.to(device=device, dtype=model.dtype)
        else:
            moved[key] = value.to(device=device)
    with torch.inference_mode():
        generated = model.generate(**moved, max_new_tokens=448)
    decoded = processor.batch_decode(generated, skip_special_tokens=True)
    if len(decoded) != len(arrays):
        raise RuntimeError(f"Cohere returned {len(decoded)} transcripts for {len(arrays)} windows")
    return [str(item).strip() for item in decoded]


def retry_bad_window(model: Any, processor: Any, waveform: Any, sample_rate: int, duration: float, device: str):
    midpoint = len(waveform) // 2
    overlap = max(1, int(round(1.5 * sample_rate)))
    pieces = [waveform[: min(len(waveform), midpoint + overlap)], waveform[max(0, midpoint - overlap):]]
    recovered = ""
    for text in transcribe_arrays(model, processor, pieces, sample_rate, device):
        recovered = merge_overlapping_transcripts(recovered, text)
    report = transcript_quality_report(recovered, duration)
    report["retry_strategy"] = "two_half_windows"
    if not recovered.strip() and duration <= 2.5:
        report.update({"ok": True, "reasons": [], "accepted_empty_short_window": True})
    if not report["ok"]:
        raise RuntimeError(f"Cohere quality check failed after retry: {report}")
    return recovered, report


def sentence_lines(text: str, max_words: int = 70) -> list[str]:
    normalized = " ".join(text.replace("\u200b", "").split())
    if not normalized:
        return []
    abbreviations = {
        "mr.", "mrs.", "ms.", "dr.", "prof.", "sr.", "jr.", "st.", "vs.",
        "etc.", "e.g.", "i.e.", "u.s.", "a.m.", "p.m.",
    }
    boundaries = []
    for index, char in enumerate(normalized):
        if char not in ".?!" or (char == "." and index + 1 < len(normalized) and normalized[index + 1] == "."):
            continue
        cursor = index + 1
        while cursor < len(normalized) and normalized[cursor].isspace():
            cursor += 1
        if cursor >= len(normalized):
            boundaries.append(len(normalized))
            continue
        if normalized[cursor] not in "\"'([{ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789":
            continue
        if char == ".":
            before = re.search(r"[A-Za-z.]+$", normalized[: index + 1])
            token = before.group(0).casefold() if before else ""
            if token in abbreviations:
                continue
            if index > 0 and index + 1 < len(normalized) and normalized[index - 1].isdigit() and normalized[index + 1].isdigit():
                continue
            if re.search(r"\.(?:com|org|net|io|ai|co|tv)\b", normalized[max(0, index - 18): index + 12], re.I):
                continue
        boundaries.append(cursor)
    raw = []
    start = 0
    for boundary in boundaries:
        piece = normalized[start:boundary].strip()
        if piece:
            raw.append(piece)
        start = boundary
    if start < len(normalized):
        raw.append(normalized[start:].strip())
    result = []
    for line in raw:
        words = line.split()
        while len(words) > max_words:
            split_at = max_words
            for candidate in range(max_words, max(20, max_words // 2), -1):
                if words[candidate - 1][-1:] in ".?!,;:":
                    split_at = candidate
                    break
            result.append(" ".join(words[:split_at]))
            words = words[split_at:]
        if words:
            result.append(" ".join(words))
    return result


def write_transcript_assets(output: Path, text: str) -> list[Path]:
    lines = sentence_lines(text)
    if not lines:
        raise RuntimeError("Cohere returned an empty transcript")
    transcript = output / "transcript.txt"
    sentences = output / "sentences.txt"
    write_text(transcript, " ".join(lines) + "\n")
    write_text(sentences, "\n".join(lines) + "\n")
    chunks = output / "chunks"
    chunks.mkdir(exist_ok=True)
    for stale in chunks.glob("chunk_*.txt"):
        stale.unlink()
    chunk_index = 0
    current: list[str] = []
    current_words = 0
    for line in lines:
        count = len(line.split())
        if current and current_words + count > 300:
            write_text(chunks / f"chunk_{chunk_index:03d}.txt", "\n".join(current) + "\n")
            chunk_index += 1
            current, current_words = [], 0
        current.append(line)
        current_words += count
    if current:
        write_text(chunks / f"chunk_{chunk_index:03d}.txt", "\n".join(current) + "\n")
        chunk_index += 1
    report = output / "transcript-report.json"
    write_json(
        report,
        {
            "schema_version": 1,
            "source": "CohereLabs/cohere-transcribe-03-2026",
            "language": "en",
            "sentence_count": len(lines),
            "word_count": sum(len(line.split()) for line in lines),
            "chunk_count": chunk_index,
        },
    )
    return [transcript, sentences, report, *sorted(chunks.glob("chunk_*.txt"))]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--window-seconds", type=float, default=28.0)
    parser.add_argument("--overlap-seconds", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--vad-chunk-seconds", type=float, default=120.0)
    parser.add_argument("--vad-padding-seconds", type=float, default=0.35)
    parser.add_argument("--vad-merge-gap-seconds", type=float, default=0.9)
    parser.add_argument("--quarantine-max-seconds", type=float, default=12.0)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import soundfile as sf

    info = sf.info(str(args.audio))
    duration = float(info.duration)
    audio_hash = sha256(args.audio)
    regions = detect_speech_regions(args.audio, args.output_dir, args, audio_hash)
    schedule = speech_schedule(regions, args.window_seconds, args.overlap_seconds)
    if not schedule:
        raise RuntimeError("VAD produced no transcription windows")
    schedule_hash = stable_hash(schedule)
    checkpoint_path = args.output_dir / "stt-windows.json"
    meta = {
        "schema_version": WINDOW_CHECKPOINT_SCHEMA,
        "algorithm": "cohere-native-windowed-v1",
        "audio_sha256": audio_hash,
        "model_revision": args.revision,
        "window_seconds": args.window_seconds,
        "overlap_seconds": args.overlap_seconds,
        "batch_size": args.batch_size,
        "quarantine_max_seconds": args.quarantine_max_seconds,
        "device": args.device,
        "language": "en",
        "schedule_hash": schedule_hash,
    }
    entries: dict[str, dict[str, Any]] = {}
    if checkpoint_path.is_file():
        try:
            old = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if all(old.get(key) == value for key, value in meta.items()) and isinstance(old.get("windows"), dict):
                schedule_by_key = {str(index): (index, start, end) for index, start, end in schedule}
                for key, expected in schedule_by_key.items():
                    entry = old["windows"].get(key)
                    if not isinstance(entry, dict):
                        continue
                    quality = entry.get("quality")
                    try:
                        valid = (
                            entry.get("index") == expected[0]
                            and abs(float(entry.get("start")) - expected[1]) <= 0.002
                            and abs(float(entry.get("end")) - expected[2]) <= 0.002
                            and isinstance(entry.get("text"), str)
                            and isinstance(quality, dict)
                            and quality.get("ok") is True
                        )
                    except (TypeError, ValueError):
                        valid = False
                    if valid:
                        entries[key] = entry
                log(f"resuming {len(entries)}/{len(schedule)} validated transcription windows")
        except Exception:
            log("ignoring unreadable STT checkpoint")

    def save_checkpoint() -> None:
        write_json(
            checkpoint_path,
            {
                **meta,
                "schedule": [[index, start, end] for index, start, end in schedule],
                "windows": {key: entries[key] for key in sorted(entries, key=int)},
            },
        )

    started = time.monotonic()
    if not all(str(index) in entries for index, _, _ in schedule):
        model = processor = None
        try:
            log("loading native CohereAsrForConditionalGeneration")
            model, processor = load_model(args.model, args.device)
            pending = []

            def process_pending() -> None:
                if not pending:
                    return
                texts = transcribe_arrays(model, processor, [item[3] for item in pending], pending[0][4], args.device)
                for window, text in zip(pending, texts):
                    index, start, end, waveform, sample_rate = window
                    window_duration = end - start
                    quality = transcript_quality_report(text, window_duration)
                    if not quality["ok"]:
                        if quality["reasons"] == ["too_few_words"] and text.strip() and window_duration <= 10:
                            quality.update({"ok": True, "reasons": [], "accepted_short_utterance": True})
                        elif quality["reasons"] == ["too_few_words"] and not text.strip() and window_duration <= 2.5:
                            quality.update({"ok": True, "reasons": [], "accepted_empty_short_window": True})
                        else:
                            log(f"quality retry for window {index}: {quality['reasons']}")
                            try:
                                text, quality = retry_bad_window(
                                    model, processor, waveform, sample_rate, window_duration, args.device
                                )
                            except RuntimeError as error:
                                allowed = {
                                    "too_few_words", "consecutive_ngram_loop", "dominant_ngram_loop",
                                    "low_vocabulary", "implausibly_fast",
                                }
                                if window_duration > args.quarantine_max_seconds or not set(quality["reasons"]).issubset(allowed):
                                    raise
                                text = ""
                                quality = transcript_quality_report(text, window_duration)
                                quality.update(
                                    {"ok": True, "reasons": [], "quarantined": True, "error": str(error)[:500]}
                                )
                    entries[str(index)] = {
                        "index": index,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": text,
                        "quality": quality,
                    }
                    save_checkpoint()
                    log(f"saved window {index} ({quality['word_count']} words)")
                pending.clear()

            for window in iter_windows(args.audio, schedule):
                if str(window[0]) in entries:
                    continue
                pending.append(window)
                if len(pending) >= args.batch_size:
                    process_pending()
            process_pending()
        finally:
            del processor, model
            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass

    ordered = [entries[str(index)] for index, _, _ in schedule if str(index) in entries]
    if len(ordered) != len(schedule):
        raise RuntimeError(f"STT checkpoint incomplete: {len(ordered)}/{len(schedule)}")
    text = ""
    for entry in ordered:
        text = merge_overlapping_transcripts(text, str(entry["text"]))
    quality = transcript_quality_report(text, duration)
    if not quality["ok"]:
        raise RuntimeError(f"merged Cohere transcript failed quality validation: {quality}")
    assets = write_transcript_assets(args.output_dir, text)
    result = args.output_dir / "stt-result.json"
    write_json(
        result,
        {
            "schema_version": SCHEMA_VERSION,
            "algorithm": "cohere-native-windowed-v1",
            "audio_sha256": audio_hash,
            "model_revision": args.revision,
            "duration_seconds": duration,
            "speech_seconds": round(sum(end - start for start, end in regions), 6),
            "window_count": len(schedule),
            "quality": quality,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "artifacts": [path.name for path in assets],
        },
    )
    log(f"complete: {quality['word_count']} words")


if __name__ == "__main__":
    main()
