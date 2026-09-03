from __future__ import annotations

import math
import statistics
import unicodedata
from typing import Sequence


def normalize_word(value: object) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKC", str(value)).casefold()
        if character == "'" or unicodedata.category(character).startswith(("L", "N"))
    )


def quality_report(prediction: dict, gold: dict) -> dict[str, object]:
    predicted_words = prediction.get("words")
    gold_words = gold.get("words")
    if not isinstance(predicted_words, list) or not predicted_words:
        raise ValueError("prediction contains no words")
    if not isinstance(gold_words, list) or not gold_words:
        raise ValueError("gold reference contains no words")

    predicted_tokens = [normalize_word(item.get("clean") or item.get("text")) for item in predicted_words]
    gold_tokens = [normalize_word(item.get("text")) for item in gold_words]
    if not all(predicted_tokens) or not all(gold_tokens):
        raise ValueError("prediction and gold words must normalize to non-empty tokens")
    edit_distance, matched_pairs = _align_tokens(predicted_tokens, gold_tokens)
    boundary_errors: list[float] = []
    for predicted_index, gold_index in matched_pairs:
        predicted = predicted_words[predicted_index]
        reference = gold_words[gold_index]
        for key in ("start", "end"):
            left = float(predicted[key])
            right = float(reference[key])
            if not math.isfinite(left) or not math.isfinite(right):
                raise ValueError(f"non-finite {key} boundary")
            boundary_errors.append(abs(left - right))

    metadata = prediction.get("metadata", {})
    effective = int(metadata.get("mfa_effective_words") or 0)
    coverage = metadata.get("mfa_word_coverage")
    return {
        "schema_version": 1,
        "reference": {
            "name": str(gold.get("name") or "gold-reference"),
            "license": str(gold.get("license") or "unspecified"),
            "source_url": str(gold.get("source_url") or ""),
        },
        "reference_word_count": len(gold_tokens),
        "predicted_word_count": len(predicted_tokens),
        "word_errors": edit_distance,
        "wer": round(edit_distance / len(gold_tokens), 6),
        "timing_matched_words": len(matched_pairs),
        "boundary_sample_count": len(boundary_errors),
        "boundary_absolute_error_median_seconds": round(statistics.median(boundary_errors), 6) if boundary_errors else None,
        "boundary_absolute_error_p95_seconds": round(_percentile(boundary_errors, 0.95), 6) if boundary_errors else None,
        "mfa_effective_words": effective,
        "mfa_application_rate": round(effective / max(1, len(predicted_tokens)), 6),
        "mfa_reported_coverage": float(coverage) if coverage is not None else None,
    }


def _align_tokens(predicted: Sequence[str], gold: Sequence[str]) -> tuple[int, list[tuple[int, int]]]:
    rows = len(predicted) + 1
    columns = len(gold) + 1
    costs = [[0] * columns for _ in range(rows)]
    moves = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0] = i
        moves[i][0] = "delete"
    for j in range(1, columns):
        costs[0][j] = j
        moves[0][j] = "insert"
    for i in range(1, rows):
        for j in range(1, columns):
            if predicted[i - 1] == gold[j - 1]:
                costs[i][j] = costs[i - 1][j - 1]
                moves[i][j] = "match"
                continue
            candidates = (
                (costs[i - 1][j - 1] + 1, "substitute"),
                (costs[i - 1][j] + 1, "delete"),
                (costs[i][j - 1] + 1, "insert"),
            )
            costs[i][j], moves[i][j] = min(candidates, key=lambda item: item[0])

    matches: list[tuple[int, int]] = []
    i, j = len(predicted), len(gold)
    while i or j:
        move = moves[i][j]
        if move == "match":
            matches.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif move == "substitute":
            i -= 1
            j -= 1
        elif move == "delete":
            i -= 1
        elif move == "insert":
            j -= 1
        else:
            raise AssertionError("token alignment backtrace is incomplete")
    matches.reverse()
    return costs[-1][-1], matches


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight
