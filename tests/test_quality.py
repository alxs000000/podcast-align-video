import pytest

from podcast_align_video.quality import quality_report


def test_quality_report_calculates_wer_boundaries_and_mfa_rate() -> None:
    prediction = {
        "metadata": {"mfa_effective_words": 2, "mfa_word_coverage": 2 / 3},
        "words": [
            {"text": "One,", "clean": "One", "start": 0.1, "end": 0.4},
            {"text": "extra", "clean": "extra", "start": 0.4, "end": 0.6},
            {"text": "three.", "clean": "three", "start": 0.7, "end": 1.1},
        ],
    }
    gold = {
        "name": "fixture",
        "license": "CC0-1.0",
        "words": [
            {"text": "one", "start": 0.0, "end": 0.5},
            {"text": "two", "start": 0.5, "end": 0.7},
            {"text": "three", "start": 0.8, "end": 1.0},
        ],
    }
    report = quality_report(prediction, gold)
    assert report["word_errors"] == 1
    assert report["wer"] == pytest.approx(1 / 3, abs=1e-6)
    assert report["timing_matched_words"] == 2
    assert report["boundary_absolute_error_median_seconds"] == 0.1
    assert report["boundary_absolute_error_p95_seconds"] == 0.1
    assert report["mfa_application_rate"] == pytest.approx(2 / 3, abs=1e-6)
