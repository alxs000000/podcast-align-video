from podcast_align_video.render.hybrid import ass_time, build_ass
from podcast_align_video.speech_cut import kept_regions, silence_candidates, subtract_protected


def test_silence_threshold_includes_edges() -> None:
    assert silence_candidates([(6, 10), (12, 15)], 22, 5) == [(0.0, 6.0), (15.0, 22)]


def test_no_qualifying_silence_produces_no_candidates() -> None:
    assert silence_candidates([(0, 10)], 10, 5) == []


def test_alignment_protection_splits_candidate() -> None:
    cuts = subtract_protected([(0, 12)], [(5, 6)], 5)
    assert cuts == [(0, 5), (6, 12)]
    assert kept_regions(cuts, 15) == [(5, 6), (12, 15)]


def test_ass_segment_times_are_relative_and_bounded() -> None:
    alignment = {
        "words": [
            {"text": "Hello", "focus_start": 1800.2, "focus_end": 1800.6},
            {"text": "world", "focus_start": 1800.6, "focus_end": 1801.0},
        ],
        "sentences": [{"word_start": 0, "word_end": 2, "start": 1800.2, "end": 1801.0}],
    }
    rect = {
        "x": 100,
        "y": 400,
        "width": 200,
        "height": 80,
        "textX": 110,
        "textY": 410,
        "textWidth": 180,
        "textHeight": 60,
        "fontSize": 76,
        "text": "Hello",
    }
    layout = {"sentence_index": 0, "words": [rect, {**rect, "x": 300, "textX": 310, "text": "world"}]}
    ass = build_ass(alignment, [layout], 1800, 1802, 1920, 1080)
    assert "0:00:00.20" in ass
    assert "0:00:01.00" in ass
    assert "Geist" in ass
    assert ass_time(3661.239) == "1:01:01.23"
