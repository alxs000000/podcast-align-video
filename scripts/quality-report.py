#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from podcast_align_video.quality import quality_report
from podcast_align_video.util import atomic_write_json
from podcast_align_video.validate import validate_word_timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare word-timings.json with a licensed gold reference")
    parser.add_argument("prediction", type=Path)
    parser.add_argument("gold", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    prediction = validate_word_timings(args.prediction)
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    report = quality_report(prediction, gold)
    if args.output is not None:
        atomic_write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
