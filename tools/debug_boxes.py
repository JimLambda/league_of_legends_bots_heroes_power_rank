"""Debug helper: dump raw OCR boxes for a screenshot.

Usage:  python tools/debug_boxes.py 2026-10-24
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from botrank.ocr import RapidEngine

FOLDER = ROOT / "league_of_legends_bot_matches_records"


def dump(name_fragment: str) -> None:
    path = next(p for p in FOLDER.glob("*.png") if name_fragment in p.name)
    engine = RapidEngine(lang="ch")
    boxes = engine.ocr_boxes(path)
    print("=== %s : %d boxes ===" % (path.name, len(boxes)))
    for box in sorted(boxes, key=lambda b: (round(b.cy / 25), b.cx)):
        print("  y=%4d x=%4d  %r  (%.2f)" % (box.cy, box.cx, box.text, box.score))


if __name__ == "__main__":
    dump(sys.argv[1] if len(sys.argv) > 1 else "2026-10-24")
