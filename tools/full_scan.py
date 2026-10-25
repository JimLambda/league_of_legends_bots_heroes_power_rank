"""Full-folder scan report: every screenshot -> OCR -> parser -> leaderboard.

Usage:  python tools/full_scan.py
Prints per-file parse quality plus the aggregated bot-hero leaderboard.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from botrank.champions import ChampionDB
from botrank.ocr import RapidEngine
from botrank.pipeline import process_folder
from botrank.scoring import build_leaderboard
from botrank.storage import Store


def main() -> None:
    folder = ROOT / "league_of_legends_bot_matches_records"
    store = Store(Path(tempfile.mkdtemp()) / "scan.db")
    started = time.time()
    stats = process_folder(folder, RapidEngine(lang="ch"), store,
                           ChampionDB.load())
    print("scan: %.0fs | %s" % (time.time() - started, stats))

    records = store.all_records()
    bots = [r for r in records if r.get("is_bot")]
    print("records: %d | bots: %d | humans: %d | champions: %d"
          % (len(records), len(bots), len(records) - len(bots),
             len({r["champion_id"] for r in bots})))
    with_damage = sum(1 for r in bots if r["damage"])
    with_gold = sum(1 for r in bots if r["gold"])
    print("bot rows with damage: %d/%d | with gold: %d/%d"
          % (with_damage, len(bots), with_gold, len(bots)))

    board = build_leaderboard(records, min_games=5)
    print("\nTOP 10 (min 5 games):")
    for row in board[:10]:
        print("  #%-2d %-14s games=%-3d %d-%d wr=%3.0f%% kda=%.2f "
              "dmg=%8s gold=%7s score=%.1f"
              % (row["rank"], row["champion"], row["games"], row["wins"],
                 row["losses"], 100 * row["win_rate"], row["kda_ratio"],
                 format(int(row["avg_damage"] or 0), ","),
                 format(int(row["avg_gold"] or 0), ","),
                 row["power_score"]))


if __name__ == "__main__":
    main()
