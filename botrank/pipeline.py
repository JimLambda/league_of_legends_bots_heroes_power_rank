"""Batch processing: screenshot folder -> parsed records in the store."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

from .champions import ChampionDB
from .ocr import BaseEngine
from .storage import Store

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

# ScreenShot_2026-10-24_232407_587.png -> 2026-10-24 23:24:07
_FILENAME_TS_RE = re.compile(
    r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})[-_]?(\d{2})")

ProgressCallback = Callable[[int, int, str, dict], None]


def sha1_of(path: Union[str, Path]) -> str:
    digest = hashlib.sha1()
    with open(str(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_images(folder: Union[str, Path],
                   recursive: bool = False) -> List[Path]:
    """All screenshot files in ``folder``, chronologically sorted."""
    root = Path(folder)
    pattern = "**/*" if recursive else "*"
    images = [p for p in root.glob(pattern)
              if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images, key=lambda p: p.name.lower())


def match_time_from_name(path: Union[str, Path]) -> Optional[str]:
    """Recover the capture time from the screenshot filename, if present."""
    found = _FILENAME_TS_RE.search(Path(path).name)
    if not found:
        return None
    year, month, day, hour, minute, second = (int(g) for g in found.groups())
    try:
        return datetime(year, month, day, hour, minute, second).isoformat(
            timespec="seconds")
    except ValueError:
        return None


def records_from_match(match, file_path: str, match_time: Optional[str],
                       source: str = "ocr") -> List[dict]:
    """Flatten a :class:`~botrank.parser.ParsedMatch` into storage rows."""
    return [{
        "raw_name": player.raw_name or player.name,
        "champion_id": player.champion_id,
        "champion_name": player.champion_name,
        "is_bot": player.is_bot,
        "name_match": player.name_match,
        "team": player.team,
        "level": player.level,
        "kills": player.kills,
        "deaths": player.deaths,
        "assists": player.assists,
        "damage": player.damage,
        "gold": player.gold,
        "gold_per_min": player.gold_per_min,
        "outcome": player.outcome,
        "duration_sec": match.duration_sec,
        "match_time": match_time,
        "confidence": player.confidence,
        "source": source,
    } for player in match.players]


def process_folder(folder: Union[str, Path], engine: BaseEngine,
                   store: Store, champ_db: ChampionDB,
                   progress_cb: Optional[ProgressCallback] = None,
                   skip_ok: bool = True,
                   max_files: Optional[int] = None) -> dict:
    """OCR every new screenshot in ``folder`` and store the parsed rows.

    ``progress_cb(done, total, current_name, stats)`` is invoked after each
    file so the UI can render a live leaderboard while scanning.
    """
    images = collect_images(folder)
    stats = {"found": len(images), "processed": 0, "skipped": 0,
             "failed": 0, "new_records": 0, "bot_records": 0}

    work: List[Path] = []
    for path in images:
        try:
            digest = sha1_of(path)
        except OSError:
            stats["failed"] += 1
            continue
        if skip_ok and store.file_status(digest) == "ok":
            stats["skipped"] += 1
            continue
        work.append(path)
        if max_files is not None and len(work) >= max_files:
            break

    total = len(work)
    for index, path in enumerate(work, 1):
        try:
            digest = sha1_of(path)
            match = engine.extract(path, champ_db)
            if not match.players:
                raise ValueError("no scoreboard rows parsed")
            match_time = match_time_from_name(path)
            store.replace_records(digest, path,
                                  records_from_match(match, str(path),
                                                     match_time))
            store.mark_file(digest, path, "ok")
            stats["processed"] += 1
            stats["new_records"] += len(match.players)
            stats["bot_records"] += len(match.bots)
        except Exception as exc:                       # keep scanning
            try:
                store.mark_file(sha1_of(path), path, "error", str(exc))
            except Exception:
                pass
            stats["failed"] += 1
        if progress_cb is not None:
            progress_cb(index, total, path.name, dict(stats))
    return stats
