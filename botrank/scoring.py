"""Power Score calculation and leaderboard aggregation.

The composite **Power Score** (0–100) blends four per-champion metrics,
each shrunk towards the global mean (Bayesian prior of 3 games) and then
min-max normalised across the field:

==================  ======  ==========================================
component           weight  meaning
==================  ======  ==========================================
kill participation  0.35    average (kills + assists) per game
damage              0.30    average total damage to champions
gold                0.20    average gold earned
win rate            0.15    shrunk victory rate of the bot's team
==================  ======  ==========================================
"""
from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Optional, Sequence

DEFAULT_WEIGHTS = {"ka": 0.35, "damage": 0.30, "gold": 0.20, "winrate": 0.15}
PRIOR_GAMES = 3.0          # shrinkage strength towards the global mean
MAX_KDA_CAP = 15.0         # cap the displayed KDA ratio


def _avg(values: Sequence[float]) -> Optional[float]:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def build_leaderboard(records: Iterable[dict], min_games: int = 1,
                      weights: Optional[Dict[str, float]] = None
                      ) -> List[dict]:
    """Aggregate player records into a ranked bot-hero leaderboard.

    ``records`` are dicts as produced by :meth:`botrank.storage.Store.all_records`
    (keys: ``champion_id``, ``champion_name``, ``is_bot``, ``kills``,
    ``deaths``, ``assists``, ``damage``, ``gold``, ``outcome`` …).
    """
    weights = dict(weights or DEFAULT_WEIGHTS)
    bots = [r for r in records if r.get("is_bot")]
    if not bots:
        return []

    groups: Dict[str, dict] = {}
    for record in bots:
        key = record.get("champion_id") or "raw:%s" % record.get("raw_name", "?")
        group = groups.setdefault(key, {
            "champion_id": record.get("champion_id") or "",
            "champion_name": record.get("champion_name") or key,
            "games": 0, "wins": 0,
            "kills": [], "deaths": [], "assists": [],
            "damage": [], "gold": [], "gold_per_min": [],
            "last_match": "",
        })
        group["games"] += 1
        if record.get("outcome") == "VICTORY":
            group["wins"] += 1
        group["kills"].append(record.get("kills") or 0)
        group["deaths"].append(record.get("deaths") or 0)
        group["assists"].append(record.get("assists") or 0)
        if record.get("damage") is not None:
            group["damage"].append(record["damage"])
        if record.get("gold") is not None:
            group["gold"].append(record["gold"])
        if record.get("gold_per_min") is not None:
            group["gold_per_min"].append(record["gold_per_min"])
        stamp = record.get("match_time") or ""
        if stamp > group["last_match"]:
            group["last_match"] = stamp

    rows: List[dict] = []
    for group in groups.values():
        games = group["games"]
        avg_k = _avg(group["kills"]) or 0.0
        avg_d = _avg(group["deaths"]) or 0.0
        avg_a = _avg(group["assists"]) or 0.0
        avg_damage = _avg(group["damage"])
        avg_gold = _avg(group["gold"])
        avg_gpm = _avg(group["gold_per_min"])
        rows.append({
            "champion_id": group["champion_id"],
            "champion": group["champion_name"],
            "games": games,
            "wins": group["wins"],
            "losses": games - group["wins"],
            "win_rate": group["wins"] / games if games else 0.0,
            "avg_kills": avg_k,
            "avg_deaths": avg_d,
            "avg_assists": avg_a,
            "avg_ka": avg_k + avg_a,
            "kda_ratio": min((avg_k + avg_a) / max(avg_d, 1.0), MAX_KDA_CAP),
            "avg_damage": avg_damage,
            "avg_gold": avg_gold,
            "avg_gpm": avg_gpm,
            "last_match": group["last_match"],
        })

    rows = [r for r in rows if r["games"] >= min_games]
    if not rows:
        return []

    # --- shrinkage towards the global mean, then min-max normalisation ----
    def shrink(value: Optional[float], games: int, mean: float) -> float:
        if value is None:
            return mean
        return (value * games + mean * PRIOR_GAMES) / (games + PRIOR_GAMES)

    def normalise(key: str, getter) -> None:
        values = [getter(r) for r in rows]
        low, high = min(values), max(values)
        span = high - low
        for row, value in zip(rows, values):
            row[key] = 0.5 if span <= 0 else (value - low) / span

    mean_ka = statistics.fmean([r["avg_ka"] for r in rows])
    mean_damage = statistics.fmean([r["avg_damage"] for r in rows
                                    if r["avg_damage"] is not None])
    mean_gold = statistics.fmean([r["avg_gold"] for r in rows
                                  if r["avg_gold"] is not None])
    mean_wr = statistics.fmean([r["win_rate"] for r in rows])

    normalise("n_ka", lambda r: shrink(r["avg_ka"], r["games"], mean_ka))
    normalise("n_damage", lambda r: shrink(r["avg_damage"], r["games"], mean_damage))
    normalise("n_gold", lambda r: shrink(r["avg_gold"], r["games"], mean_gold))
    normalise("n_winrate", lambda r: shrink(r["win_rate"], r["games"], mean_wr))

    for row in rows:
        row["power_score"] = round(100.0 * (
            weights["ka"] * row["n_ka"]
            + weights["damage"] * row["n_damage"]
            + weights["gold"] * row["n_gold"]
            + weights["winrate"] * row["n_winrate"]
        ), 1)

    rows.sort(key=lambda r: (-r["power_score"], -r["games"]))
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def human_vs_bot_summary(records: Iterable[dict]) -> dict:
    """Side-by-side bot vs human performance (for the Insights tab)."""
    summary = {"bot": _side_summary([]), "human": _side_summary([])}
    for record in records:
        side = "bot" if record.get("is_bot") else "human"
        summary[side] = _accumulate(summary[side], record)
    for side in summary:
        _finalise(summary[side])
    return summary


def _side_summary(base: Optional[dict] = None) -> dict:
    return {"games": 0, "wins": 0, "kills": [], "deaths": [], "assists": [],
            "damage": [], "gold": []}


def _accumulate(side: dict, record: dict) -> dict:
    side["games"] += 1
    if record.get("outcome") == "VICTORY":
        side["wins"] += 1
    side["kills"].append(record.get("kills") or 0)
    side["deaths"].append(record.get("deaths") or 0)
    side["assists"].append(record.get("assists") or 0)
    if record.get("damage") is not None:
        side["damage"].append(record["damage"])
    if record.get("gold") is not None:
        side["gold"].append(record["gold"])
    return side


def _finalise(side: dict) -> None:
    games = max(side["games"], 1)
    avg_k = sum(side["kills"]) / games
    avg_d = sum(side["deaths"]) / games
    avg_a = sum(side["assists"]) / games
    side["win_rate"] = side["wins"] / games
    side["kda_ratio"] = (avg_k + avg_a) / max(avg_d, 1.0)
    side["avg_damage"] = (sum(side["damage"]) / len(side["damage"])
                          if side["damage"] else None)
    side["avg_gold"] = (sum(side["gold"]) / len(side["gold"])
                        if side["gold"] else None)


def side_split(records: Iterable[dict]) -> Dict[str, dict]:
    """Bot win rate when playing on team 1 (blue) vs team 2 (red)."""
    split = {"team1": {"games": 0, "wins": 0}, "team2": {"games": 0, "wins": 0}}
    for record in records:
        if not record.get("is_bot"):
            continue
        key = "team%d" % (record.get("team") or 1)
        split[key]["games"] += 1
        if record.get("outcome") == "VICTORY":
            split[key]["wins"] += 1
    for entry in split.values():
        entry["win_rate"] = (entry["wins"] / entry["games"]
                             if entry["games"] else 0.0)
    return split
