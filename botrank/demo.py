"""Seed plausible demo records so the dashboard can be explored without OCR.

The data is clearly marked ``source='demo'`` and can be wiped at any time
from the app's *Data* tab.
"""
from __future__ import annotations

import random
from typing import List

from .pipeline import records_from_match
from .parser import ParsedMatch, PlayerRow
from .storage import Store

_DEMY_BOTS = [
    # (champion_id, champion_name, games, skill)  skill: 0 (weak) .. 1 (strong)
    ("Yi", "Master Yi", 22, 0.95),
    ("Viego", "Viego", 18, 0.90),
    ("Tryndamere", "Tryndamere", 16, 0.82),
    ("Nasus", "Nasus", 15, 0.75),
    ("Amumu", "Amumu", 14, 0.70),
    ("Veigar", "Veigar", 14, 0.68),
    ("Sejuani", "Sejuani", 12, 0.55),
    ("Katarina", "Katarina", 11, 0.60),
    ("Caitlyn", "Caitlyn", 10, 0.52),
    ("Lux", "Lux", 9, 0.48),
    ("Teemo", "Teemo", 8, 0.35),
    ("Yuumi", "Yuumi", 7, 0.30),
]

_HUMAN_NAMES = ["SuperLambda", "Tomwwy", "正义敖辣汤姆", "自己嘛潘的影子",
                "RBB0911", "Moonlight", "ZZZ_终结者"]


def seed_demo_data(store: Store, seed: int = 20261024) -> int:
    """Insert synthetic bot-game records; returns the number inserted."""
    rng = random.Random(seed)
    inserted = 0
    for champion_id, champion_name, games, skill in _DEMY_BOTS:
        for game in range(games):
            strength = max(0.0, min(1.0, skill + rng.uniform(-0.15, 0.15)))
            kills = max(0, int(rng.gauss(3 + 12 * strength, 2.5)))
            deaths = max(0, int(rng.gauss(9 - 5 * strength, 2.0)))
            assists = max(0, int(rng.gauss(3 + 7 * strength, 2.5)))
            damage = max(2000, int(rng.gauss(9000 + 28000 * strength, 4500)))
            gold = max(2500, int(rng.gauss(5500 + 9000 * strength, 1800)))
            won = rng.random() < 0.25 + 0.65 * strength
            team = 1 if rng.random() < 0.5 else 2
            match = ParsedMatch(
                team1_outcome="VICTORY" if (team == 1) == won else "DEFEAT",
                duration_sec=rng.randint(1200, 2400))
            bot = PlayerRow(
                name=champion_name, raw_name=champion_name, level=18,
                kills=kills, deaths=deaths, assists=assists,
                damage=damage, gold=gold,
                gold_per_min=gold // max(match.duration_sec // 60, 1),
                team=team, is_bot=True, champion_id=champion_id,
                champion_name=champion_name, name_match="exact",
                confidence=0.99, outcome="VICTORY" if won else "DEFEAT")
            match.players.append(bot)
            # one friendly human per demo game for the bot-vs-human compare
            human = PlayerRow(
                name=rng.choice(_HUMAN_NAMES),
                raw_name=rng.choice(_HUMAN_NAMES), level=18,
                kills=max(1, int(rng.gauss(9, 3))),
                deaths=max(0, int(rng.gauss(6, 2.5))),
                assists=max(0, int(rng.gauss(8, 3))),
                damage=max(5000, int(rng.gauss(26000, 6000))),
                gold=max(6000, int(rng.gauss(14500, 2500))),
                gold_per_min=gold // max(match.duration_sec // 60, 1),
                team=team, is_bot=False, confidence=0.99,
                outcome=bot.outcome)
            match.players.append(human)
            store.replace_records("demo-%s-%d" % (champion_id, game),
                                  "<demo>", records_from_match(
                                      match, "<demo>", None, source="demo"))
            inserted += 1
    return inserted
