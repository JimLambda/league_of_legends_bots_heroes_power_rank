"""Unit tests for the stdlib-only core (champions, parser, scoring, storage).

Run with:  python -m pytest tests/ -q      (or: python tests/test_core.py)
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from botrank.champions import ChampionDB, levenshtein, normalize_name  # noqa: E402
from botrank.demo import seed_demo_data  # noqa: E402
from botrank.parser import Box, parse_scoreboard  # noqa: E402
from botrank.pipeline import (collect_images, match_time_from_name,  # noqa: E402
                              records_from_match)
from botrank.scoring import build_leaderboard, human_vs_bot_summary  # noqa: E402
from botrank.storage import Store  # noqa: E402

IMG_W, IMG_H = 1920, 1080


class TestPaddleOutputFlattening(unittest.TestCase):
    """PaddleOCR 2.8 returns ``[[line, …]]`` — the old flattener dropped all
    but the first line, producing 0 scoreboard rows. Regression coverage."""

    def _line(self, label, x=100, y=100):
        poly = [[x, y], [x + 50, y], [x + 50, y + 20], [x, y + 20]]
        return [poly, (label, 0.95)]

    def test_flatten_double_wrapped_2_8(self):
        from botrank.ocr import _flatten_paddle
        # PaddleOCR 2.8.1: one outer list (per image) of lines
        result = [[self._line("A"), self._line("B"), self._line("C")]]
        lines = _flatten_paddle(result)
        self.assertEqual([l[1][0] for l in lines], ["A", "B", "C"])

    def test_flatten_flat_list(self):
        from botrank.ocr import _flatten_paddle
        # some builds return the line list directly
        result = [self._line("A"), self._line("B")]
        lines = _flatten_paddle(result)
        self.assertEqual([l[1][0] for l in lines], ["A", "B"])

    def test_flatten_is_line_detects_line_not_list(self):
        from botrank.ocr import _is_line
        self.assertTrue(_is_line(self._line("A")))
        # a list of lines must NOT be mistaken for a line
        self.assertFalse(_is_line([self._line("A"), self._line("B")]))


class TestFlatBoxGeometry(unittest.TestCase):
    """PaddleOCR can hand back a flat 8-element box polygon instead of 4 points.

    Regression for: "'float' object is not subscriptable".
    """

    def test_box_from_flat_polygon(self):
        from botrank.ocr import _box_from_points
        # flat [x1,y1,x2,y2,x3,y3,x4,y4]
        box = _box_from_points([100, 200, 300, 200, 300, 230, 100, 230], "X", 0.9)
        self.assertEqual((box.x1, box.y1, box.x2, box.y2), (100, 200, 300, 230))
        self.assertEqual(box.text, "X")
        self.assertEqual(box.score, 0.9)

    def test_boxes_from_paddle_flat(self):
        from botrank.ocr import boxes_from_paddle
        # PaddleOCR 3.x predict() returns a list of dicts; polygons can be
        # flat 8-element lists [x1,y1,…,x4,y4] instead of 4 [x,y] points.
        result = [{"dt_polys": [[100, 200, 300, 200, 300, 230, 100, 230]],
                   "rec_texts": ["Veigar"], "rec_scores": [0.95]},
                  {"dt_polys": [[400, 200, 500, 200, 500, 230, 400, 230]],
                   "rec_texts": ["Sejuani"], "rec_scores": [0.91]}]
        boxes = boxes_from_paddle(result)
        self.assertEqual(len(boxes), 2)
        self.assertEqual(boxes[0].text, "Veigar")
        self.assertEqual((boxes[0].x1, boxes[0].y2), (100, 230))

    def test_parse_tolerates_flat_boxes(self):
        from botrank.ocr import boxes_from_paddle
        result = [
            {"dt_polys": [[80, 30, 200, 30, 200, 80, 80, 80]],
             "rec_texts": ["胜利"], "rec_scores": [0.99]},
            {"dt_polys": [[700, 160, 800, 160, 800, 190, 700, 190]],
             "rec_texts": ["5/12/16"], "rec_scores": [0.98]},
            {"dt_polys": [[150, 150, 300, 150, 300, 200, 150, 200]],
             "rec_texts": ["Veigar"], "rec_scores": [0.97]},
            {"dt_polys": [[830, 160, 930, 160, 930, 190, 830, 190]],
             "rec_texts": ["32,637"], "rec_scores": [0.96]},
            {"dt_polys": [[920, 160, 1020, 160, 1020, 190, 920, 190]],
             "rec_texts": ["13,565"], "rec_scores": [0.96]},
        ]
        match = parse_scoreboard(
            boxes_from_paddle(result), IMG_W, IMG_H, ChampionDB.load())
        self.assertEqual(len(match.players), 1)
        self.assertEqual(match.players[0].name, "Veigar")
        self.assertEqual(match.players[0].kills, 5)
        self.assertEqual(match.players[0].damage, 32637)
        self.assertEqual(match.players[0].gold, 13565)
        self.assertEqual(match.team1_outcome, "VICTORY")




# --------------------------------------------------------------------------- #
# synthetic scoreboard builder (mirrors the real post-game layout)
# --------------------------------------------------------------------------- #

def _row(cy, level, name, title, kda, damage, gold, gpm, x_name=100):
    level_box = Box(55, cy - 14, 85, cy + 14, level)
    name_box = Box(x_name, cy - 18, x_name + 120, cy + 2, name)
    title_box = Box(x_name, cy + 4, x_name + 120, cy + 18, title)
    kda_box = Box(700, cy - 16, 850, cy + 16, kda)
    dmg_box = Box(900, cy - 14, 990, cy + 14, damage)
    gold_box = Box(1020, cy - 14, 1110, cy + 14, gold)
    gpm_box = Box(1140, cy - 12, 1250, cy + 12, gpm)
    ratio_box = Box(700, cy + 18, 850, cy + 32, "1.8战损比")
    return [level_box, name_box, title_box, kda_box, dmg_box, gold_box,
            gpm_box, ratio_box]


def _victory_boxes():
    boxes = [
        Box(60, 25, 220, 75, "胜利"),                       # banner
        Box(700, 35, 790, 65, "45:14"),                    # match duration
        Box(60, 110, 160, 145, "第1队"),                   # team anchors
        Box(200, 115, 350, 140, "81 / 51 / 83"),
        Box(400, 118, 490, 138, "92,128"),
        Box(60, 455, 160, 490, "第2队"),
        Box(200, 460, 350, 485, "50 / 84 / 57"),
        Box(400, 462, 490, 482, "81,660"),
    ]
    team1 = [
        (190, "20", "Kennen", "狂暴之心", "5 / 12 / 16", "32,637", "13,565", "300/分钟"),
        (245, "18", "自己嘛潘的影子", "虚空女皇", "10 / 3 / 18", "29,039", "19,244", "425/分钟"),
        (300, "18", "Lux", "光辉女郎", "9 / 22 / 11", "31,793", "15,159", "335/分钟"),
        (355, "18", "正义敖辣汤姆", "花血武娘", "50 / 8 / 12", "135,034", "30,626", "677/分钟"),
        (410, "18", "Taric", "瓦洛兰之盾", "7 / 6 / 26", "23,302", "13,534", "299/分钟"),
    ]
    team2 = [
        (550, "20", "Urgot", "无畏战车", "13 / 13 / 5", "49,016", "16,871", "373/分钟"),
        (605, "18", "Sejuani", "北地之怒", "2 / 16 / 14", "32,942", "13,356", "295/分钟"),
        (660, "17", "Veigar", "邪恶小法师", "15 / 15 / 11", "59,570", "16,279", "360/分钟"),
        (715, "19", "Veigar", "邪恶小法师", "19 / 19 / 9", "92,316", "23,026", "509/分钟"),
        (770, "16", "Amumu", "殇之木乃伊", "1 / 21 / 18", "24,876", "12,128", "268/分钟"),
    ]
    for row in team1 + team2:
        boxes.extend(_row(*row))
    return boxes


def _defeat_boxes():
    boxes = [Box(60, 25, 220, 75, "敗北"), Box(700, 35, 790, 65, "13:39:11")]
    team1 = [
        (190, "18", "SuperLambda", "チョナス", "40 / 9 / 5", "137,675", "34,436", "658/分"),
        (245, "19", "Darius", "ダリウス", "3 / 24 / 8", "13,290", "12,989", "248/分"),
    ]
    team2 = [
        (550, "18", "Urgot", "アーゴット", "24 / 1 / 2", "54,435", "19,659", "376/分"),
        (605, "17", "Morganna", "モルガナ", "7 / 21 / 16", "32,427", "16,195", "309/分"),
    ]
    boxes.extend([Box(60, 110, 160, 145, "チーム 1"),
                  Box(60, 455, 160, 490, "チーム 2")])
    for row in team1 + team2:
        boxes.extend(_row(*row, x_name=100))
    return boxes


# --------------------------------------------------------------------------- #


class TestChampions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = ChampionDB.load()

    def test_database_loaded(self):
        self.assertGreater(len(self.db), 150)

    def test_normalize(self):
        self.assertEqual(normalize_name("Bel'Veth"), "belveth")
        self.assertEqual(normalize_name("ヌヌ＆ウィルンプ"), "ヌヌウィルンプ")
        self.assertEqual(normalize_name("  Veigar Bot "), "veigarbot")

    def test_exact_match_is_bot(self):
        for name in ("Veigar", "Sejuani", "Kennen", "Master Yi",
                     "Bel'Veth", "Nunu & Willump", "Kai'Sa"):
            result = self.db.match(name)
            self.assertIsNotNone(result, name)
            self.assertEqual(result.method, "exact", name)

    def test_human_names_are_not_bots(self):
        for name in ("SuperLambda", "Tomwwy", "正义敖辣汤姆", "RBB0911",
                     "Kenneth", "Moonlight"):
            self.assertIsNone(self.db.match(name), name)

    def test_fuzzy_absorbs_ocr_noise(self):
        result = self.db.match("Morganna")      # OCR of "Morgana"
        self.assertIsNotNone(result)
        self.assertEqual(result.method, "fuzzy")
        self.assertEqual(result.champion_id, "Morgana")

    def test_bot_affix_stripped(self):
        self.assertIsNotNone(self.db.match("Veigar Bot"))
        self.assertIsNotNone(self.db.match("bot Veigar"))

    def test_levenshtein(self):
        self.assertEqual(levenshtein("morganna", "morgana"), 1)
        self.assertEqual(levenshtein("abc", "abc"), 0)


class TestParser(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.db = ChampionDB.load()

    def test_victory_screenshot(self):
        match = parse_scoreboard(_victory_boxes(), IMG_W, IMG_H, self.db)
        self.assertEqual(len(match.players), 10, match.warnings)
        self.assertEqual(match.team1_outcome, "VICTORY")
        self.assertEqual(match.duration_sec, 45 * 60 + 14)

        by_name = {p.name: p for p in match.players}
        kennen = by_name["Kennen"]
        self.assertTrue(kennen.is_bot)
        self.assertEqual(kennen.champion_id, "Kennen")
        self.assertEqual((kennen.kills, kennen.deaths, kennen.assists),
                         (5, 12, 16))
        self.assertEqual(kennen.damage, 32637)
        self.assertEqual(kennen.gold, 13565)
        self.assertEqual(kennen.gold_per_min, 300)
        self.assertEqual(kennen.level, 20)
        self.assertEqual(kennen.team, 1)
        self.assertEqual(kennen.outcome, "VICTORY")

        human = by_name["自己嘛潘的影子"]
        self.assertFalse(human.is_bot)
        self.assertEqual((human.kills, human.deaths, human.assists),
                         (10, 3, 18))

        urgot = by_name["Urgot"]
        self.assertTrue(urgot.is_bot)
        self.assertEqual(urgot.team, 2)
        self.assertEqual(urgot.outcome, "DEFEAT")

        bots = {p.champion_id for p in match.bots}
        self.assertEqual(bots, {"Kennen", "Lux", "Taric", "Urgot", "Sejuani",
                                "Veigar", "Amumu"})
        self.assertEqual(len(match.bots), 8)
        self.assertEqual(len(match.humans), 2)

    def test_defeat_screenshot(self):
        match = parse_scoreboard(_defeat_boxes(), IMG_W, IMG_H, self.db)
        self.assertEqual(len(match.players), 4)
        self.assertEqual(match.team1_outcome, "DEFEAT")
        by_name = {p.name: p for p in match.players}
        self.assertTrue(by_name["SuperLambda"].is_bot is False)
        self.assertTrue(by_name["Urgot"].is_bot)
        self.assertEqual(by_name["Urgot"].outcome, "VICTORY")
        self.assertEqual(by_name["Morganna"].champion_id, "Morgana")
        self.assertEqual(by_name["Darius"].outcome, "DEFEAT")

    def test_split_kda_boxes_are_recovered(self):
        boxes = _victory_boxes()
        # replace Kennen's single KDA box with three separate digit boxes
        boxes = [b for b in boxes if not (690 < b.cx < 860
                                          and abs(b.cy - 190) < 20)]
        boxes.extend([Box(710, 184, 730, 200, "5"),
                      Box(760, 184, 790, 200, "12"),
                      Box(810, 184, 830, 200, "16")])
        match = parse_scoreboard(boxes, IMG_W, IMG_H, self.db)
        kennen = next(p for p in match.players if p.name == "Kennen")
        self.assertEqual((kennen.kills, kennen.deaths, kennen.assists),
                         (5, 12, 16))
        self.assertEqual(kennen.damage, 32637)

    def test_no_scoreboard(self):
        match = parse_scoreboard([Box(10, 10, 100, 40, "hello world")],
                                 IMG_W, IMG_H, self.db)
        self.assertEqual(match.players, [])
        self.assertTrue(match.warnings)


class TestScoring(unittest.TestCase):
    def _records(self):
        records = []
        for champion, skill, games in (("Master Yi", 0.9, 20),
                                       ("Veigar", 0.5, 6),
                                       ("Teemo", 0.2, 2)):
            for i in range(games):
                records.append({
                    "champion_id": champion, "champion_name": champion,
                    "is_bot": True, "name_match": "exact",
                    "kills": int(3 + 10 * skill), "deaths": int(9 - 5 * skill),
                    "assists": int(3 + 7 * skill),
                    "damage": int(9000 + 28000 * skill),
                    "gold": int(5500 + 9000 * skill),
                    "outcome": "VICTORY" if (i % 2 == 0) == (skill > 0.4)
                    else "DEFEAT",
                    "team": 1, "match_time": "2026-10-24T20:00:00",
                    "raw_name": champion,
                })
        return records

    def test_leaderboard_order_and_bounds(self):
        board = build_leaderboard(self._records())
        self.assertEqual([r["champion"] for r in board],
                         ["Master Yi", "Veigar", "Teemo"])
        for row in board:
            self.assertGreaterEqual(row["power_score"], 0.0)
            self.assertLessEqual(row["power_score"], 100.0)
            self.assertEqual(row["rank"], board.index(row) + 1)
        self.assertEqual(board[0]["games"], 20)
        self.assertAlmostEqual(board[0]["win_rate"], 0.5, places=6)

    def test_min_games_filter(self):
        filtered = build_leaderboard(self._records(), min_games=5)
        self.assertEqual([r["champion"] for r in filtered],
                         ["Master Yi", "Veigar"])
        self.assertEqual(len(filtered), 2)

    def test_empty(self):
        self.assertEqual(build_leaderboard([]), [])
        self.assertEqual(build_leaderboard([{"is_bot": False}]), [])

    def test_human_vs_bot_summary(self):
        records = self._records()
        records.append({"champion_id": None, "champion_name": None,
                        "is_bot": False, "name_match": "none", "kills": 9,
                        "deaths": 6, "assists": 8, "damage": 26000,
                        "gold": 14500, "outcome": "VICTORY", "team": 1,
                        "match_time": "", "raw_name": "Tomwwy"})
        summary = human_vs_bot_summary(records)
        self.assertEqual(summary["bot"]["games"], 28)
        self.assertEqual(summary["human"]["games"], 1)
        self.assertEqual(summary["human"]["win_rate"], 1.0)
        self.assertIsNotNone(summary["bot"]["avg_damage"])
        self.assertIsNotNone(summary["bot"]["avg_gold"])


class TestStorageAndPipeline(unittest.TestCase):
    def test_store_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "test.db")
            records = [{"raw_name": "Veigar", "champion_id": "Veigar",
                        "champion_name": "Veigar", "is_bot": True,
                        "name_match": "exact", "team": 1, "level": 18,
                        "kills": 5, "deaths": 12, "assists": 16,
                        "damage": 32637, "gold": 13565, "gold_per_min": 300,
                        "outcome": "VICTORY", "duration_sec": 2714,
                        "match_time": "2026-10-24T23:24:07",
                        "confidence": 0.95, "source": "ocr"}]
            store.replace_records("hash1", "a.png", records)
            store.mark_file("hash1", "a.png", "ok")
            self.assertEqual(store.file_status("hash1"), "ok")
            self.assertIsNone(store.file_status("nope"))
            loaded = store.all_records()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["champion_id"], "Veigar")
            self.assertEqual(store.stats()["records"], 1)
            self.assertEqual(store.stats()["bot_records"], 1)
            store.update_record(loaded[0]["id"], {"kills": 7})
            self.assertEqual(store.all_records()[0]["kills"], 7)
            store.mark_file("hash2", "b.png", "error", "boom")
            self.assertEqual(store.stats()["files_error"], 1)
            self.assertEqual(store.failed_files()[0]["error"], "boom")
            store.clear()
            self.assertEqual(store.stats()["records"], 0)
            store.close()

    def test_records_from_match(self):
        from botrank.parser import ParsedMatch, PlayerRow
        match = ParsedMatch(team1_outcome="VICTORY", duration_sec=2714)
        match.players.append(PlayerRow(name="Veigar", raw_name="Veigar",
                                       kills=5, deaths=12, assists=16,
                                       is_bot=True, champion_id="Veigar",
                                       champion_name="Veigar", team=1,
                                       outcome="VICTORY"))
        rows = records_from_match(match, "x.png", "2026-10-24T23:24:07")
        self.assertEqual(rows[0]["kills"], 5)
        self.assertEqual(rows[0]["duration_sec"], 2714)
        self.assertEqual(rows[0]["match_time"], "2026-10-24T23:24:07")

    def test_match_time_from_name(self):
        self.assertEqual(
            match_time_from_name("ScreenShot_2026-10-24_232407_587.png"),
            "2026-10-24T23:24:07")
        self.assertIsNone(match_time_from_name("random.png"))

    def test_demo_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(Path(tmp) / "demo.db")
            inserted = seed_demo_data(store)
            self.assertGreater(inserted, 100)
            board = build_leaderboard(store.all_records())
            self.assertGreater(len(board), 8)
            store.close()


class TestRealWorldOcrQuirks(unittest.TestCase):
    """Regression tests for quirks found on the user's real screenshots."""

    @classmethod
    def setUpClass(cls):
        cls.db = ChampionDB.load()

    def _boxes(self):
        """Layout of ScreenShot_2026-10-24 (zh client, real OCR quirks)."""
        boxes = [
            Box(80, 20, 200, 70, "胜利"),                       # banner
            Box(90, 45, 320, 75, "自定义·"),                    # merged top bar
            Box(230, 45, 480, 75, "胜：0-负：0·45:14·游戏ID"),   # duration inside
            Box(60, 125, 130, 150, "第1队"),                    # team anchors
            Box(140, 125, 220, 148, "81/51/83"),
            Box(250, 125, 340, 145, "92,128"),
            Box(60, 445, 130, 470, "第2队"),
            Box(140, 445, 220, 468, "50/84/57"),
            Box(250, 445, 340, 465, "81,660"),
        ]
        rows = [
            # (cy, level, name, title, kda, damage, gold, gpm, x_name)
            (166, "20", "Kennen", "狂暴之心", "5/12/16", "32,637", "13,565",
             "300/分钟", 194),
            (207, "18", "自己脚滑的影子", "虚空女皇", "10/3/18", "29,039",
             "19,244", "425/分钟", 196),
            (248, "18", "Lux", "光辉女郎", "9/22/11", "31,793", "15,159",
             "335/分钟", 195),
            (289, "18", "正义放肆汤姆", "龙血武姬", "50/8/12", "★135,034",
             "30,626", "677/分钟", 194),
            (331, "18", "Tarie", "瓦洛兰之盾", "7/6/26", "23,302", "13,534",
             "299/分钟", 194),
            (480, "20", "Urgot", "无畏战车", "13/13/5", "49,016", "16,871",
             "373/分钟", 194),
            (521, "18", "Sejuani", "北地之怒", "2/16/14", "32,942", "13,356",
             "295/分钟", 194),
            (562, "17", "Veigar", "邪恶小法师", "15/15/11", "59,570",
             "16,279", "360/分钟", 194),
            (603, "19", "Veigar", "邪恶小法师", "19/19/9", "92,316",
             "23,026", "509/分钟", 194),
            (644, "16", "Amumu", "殇之木乃伊", "1/21/18", "24,876",
             "12,128", "268/分钟", 194),
        ]
        for cy, level, name, title, kda, dmg, gold, gpm, x in rows:
            boxes.extend([
                Box(59, cy - 11, 87, cy + 10, level),
                # name and title start at (almost) the same x — name on top
                Box(x, cy - 9, x + 52, cy + 8, name),
                Box(x - 1, cy + 6, x + 58, cy + 22, title),
                Box(700, cy - 9, 800, cy + 9, kda),
                Box(830, cy - 9, 930, cy + 9, dmg),
                Box(920, cy - 9, 1020, cy + 9, gold),
                Box(920, cy + 12, 1030, cy + 26, gpm),
            ])
        return boxes

    def test_full_parse(self):
        match = parse_scoreboard(self._boxes(), IMG_W, IMG_H, self.db)
        self.assertEqual(len(match.players), 10, match.warnings)
        self.assertEqual(match.team1_outcome, "VICTORY")
        self.assertEqual(match.duration_sec, 45 * 60 + 14)
        by_name = {p.name: p for p in match.players}
        # the name line (topmost box) wins over the champion title
        self.assertIn("Kennen", by_name)
        self.assertIn("Lux", by_name)
        self.assertIn("Veigar", by_name)
        self.assertNotIn("狂暴之心", by_name)
        self.assertNotIn("光辉女郎", by_name)
        # MVP star glued to the damage value
        self.assertEqual(by_name["正义放肆汤姆"].damage, 135034)
        self.assertEqual(by_name["正义放肆汤姆"].gold, 30626)
        # fuzzy OCR variant still classified as a bot
        self.assertTrue(by_name["Tarie"].is_bot)
        self.assertEqual(by_name["Tarie"].champion_id, "Taric")
        # humans stay human
        self.assertFalse(by_name["自己脚滑的影子"].is_bot)
        self.assertFalse(by_name["正义放肆汤姆"].is_bot)
        # team split
        teams = {p.name: p.team for p in match.players}
        self.assertEqual(teams["Kennen"], 1)
        self.assertEqual(teams["Amumu"], 2)
        self.assertEqual(teams["Urgot"], 2)

    def test_misread_team_anchor_japanese(self):
        """チーム 1 misread as 于一么1 — structural anchor must still work."""
        boxes = [
            Box(80, 20, 200, 70, "敗北"),
            Box(100, 160, 170, 185, "于一么1"),        # misread チーム 1
            Box(280, 160, 380, 180, "X55/95/43"),
            Box(410, 160, 500, 180, "89,929"),
            Box(100, 440, 170, 465, "于一么2"),
            Box(280, 440, 380, 460, "X74/33/63"),
            Box(410, 440, 500, 460, "96,569"),
        ]
        for cy, name in ((210, "SuperLambda"), (250, "Darius"),
                         (290, "Zaahen"), (330, "Ziggs"), (370, "Morganna"),
                         (490, "Urgot"), (530, "Diana"), (570, "Brand"),
                         (610, "Ashe"), (650, "Poppy")):
            boxes.extend([
                Box(59, cy - 11, 87, cy + 10, "18"),
                Box(194, cy - 9, 250, cy + 8, name),
                Box(193, cy + 6, 250, cy + 22, "タイトル"),
                Box(930, cy - 9, 1030, cy + 9, "%d/%d/%d" % (10, 5, 5)),
                Box(1040, cy - 9, 1140, cy + 9, "20,000"),
                Box(1130, cy - 9, 1230, cy + 9, "12,000"),
                Box(1130, cy + 12, 1240, cy + 26, "300/分"),
            ])
        match = parse_scoreboard(boxes, IMG_W, IMG_H, self.db)
        self.assertEqual(len(match.players), 10, match.warnings)
        self.assertEqual(match.team1_outcome, "DEFEAT")
        teams = {p.name: p.team for p in match.players}
        self.assertEqual([teams["SuperLambda"], teams["Darius"],
                          teams["Zaahen"], teams["Ziggs"], teams["Morganna"]],
                         [1, 1, 1, 1, 1])
        self.assertEqual([teams["Urgot"], teams["Diana"], teams["Brand"],
                          teams["Ashe"], teams["Poppy"]],
                         [2, 2, 2, 2, 2])
        # human vs bots
        self.assertFalse(next(p for p in match.players
                              if p.name == "SuperLambda").is_bot)
        self.assertTrue(next(p for p in match.players
                             if p.name == "Poppy").is_bot)
        self.assertTrue(next(p for p in match.players
                             if p.name == "Morganna").is_bot)

    def test_team_split_without_anchors(self):
        """No team labels at all: split at the widest vertical gap."""
        boxes = []
        team1_cys = [200, 240, 280, 320, 360]
        team2_cys = [520, 560, 600, 640, 680]      # wide gap between teams
        for index, cy in enumerate(team1_cys + team2_cys):
            boxes.extend([
                Box(59, cy - 11, 87, cy + 10, "18"),
                Box(194, cy - 9, 250, cy + 8, "Veigar" if index else "Tomwwy"),
                Box(193, cy + 6, 250, cy + 22, "title"),
                Box(930, cy - 9, 1030, cy + 9, "5/5/5"),
                Box(1040, cy - 9, 1140, cy + 9, "20,000"),
                Box(1130, cy - 9, 1230, cy + 9, "12,000"),
                Box(1130, cy + 12, 1240, cy + 26, "300/分"),
            ])
        match = parse_scoreboard(boxes, IMG_W, IMG_H, self.db)
        self.assertEqual(len(match.players), 10, match.warnings)
        self.assertEqual([p.team for p in match.players],
                         [1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    def test_icon_boxes_are_not_names(self):
        """A stray icon ('->') must not become the player name."""
        boxes = [
            Box(80, 20, 200, 70, "胜利"),
            Box(60, 125, 130, 150, "第1队"),
            Box(140, 125, 220, 148, "81/51/83"),
            Box(270, 320, 300, 340, "->"),          # icon misread
        ]
        for cy, name in ((160, "Kennen"), (200, "Lux"), (240, "Taric")):
            boxes.extend([
                Box(59, cy - 11, 87, cy + 10, "18"),
                Box(194, cy - 9, 250, cy + 8, name),
                Box(193, cy + 6, 250, cy + 22, "title"),
                Box(930, cy - 9, 1030, cy + 9, "5/5/5"),
                Box(1040, cy - 9, 1140, cy + 9, "20,000"),
                Box(1130, cy - 9, 1230, cy + 9, "12,000"),
            ])
        match = parse_scoreboard(boxes, IMG_W, IMG_H, self.db)
        names = [p.name for p in match.players]
        self.assertEqual(names, ["Kennen", "Lux", "Taric"])


    def test_japanese_data_view(self):
        """The 'データ' tab: split KDA cells, CS+gold columns, h:mm:ss length."""
        boxes = [
            Box(80, 125, 220, 155, "勝利"),
            Box(300, 150, 700, 180, "廿一U。力夕。1:02:37。2026/04/02。一ID"),
            Box(80, 570, 180, 595, "千一么2"),           # misread チーム 2
            Box(230, 295, 330, 320, "90/57/94X"),       # team totals, left third
            Box(460, 295, 550, 320, "120,447"),
            Box(230, 570, 330, 595, "57/90/85X"),
            Box(460, 570, 550, 595, "121,456"),
        ]
        rows = [
            (360, "18", "SuperLambda", ["21/"], 371, "30,632"),
            (405, "20", "Malphite", ["13/11/24"], 268, "25,124"),
            (448, "18", "Sejuani", ["18/6/14"], 197, "20,479"),
            (491, "18", "Sion", ["12/15/23"], 96, "17,696"),
            (535, "18", "Swain", ["26/23/26"], 207, "26,516"),
            (634, "20", "Kayle", ["8", "/14/17"], 357, "26,105"),
            (678, "18", "LeeSin", ["8", "/18/22"], 200, "20,810"),
            (722, "18", "Corki", ["19/14/9"], 373, "27,903"),
            (765, "18", "Caitlyn", ["20/22/17"], 287, "30,142"),
            (810, "18", "Pantheon", ["2/22/20"], 73, "16,496"),
        ]
        for cy, level, name, kda_parts, cs, gold in rows:
            boxes.extend([
                Box(100, cy - 11, 130, cy + 10, level),
                Box(150, cy - 9, 260, cy + 8, name),
                Box(150, cy + 6, 260, cy + 22, "タイトル"),
                Box(820, cy - 9, 880, cy + 9, str(cs)),
                Box(910, cy - 9, 1010, cy + 9, gold),
            ])
            for offset, part in enumerate(kda_parts):
                boxes.append(Box(660 + offset * 60, cy - 9,
                                 660 + offset * 60 + 90, cy + 9, part))
        match = parse_scoreboard(boxes, IMG_W, IMG_H, self.db)
        # SuperLambda's KDA ("21 /") is truncated beyond recovery
        self.assertEqual(len(match.players), 9, match.warnings)
        self.assertEqual(match.team1_outcome, "VICTORY")
        self.assertEqual(match.duration_sec, 3757)          # 1:02:37
        by_name = {p.name: p for p in match.players}
        # split KDA cells are stitched back together
        self.assertEqual((by_name["Kayle"].kills, by_name["Kayle"].deaths,
                          by_name["Kayle"].assists), (8, 14, 17))
        self.assertEqual((by_name["LeeSin"].kills, by_name["LeeSin"].deaths,
                          by_name["LeeSin"].assists), (8, 18, 22))
        # data view: CS + gold — no damage column, so damage stays empty
        self.assertIsNone(by_name["Malphite"].damage)
        self.assertEqual(by_name["Malphite"].gold, 25124)
        self.assertEqual(by_name["Kayle"].gold, 26105)
        # teams still split correctly via the structural anchors
        self.assertEqual(by_name["Malphite"].team, 1)
        self.assertEqual(by_name["Pantheon"].team, 2)
        self.assertEqual(by_name["Caitlyn"].team, 2)
        # humans and bots (SuperLambda is the only human, row lost to OCR)
        self.assertTrue(by_name["Malphite"].is_bot)
        self.assertTrue(by_name["Corki"].is_bot)
        self.assertTrue(by_name["Sion"].is_bot)


class TestRealFolder(unittest.TestCase):
    def test_collect_images(self):
        folder = Path(__file__).resolve().parent.parent / \
            "league_of_legends_bot_matches_records"
        if not folder.is_dir():
            self.skipTest("screenshot folder not present")
        images = collect_images(folder)
        self.assertGreater(len(images), 50)
        self.assertTrue(all(p.suffix.lower() == ".png" for p in images))


if __name__ == "__main__":
    unittest.main(verbosity=2)
