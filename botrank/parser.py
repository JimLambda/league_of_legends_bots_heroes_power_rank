"""Parse a League of Legends post-game scoreboard from OCR text boxes.

The parser is layout-driven and tolerant to OCR noise:

* **Outcome** — the big banner text (``VICTORY`` / ``DEFEAT`` / 胜利 / 敗北 …) in
  any client locale, with a blue-vs-red banner-colour fallback.
* **Player rows** — anchored on the KDA cells (``5 / 12 / 16``).  Every OCR
  box is attached to its nearest KDA row; boxes left of the KDA hold the
  level and the player/champion name, boxes right of it hold damage, gold
  and gold-per-minute.
* **Column mapping** — when the *advanced details* scoreboard is open, the
  real column headers (DAMAGE / GOLD / CS …) are detected and used to pin
  the damage/gold columns; otherwise the x-order fallback (damage first,
  gold second) is used.
* **Bot detection** — delegated to :class:`~botrank.champions.ChampionDB`:
  a row is a bot when its name matches a champion name.
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .champions import ChampionDB, MatchResult, normalize_name

# --------------------------------------------------------------------------- #
# data structures
# --------------------------------------------------------------------------- #


@dataclass
class Box:
    """One OCR'd text box (normalised from PaddleOCR/RapidOCR output)."""
    x1: float
    y1: float
    x2: float
    y2: float
    text: str
    score: float = 1.0

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1


@dataclass
class PlayerRow:
    """A single scoreboard row (human or bot)."""
    name: str = ""
    raw_name: str = ""
    level: Optional[int] = None
    kills: int = 0
    deaths: int = 0
    assists: int = 0
    damage: Optional[int] = None
    gold: Optional[int] = None
    gold_per_min: Optional[int] = None
    team: int = 1
    is_bot: bool = False
    champion_id: Optional[str] = None
    champion_name: Optional[str] = None
    name_match: str = "none"        # exact | fuzzy | none
    confidence: float = 0.0
    outcome: str = "UNKNOWN"

    @property
    def kda_ratio(self) -> float:
        return (self.kills + self.assists) / max(self.deaths, 1)


@dataclass
class ParsedMatch:
    """Everything extracted from one screenshot."""
    outcome: str = "UNKNOWN"               # local player's team result
    team1_outcome: str = "UNKNOWN"
    duration_sec: Optional[int] = None
    players: List[PlayerRow] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_boxes: int = 0

    @property
    def bots(self) -> List[PlayerRow]:
        return [p for p in self.players if p.is_bot]

    @property
    def humans(self) -> List[PlayerRow]:
        return [p for p in self.players if not p.is_bot]


# --------------------------------------------------------------------------- #
# patterns
# --------------------------------------------------------------------------- #

KDA_RE = re.compile(r"^\s*(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})\s*$")
# same, but for searching inside joined OCR fragments ("8" + "/14/17")
KDA_SEARCH_RE = re.compile(r"(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(\d{1,2})")
# team-total KDA cell, e.g. "81 / 51 / 83" or OCR junk like "X55/95/43".
# Deliberately strict (digits + slashes + at most 2 junk chars) so that the
# top bar "…1:02:37。2026/04/02…" — whose date contains "026/04/02" — cannot
# be mistaken for a team header.
TEAM_TOTAL_RE = re.compile(
    r"^.{0,2}\s*\d{1,3}\s*/\s*\d{1,3}\s*/\s*\d{1,3}\s*.{0,2}$")
TWO_DIGIT_RE = re.compile(r"^\d{1,2}$")
# thousands separators: 38,299 / 38，299 / 38.299 (European)
NUM_THOUSANDS_RE = re.compile(r"^\d{1,3}(?:[,，.]\d{3})+$")
PLAIN_NUM_RE = re.compile(r"^\d{4,}$")          # commas dropped by OCR
GPM_RE = re.compile(r"^\s*(\d{2,4})\s*[/／/]\s*\D{0,6}$")
RATIO_RE = re.compile(r"^\s*\d{1,2}[.,]\d\s*(kda|战损比|kd)?\s*$", re.I)
SMALL_INT_RE = re.compile(r"^\d{1,3}$")
DURATION_HMS_RE = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})")
DURATION_SEARCH_RE = re.compile(r"(\d{1,2})[:：](\d{2})")

_TEAM_PATTERNS: List[Tuple[int, "re.Pattern[str]"]] = [
    (1, re.compile(r"第\s*1\s*[队隊]")),
    (2, re.compile(r"第\s*2\s*[队隊]")),
    (1, re.compile(r"チーム\s*1")),
    (2, re.compile(r"チーム\s*2")),
    (1, re.compile(r"team\s*1", re.I)),
    (2, re.compile(r"team\s*2", re.I)),
    (1, re.compile(r"팀\s*1")),
    (2, re.compile(r"팀\s*2")),
    (1, re.compile(r"(?:équipe|equipe|equipo)\s*1", re.I)),
    (2, re.compile(r"(?:équipe|equipe|equipo)\s*2", re.I)),
    (1, re.compile(r"команда\s*1", re.I)),
    (2, re.compile(r"команда\s*2", re.I)),
]

VICTORY_WORDS = ("victory", "胜利", "勝利", "vitoria", "sieg", "victoire",
                 "triunfo", "victòria", "victoria", "승리")
DEFEAT_WORDS = ("defeat", "失败", "失敗", "败北", "敗北", "derrota",
                "niederlage", "défaite", "defaite", "패배", "패전")

# column headers (only present in the "advanced details" scoreboard view)
TAKEN_HDR_RE = re.compile(
    r"(taken|承受|受けた|받은|받은\s*피해|erlitten|received|subi)", re.I)
DAMAGE_HDR_RE = re.compile(
    r"(damage|dmg|伤害|ダメージ|与えた|피해|데미지|schaden|dano|daño|dégâts)", re.I)
GOLD_HDR_RE = re.compile(r"(gold|金币|ゴールド|골드|oro)", re.I)
CS_HDR_RE = re.compile(
    r"(^|\W)(cs|creep|minion|兵|ミニオン|jungle)(\W|$)", re.I)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _ratio(a: str, b: str) -> float:
    """Cheap similarity in the spirit of difflib.SequenceMatcher."""
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    longer = max(len(a), len(b))
    return matches / longer if longer else 0.0


def _cluster_by_cy(items: Sequence[Box], tol: float) -> List[List[Box]]:
    clusters: List[List[Box]] = []
    for box in sorted(items, key=lambda b: b.cy):
        if clusters and abs(box.cy - clusters[-1][-1].cy) <= tol:
            clusters[-1].append(box)
        else:
            clusters.append([box])
    return clusters


def _to_int(text: str) -> Optional[int]:
    """``"38,299"`` / ``"38.299"`` / ``"38299"`` -> ``38299``."""
    cleaned = re.sub(r"[,.，\s]", "", text.strip())
    if not cleaned.isdigit():
        return None
    return int(cleaned)


def _stat_value(text: str) -> Optional[int]:
    """Extract a stat number, tolerating OCR junk (``★137,675``, icons …)."""
    cleaned = re.sub(r"[^\d,.，]", "", text)
    if NUM_THOUSANDS_RE.match(cleaned) or PLAIN_NUM_RE.match(cleaned):
        return _to_int(cleaned)
    return None


_NAME_CHAR_RE = re.compile(
    r"[0-9A-Za-z\u00c0-\u024f\u0370-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")


def _is_name_like(text: str) -> bool:
    """A player name needs at least two name characters (no icons/junk)."""
    return len(_NAME_CHAR_RE.findall(text)) >= 2


def _clean_name(text: str) -> str:
    """First line of the name box, without leading level digits."""
    first = text.split("\n")[0].strip()
    first = re.sub(r"^\d{1,2}\s+", "", first)     # merged level: "18 Darius"
    first = first.strip(" \t|:：·-—~*")
    return first.strip()


def _team_index(text: str) -> Optional[int]:
    for index, pattern in _TEAM_PATTERNS:
        if pattern.search(text):
            return index
    return None


def _color_outcome(image) -> str:
    """Blue-vs-red banner colour fallback (victory = blue, defeat = red)."""
    try:
        width, height = image.size
        region = image.crop((0, 0, int(width * 0.32), int(height * 0.12)))
        region = region.convert("RGB").resize((160, 48))
        blue = red = 0
        for r, g, b in region.getdata():
            if b > r + 25 and b > 90:
                blue += 1
            elif r > b + 25 and r > 90:
                red += 1
        if blue > red * 1.2 and blue > 50:
            return "VICTORY"
        if red > blue * 1.2 and red > 50:
            return "DEFEAT"
    except Exception:
        pass
    return "UNKNOWN"


def _detect_outcome(boxes: Sequence[Box], img_w: int, img_h: int,
                    image=None) -> str:
    candidates: List[Tuple[str, Box]] = []
    for box in boxes:
        if box.cy > img_h * 0.2:
            continue
        text = normalize_name(box.text)
        if len(text) < 2:
            continue
        for word in VICTORY_WORDS:
            target = normalize_name(word)
            if text == target or (len(target) >= 4
                                  and _ratio(text, target) >= 0.82):
                candidates.append(("VICTORY", box))
                break
        else:
            for word in DEFEAT_WORDS:
                target = normalize_name(word)
                if text == target or (len(target) >= 4
                                      and _ratio(text, target) >= 0.82):
                    candidates.append(("DEFEAT", box))
                    break
    if candidates:
        candidates.sort(key=lambda c: -(c[1].w * c[1].h))
        return candidates[0][0]
    if image is not None:
        return _color_outcome(image)
    return "UNKNOWN"


def _detect_duration(boxes: Sequence[Box], img_h: int) -> Optional[int]:
    """Match length from the top bar (often merged with other text)."""
    for box in sorted(boxes, key=lambda b: b.cy):
        if box.cy > img_h * 0.20:
            break
        long_form = DURATION_HMS_RE.search(box.text)
        if long_form:
            return (int(long_form.group(1)) * 3600
                    + int(long_form.group(2)) * 60
                    + int(long_form.group(3)))
        found = DURATION_SEARCH_RE.search(box.text)
        if found:
            return int(found.group(1)) * 60 + int(found.group(2))
    return None


def _merge_span(existing, new) -> Tuple[float, float]:
    if existing is None:
        return new
    return (min(existing[0], new[0]), max(existing[1], new[1]))


def _find_headers(boxes: Sequence[Box],
                  first_row_cy: float) -> Dict[str, Tuple[float, float]]:
    """Locate DAMAGE / GOLD / CS column headers above the first player row."""
    headers: Dict[str, Tuple[float, float]] = {}
    for box in boxes:
        if not (first_row_cy - 260 < box.cy < first_row_cy - 20):
            continue
        text = box.text.strip()
        if not text or len(text) > 24:
            continue
        span = (box.x1 - 10, box.x2 + 10)
        if TAKEN_HDR_RE.search(text):
            headers["damage_taken"] = _merge_span(headers.get("damage_taken"), span)
        elif DAMAGE_HDR_RE.search(text):
            headers["damage"] = _merge_span(headers.get("damage"), span)
        elif GOLD_HDR_RE.search(text):
            headers["gold"] = _merge_span(headers.get("gold"), span)
        elif CS_HDR_RE.search(text):
            headers["cs"] = _merge_span(headers.get("cs"), span)
    return headers


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #


def parse_scoreboard(boxes: Sequence[Box], img_w: int, img_h: int,
                     champ_db: Optional[ChampionDB] = None,
                     image=None) -> ParsedMatch:
    """Turn OCR boxes into a :class:`ParsedMatch`."""
    boxes = [b for b in boxes if b.text and b.text.strip()]
    match = ParsedMatch(n_boxes=len(boxes))
    if not boxes:
        match.warnings.append("No OCR text found in image.")
        return match

    # --- team anchors -----------------------------------------------------
    # (a) text anchors: 第1队 / チーム 1 / Team 1 / 팀 1 …
    # (b) structural anchors: the team-total KDA cell always sits in the
    #     left third of the table, above its five players.  This survives
    #     OCR misreads of the label itself (e.g. チーム1 -> 于一么1).
    anchors: Dict[int, List[float]] = {1: [], 2: []}
    for box in boxes:
        index = _team_index(box.text)
        if index is not None and img_h * 0.08 < box.cy < img_h * 0.75:
            anchors[index].append(box.cy)

    kda_boxes = [b for b in boxes if KDA_RE.match(b.text.strip())]
    team_total_boxes = [b for b in boxes
                        if TEAM_TOTAL_RE.match(b.text.strip())
                        and img_w * 0.30 > b.cx
                        and img_h * 0.05 < b.cy < img_h * 0.85]
    for order, box in enumerate(sorted(team_total_boxes,
                                       key=lambda b: b.cy)[:2], start=1):
        anchors[order].append(box.cy)
    team1_cy = min(anchors[1]) if anchors[1] else None
    team2_cy = min(anchors[2]) if anchors[2] else None

    def in_banner(box: Box) -> bool:
        return box.cy < img_h * 0.08

    def near_team_anchor(box: Box, radius: float = 12.0) -> bool:
        for anchor_cy in (team1_cy, team2_cy):
            if anchor_cy is not None and abs(box.cy - anchor_cy) < radius:
                return True
        return False

    player_kda = [b for b in kda_boxes
                  if not in_banner(b)
                  and b.cx >= img_w * 0.30        # team totals sit left
                  and not near_team_anchor(b)]
    if team1_cy is not None:
        player_kda = [b for b in player_kda if b.cy > team1_cy - 8]

    # --- recovery: KDA split into several boxes ("8" + "/14/17") ----------
    if kda_boxes and len(player_kda) < 10:
        col_x1 = min(b.x1 for b in kda_boxes) - 40
        col_x2 = max(b.x2 for b in kda_boxes) + 40
        col_boxes = [b for b in boxes
                     if col_x1 <= b.cx <= col_x2
                     and not in_banner(b) and not near_team_anchor(b)
                     and (team1_cy is None or b.cy > team1_cy - 8)]
        known_cys = [b.cy for b in player_kda]
        for cluster in _cluster_by_cy(col_boxes, tol=12):
            ordered = sorted(cluster, key=lambda b: b.cx)
            joined = "".join(b.text for b in ordered)
            found = KDA_SEARCH_RE.search(joined)
            if found:
                kills, deaths, assists = found.groups()
                fragments = [b for b in ordered
                             if re.fullmatch(r"[\d/\s]+", b.text.strip())]
            else:               # three bare digit boxes, no slashes
                fragments = [b for b in ordered
                             if TWO_DIGIT_RE.match(b.text.strip())]
                if len(fragments) != 3:
                    continue
                kills, deaths, assists = (b.text.strip() for b in fragments)
            source = fragments or cluster
            cy = statistics.median([b.cy for b in cluster])
            if any(abs(cy - known) < 20 for known in known_cys):
                continue
            player_kda.append(Box(
                min(b.x1 for b in source), min(b.y1 for b in source),
                max(b.x2 for b in source), max(b.y2 for b in source),
                "%s/%s/%s" % (kills, deaths, assists),
                min(b.score for b in source)))
            known_cys.append(cy)
        player_kda.sort(key=lambda b: b.cy)

    match.outcome = _detect_outcome(boxes, img_w, img_h, image)
    match.team1_outcome = match.outcome
    match.duration_sec = _detect_duration(boxes, img_h)

    if not player_kda:
        match.warnings.append(
            "No scoreboard rows detected — is this a LoL post-game screenshot?")
        return match
    if len(player_kda) != 10:
        match.warnings.append(
            "Expected 10 player rows, found %d." % len(player_kda))

    # --- row geometry -----------------------------------------------------
    cys = sorted(b.cy for b in player_kda)
    gaps = [b - a for a, b in zip(cys, cys[1:]) if b - a > 5]
    median_gap = statistics.median(gaps) if gaps else 55.0
    row_tol = max(18.0, median_gap * 0.5)

    headers = _find_headers(boxes, min(b.cy for b in player_kda))

    # --- attach every box to its nearest KDA row --------------------------
    rows: List[List[Box]] = [[] for _ in player_kda]
    kda_ids = {id(k) for k in player_kda}
    for box in boxes:
        if id(box) in kda_ids:
            continue
        best_index, best_distance = None, row_tol
        for index, kda in enumerate(player_kda):
            distance = abs(box.cy - kda.cy)
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index is not None:
            rows[best_index].append(box)

    # --- team split fallback: largest vertical gap between the two teams ----
    team_split_cy: Optional[float] = None
    if team2_cy is None and len(player_kda) == 10:
        sorted_cys = sorted(b.cy for b in player_kda)
        gaps = [(sorted_cys[i + 1] - sorted_cys[i], i) for i in range(9)]
        widest, widest_index = max(gaps)
        others = [gap for gap, _ in gaps if gap != widest]
        if widest > 1.4 * (statistics.median(others) if others else 0.0):
            team_split_cy = sorted_cys[widest_index] + widest / 2.0

    # --- build player rows ------------------------------------------------
    for index, kda in enumerate(player_kda):
        kda_match = KDA_RE.match(kda.text.strip())
        kills, deaths, assists = (int(g) for g in kda_match.groups())
        row_boxes = rows[index]

        team = 1
        if team2_cy is not None and kda.cy > team2_cy - 8:
            team = 2
        elif team_split_cy is not None and kda.cy > team_split_cy:
            team = 2

        # left of the KDA: level + player/champion name.  The name is the
        # TOPMOST text box of the row (the champion title / localised name
        # sits underneath it), which is far more reliable than x-ordering
        # because both lines start at the same x position.
        left = [b for b in row_boxes if b.cx < kda.x1 - 15]
        level: Optional[int] = None
        for box in sorted(left, key=lambda b: b.cx):
            text = box.text.strip()
            if (level is None and TWO_DIGIT_RE.match(text)
                    and 1 <= int(text) <= 30):
                level = int(text)
        name_candidates = [b for b in left
                           if not (TWO_DIGIT_RE.match(b.text.strip())
                                   and 1 <= int(b.text.strip()) <= 30)
                           and not RATIO_RE.match(b.text.strip())
                           and _is_name_like(b.text)]
        name_box = (min(name_candidates, key=lambda b: (b.cy, b.x1))
                    if name_candidates else None)
        raw_name = name_box.text if name_box else ""
        name = _clean_name(raw_name) if name_box else ""

        # right of the KDA: damage / gold / gold-per-minute.  The "data"
        # scoreboard view shows CS + gold instead of damage + gold, which is
        # recognised by a single comma-number preceded by a small integer.
        right = sorted((b for b in row_boxes if b.cx > kda.x2 + 5),
                       key=lambda b: b.cx)
        comma_numbers = [(b, _stat_value(b.text)) for b in right]
        comma_numbers = [(b, value) for b, value in comma_numbers
                         if value is not None]
        small_ints = [(b, int(b.text.strip())) for b in right
                      if SMALL_INT_RE.match(b.text.strip())]
        data_view = (len(comma_numbers) == 1 and small_ints
                     and small_ints[0][0].cx < comma_numbers[0][0].cx)

        damage = gold = None
        damage_box = gold_box = None
        if len(comma_numbers) >= 2:
            for box, value in comma_numbers:
                if ("gold" in headers
                        and headers["gold"][0] <= box.cx <= headers["gold"][1]):
                    gold, gold_box = value, box
                elif ("damage" in headers
                      and headers["damage"][0] <= box.cx <= headers["damage"][1]):
                    damage, damage_box = value, box
            for box, value in comma_numbers:         # x-order fallback
                if damage is None and box is not gold_box:
                    damage, damage_box = value, box
                elif gold is None and box is not damage_box:
                    gold, gold_box = value, box
                if damage is not None and gold is not None:
                    break
        elif comma_numbers:
            box, value = comma_numbers[0]
            if data_view:
                gold, gold_box = value, box
            else:
                damage, damage_box = value, box

        gold_per_min = None
        for box in right:
            gpm_match = GPM_RE.match(box.text.strip())
            if gpm_match:
                gold_per_min = int(gpm_match.group(1))
                break

        # --- bot / human classification -----------------------------------
        result: Optional[MatchResult] = None
        if champ_db is not None and name:
            result = champ_db.match(name)

        used_scores = [b.score for b in (name_box, kda, damage_box, gold_box)
                       if b is not None]
        confidence = (sum(used_scores) / len(used_scores)
                      if used_scores else 0.0)

        player = PlayerRow(
            name=name, raw_name=raw_name.strip(), level=level,
            kills=kills, deaths=deaths, assists=assists,
            damage=damage, gold=gold, gold_per_min=gold_per_min,
            team=team, confidence=round(confidence, 3),
            is_bot=result is not None,
            champion_id=result.champion_id if result else None,
            champion_name=result.champion_name if result else None,
            name_match=result.method if result else "none",
        )
        match.players.append(player)

    # --- outcome fallback from team kill totals ---------------------------
    if match.team1_outcome == "UNKNOWN" and len(match.players) >= 2:
        kills = {1: 0, 2: 0}
        for player in match.players:
            kills[player.team] += player.kills
        if kills[1] != kills[2]:
            match.team1_outcome = ("VICTORY" if kills[1] > kills[2]
                                   else "DEFEAT")
        match.warnings.append("Outcome banner unreadable; inferred from kills.")
    for player in match.players:
        if player.team == 1:
            player.outcome = match.team1_outcome
        elif match.team1_outcome in ("VICTORY", "DEFEAT"):
            player.outcome = ("DEFEAT" if match.team1_outcome == "VICTORY"
                              else "VICTORY")
    match.outcome = match.team1_outcome
    return match
