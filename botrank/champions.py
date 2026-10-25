"""Champion database: loading, bot-name matching, and Data Dragon refresh.

Bot players in League of Legends bot games are named after their champion
(e.g. ``Veigar``, ``Sejuani``), while human players use custom summoner
names.  A player row is therefore classified as a *bot* when its OCR'd name
matches a champion name in any supported client locale.

The bundled database (``data/champions.json``) ships English + Chinese names
and a partial Japanese/Korean set.  Use :meth:`ChampionDB.refresh_from_ddragon`
once (button in the app's *Data* tab) to pull every official name for all
locales from Riot's Data Dragon CDN.
"""
from __future__ import annotations

import json
import re
import unicodedata
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DEFAULT_CHAMPION_PATH = DATA_DIR / "champions.json"
CUSTOM_CHAMPION_PATH = DATA_DIR / "custom_champions.json"

DD_VERSIONS_URL = "https://ddragon.leagueoflegends.com/api/versions.json"
DD_CHAMPION_URL = ("https://ddragon.leagueoflegends.com/cdn/{version}/data/"
                   "{locale}/champion.json")

#: locale -> field name stored in the champion JSON
LOCALES = {
    "en_US": "en", "zh_CN": "zh", "zh_TW": "zh_tw", "ja_JP": "ja",
    "ko_KR": "ko", "de_DE": "de", "es_ES": "es", "fr_FR": "fr",
    "it_IT": "it", "pl_PL": "pl", "pt_BR": "pt", "ru_RU": "ru", "tr_TR": "tr",
}
#: name fields consulted when building the match index
NAME_FIELDS = ("en", "zh", "zh_tw", "ja", "ko", "de", "es", "fr", "it",
               "pl", "pt", "ru", "tr")

# Characters kept by ``normalize_name``: latin letters (incl. accents),
# Greek, Cyrillic, kana, CJK unified ideographs and Hangul syllables.
_KEEP_RE = re.compile(
    r"[^0-9a-z\u00c0-\u024f\u0370-\u04ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]+")


def normalize_name(text: str) -> str:
    """Normalize a player/champion name for comparison.

    Lower-cases, applies NFKC (full-width -> half-width) and removes every
    character that is not a letter or digit of the supported scripts.
    ``"Bel'Veth"``, ``"bel veth"`` and ``"ベル＝ヴェス"`` all normalize to a
    stable form.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    return _KEEP_RE.sub("", text)


def levenshtein(a: str, b: str, cap: int = 2) -> int:
    """Edit distance with early exit once it exceeds ``cap``."""
    if a == b:
        return 0
    if abs(len(a) - len(b)) > cap:
        return cap + 1
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1,
                               current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


@dataclass
class MatchResult:
    """Outcome of matching an OCR'd player name against the champion DB."""
    champion: dict
    method: str          # "exact" | "fuzzy"
    score: float         # 1.0 for exact, similarity for fuzzy

    @property
    def champion_id(self) -> str:
        return self.champion.get("id", "")

    @property
    def champion_name(self) -> str:
        return self.champion.get("en") or self.champion.get("id", "")


def _strip_bot_affix(name: str) -> Optional[str]:
    """Handle names such as ``Veigar Bot`` / ``bot Veigar`` / ``VeigarAI``."""
    for suffix in ("bot", "ai"):
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            return name[: -len(suffix)]
    for prefix in ("bot", "ai"):
        if name.startswith(prefix) and len(name) > len(prefix) + 2:
            return name[len(prefix):]
    return None


class ChampionDB:
    """In-memory champion name index with exact + fuzzy matching."""

    def __init__(self, champions: List[dict], source: str = ""):
        self.champions = champions
        self.source = source
        self._index: Dict[str, dict] = {}
        for champ in champions:
            for alias in self._aliases_of(champ):
                self._index.setdefault(alias, champ)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @staticmethod
    def _aliases_of(champ: dict) -> List[str]:
        aliases = set()
        for field in NAME_FIELDS:
            norm = normalize_name(champ.get(field, ""))
            if len(norm) >= 2:
                aliases.add(norm)
        for alias in champ.get("aliases", []):
            norm = normalize_name(alias)
            if len(norm) >= 2:
                aliases.add(norm)
        return sorted(aliases)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "ChampionDB":
        """Load the bundled database merged with the user's custom aliases."""
        path = Path(path) if path else DEFAULT_CHAMPION_PATH
        payload: dict = {}
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                payload = {}
        champions = [dict(c) for c in payload.get("champions", [])]

        if CUSTOM_CHAMPION_PATH.exists():
            try:
                custom = json.loads(CUSTOM_CHAMPION_PATH.read_text(
                    encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                custom = {}
            by_id = {c.get("id"): c for c in champions}
            by_name = {normalize_name(c.get("en", "")): c for c in champions}
            for canonical, aliases in custom.items():
                target = by_id.get(canonical) or by_name.get(
                    normalize_name(canonical))
                if target is not None:
                    target.setdefault("aliases", []).extend(aliases)
                else:  # teach a brand-new bot name that is not a champion
                    champions.append({"id": canonical, "en": canonical,
                                      "aliases": list(aliases)})
        return cls(champions, source=str(payload.get("source", "bundled")))

    # ------------------------------------------------------------------ #
    # matching
    # ------------------------------------------------------------------ #
    def __len__(self) -> int:
        return len(self.champions)

    def match(self, raw_name: str) -> Optional[MatchResult]:
        """Return a :class:`MatchResult` when ``raw_name`` is a bot name.

        Exact (normalized) matches are tried first, including names carrying
        a ``bot``/``ai`` affix.  As a second pass a conservative fuzzy match
        absorbs OCR noise (``Morganna`` -> ``Morgana``) while rejecting
        look-alike human names (``Kenneth`` stays human).
        """
        name = normalize_name(raw_name)
        if len(name) < 2:
            return None

        for candidate in (name, _strip_bot_affix(name)):
            if candidate and candidate in self._index:
                return MatchResult(self._index[candidate], "exact", 1.0)

        best: Optional[MatchResult] = None
        for alias, champ in self._index.items():
            if abs(len(alias) - len(name)) > 2:
                continue
            if alias[0] != name[0]:
                continue
            distance = levenshtein(alias, name, cap=2)
            if distance == 0:
                continue
            # tolerate one edit from 5 chars up (absorbs OCR noise such as
            # "Tarie" -> "Taric" or "Morganna" -> "Morgana"), two edits for
            # longer names; shorter names must match exactly.
            limit = 1 if len(name) >= 5 else 0
            if len(name) >= 8:
                limit = 2
            if distance <= limit:
                similarity = 1.0 - distance / max(len(alias), len(name))
                if best is None or similarity > best.score:
                    best = MatchResult(champ, "fuzzy", similarity)
        return best

    # ------------------------------------------------------------------ #
    # Data Dragon refresh
    # ------------------------------------------------------------------ #
    def refresh_from_ddragon(self, timeout: int = 30) -> dict:
        """Download official champion names for all locales and save them.

        Returns the payload that was written to ``data/champions.json``.
        Requires internet access; the bundled database stays untouched on
        failure.
        """
        with urllib.request.urlopen(DD_VERSIONS_URL, timeout=timeout) as resp:
            version = json.loads(resp.read().decode("utf-8"))[0]

        merged: Dict[str, dict] = {c["id"]: dict(c) for c in self.champions
                                   if c.get("id")}
        for locale, field in LOCALES.items():
            url = DD_CHAMPION_URL.format(version=version, locale=locale)
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            for champ_id, info in payload.get("data", {}).items():
                entry = merged.setdefault(
                    champ_id, {"id": champ_id,
                               "en": info.get("name", champ_id),
                               "aliases": []})
                entry[field] = info.get("name", "")
                if locale == "en_US":
                    entry["en"] = info.get("name", champ_id)

        for champ_id, entry in merged.items():
            entry.setdefault("aliases", [])
            key = normalize_name(champ_id)          # e.g. "MonkeyKing"
            if len(key) >= 3 and key not in entry["aliases"]:
                entry["aliases"].append(champ_id)

        payload = {
            "version": version,
            "source": "ddragon:%s" % version,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "champions": sorted(merged.values(),
                                key=lambda c: c.get("en", "")),
        }
        DEFAULT_CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEFAULT_CHAMPION_PATH.write_text(
            json.dumps(payload, ensure_ascii=False, indent=1),
            encoding="utf-8")
        return payload


def save_custom_aliases(mapping: Dict[str, List[str]]) -> None:
    """Persist user-taught aliases (``{"Veigar": ["小法师"]}``)."""
    CUSTOM_CHAMPION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CUSTOM_CHAMPION_PATH.write_text(
        json.dumps(mapping, ensure_ascii=False, indent=1), encoding="utf-8")
