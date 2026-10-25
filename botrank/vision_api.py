"""Optional vision-model engine (OpenAI-compatible chat completions API).

When a vision-capable LLM endpoint is configured (any provider exposing the
``/chat/completions`` schema: OpenAI, DashScope, vLLM, Ollama's OpenAI
shim, …), the screenshot is sent as base64 and the model returns structured
JSON — usually the most accurate option, at the cost of an API key.
"""
from __future__ import annotations

import base64
import io
import json
import re
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

from .champions import ChampionDB
from .parser import ParsedMatch, PlayerRow

PROMPT = """You are a League of Legends post-game scoreboard reader.
Look at the screenshot and extract the full scoreboard.
Return ONLY a JSON object (no markdown, no comments) with this schema:
{
  "outcome": "VICTORY" or "DEFEAT",      // result of the local player's team (team 1)
  "duration": "mm:ss",                   // match length shown at the top, or null
  "players": [
    {
      "name": "<player name exactly as displayed>",
      "team": 1 or 2,
      "level": <int or null>,
      "kills": <int>, "deaths": <int>, "assists": <int>,
      "damage": <int or null>,           // total damage to champions
      "gold": <int or null>              // gold earned
    }
  ]
}
Rules:
- Include ALL 10 players (5 per team), team 1 first.
- Bot players are named after their champion (e.g. "Veigar", "Sejuani");
  human players have custom names. Copy the name text exactly.
- Use plain integers without thousands separators.
"""


class VisionEngine:
    """Send screenshots to an OpenAI-compatible vision model."""

    name = "vision-api"

    def __init__(self, base_url: str, api_key: str, model: str,
                 max_side: int = 1600, timeout: int = 120):
        self.base_url = (base_url or "").rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.max_side = max_side
        self.timeout = timeout

    # ------------------------------------------------------------------ #
    def _encode_image(self, image_path: Union[str, Path]) -> str:
        from PIL import Image
        image = Image.open(str(image_path))
        image.load()
        if max(image.size) > self.max_side:
            ratio = self.max_side / max(image.size)
            image = image.resize((round(image.width * ratio),
                                  round(image.height * ratio)),
                                 Image.LANCZOS)
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=88)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _request(self, image_path: Union[str, Path]) -> str:
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {
                        "url": "data:image/jpeg;base64,%s"
                               % self._encode_image(image_path)}},
                ],
            }],
            "temperature": 0,
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json",
                     "Authorization": "Bearer %s" % self.api_key},
            method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

    @staticmethod
    def _to_int(value) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(re.sub(r"[,.，\s]", "", str(value)))
        except (TypeError, ValueError):
            return None

    def extract(self, image_path: Union[str, Path],
                champ_db: ChampionDB) -> ParsedMatch:
        raw = self._request(image_path)
        data = self._parse_json(raw)
        if not data:
            match = ParsedMatch(warnings=["Vision API returned no valid JSON."])
            return match

        match = ParsedMatch()
        outcome = str(data.get("outcome", "")).upper()
        match.team1_outcome = outcome if outcome in ("VICTORY", "DEFEAT") \
            else "UNKNOWN"
        match.outcome = match.team1_outcome
        duration = data.get("duration")
        if isinstance(duration, str) and ":" in duration:
            parts = [int(p) for p in duration.split(":") if p.isdigit()]
            if len(parts) == 2:
                match.duration_sec = parts[0] * 60 + parts[1]
            elif len(parts) == 3:
                match.duration_sec = parts[0] * 3600 + parts[1] * 60 + parts[2]

        for entry in data.get("players", []):
            name = str(entry.get("name", "")).strip()
            team = 2 if int(entry.get("team", 1) or 1) == 2 else 1
            outcome_for_player = match.team1_outcome
            if team == 2 and outcome_for_player in ("VICTORY", "DEFEAT"):
                outcome_for_player = ("DEFEAT" if outcome_for_player == "VICTORY"
                                      else "VICTORY")
            result = champ_db.match(name) if name else None
            match.players.append(PlayerRow(
                name=name, raw_name=name,
                level=self._to_int(entry.get("level")),
                kills=self._to_int(entry.get("kills")) or 0,
                deaths=self._to_int(entry.get("deaths")) or 0,
                assists=self._to_int(entry.get("assists")) or 0,
                damage=self._to_int(entry.get("damage")),
                gold=self._to_int(entry.get("gold")),
                team=team, confidence=0.95,
                is_bot=result is not None,
                champion_id=result.champion_id if result else None,
                champion_name=result.champion_name if result else None,
                name_match=result.method if result else "none",
                outcome=outcome_for_player,
            ))
        if len(match.players) != 10:
            match.warnings.append(
                "Vision API returned %d players (expected 10)."
                % len(match.players))
        return match
