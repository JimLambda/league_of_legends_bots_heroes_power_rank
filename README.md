# 🏆 LoL Bot Heroes — Power Rank

A local Python + Streamlit app that OCRs your folder of League of Legends
**post-game screenshots**, separates **bot** players from **human** players,
extracts KDA / damage / gold / match outcome, and renders a **live ranked
leaderboard** of the strongest bot heroes with a composite **Power Score**.

```
run.bat   ←  double-click this (Windows, one click)
```

---

## 1. Requirements

| Requirement | Notes |
|---|---|
| Windows 10/11 (or macOS/Linux) | Folder picker needs a desktop session |
| Python **3.9 – 3.12** | paddlepaddle has no wheels for 3.13/3.14 — on those versions the app automatically falls back to RapidOCR |
| ~4 GB free disk | Most of it is the PaddleOCR/paddlepaddle install |
| Internet access (first run) | Downloads OCR model weights (~10–30 MB) once |

## 2. Run it with one click

1. Double-click **`run.bat`**.
2. The first run creates a virtual environment (`.venv`), installs
   `requirements.txt` (this can take 5–15 minutes, mostly PaddleOCR), then
   starts the dashboard and opens <http://localhost:8501> in your browser.

That's it — every later start skips straight to the dashboard.

### Manual start (any OS)

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## 3. Using the dashboard

1. **Pick your screenshots folder** — the sidebar is pre-filled with
   `league_of_legends_bot_matches_records` when it sits next to `app.py`
   (or use **📂 Browse**).
2. Choose the **OCR engine** (PaddleOCR by default) matching your game
   client language (Chinese / English / Japanese / Korean).
3. Press **🚀 Scan & Rank**. The leaderboard **updates live** while the OCR
   runs; already-processed screenshots are skipped on re-scans
   (content-hash based, so renamed files are still recognised).
4. Explore the tabs:

| Tab | What it shows |
|---|---|
| 🏆 **Leaderboard** | Ranked bot heroes by Power Score, top-15 bar chart, per-champion drill-down (history + damage/gold trend) |
| 📊 **Insights** | Bots vs humans comparison, bot win rate by side, gold-vs-damage map |
| 📋 **Records** | Every parsed player row; **fix OCR mistakes inline** and save; CSV export |
| 🔍 **OCR Debug** | Annotated screenshot + parsed rows for any single file — the first place to look when a number seems off |
| ⚙️ **Data** | Champion database info, Data Dragon refresh, custom bot-name aliases, export, demo data, danger zone |

## 4. How it works

```
screenshots/ ──► OCR (PaddleOCR / RapidOCR / Vision API)
                     │  text boxes with coordinates
                     ▼
              parser.py
                     │  • KDA cells anchor the 10 player rows
                     │  • level + name left of KDA, damage/gold/gpm right of it
                     │  • column headers (advanced-details view) pin the columns
                     │  • banner text (胜利 / VICTORY / 敗北 …) + colour fallback
                     ▼
              champions.py   ──► bot = name matches a champion name
                     │         (exact, "Veigar Bot" affixes, fuzzy OCR noise)
                     ▼
              scoring.py     ──► Power Score = 0.35·(K+A) + 0.30·damage
                     │             + 0.20·gold + 0.15·win rate
                     ▼             (shrunk 3 games toward the mean, min-max normed)
              storage.py (SQLite, content-hash dedupe)
                     ▼
              Streamlit dashboard (live leaderboard)
```

**Bot detection.** In bot games, bot players are named after their champion
(`Veigar`, `Sejuani`, `Kennen`…), while humans have custom summoner names.
The bundled champion database (`data/champions.json`, ~170 champions with
English/Chinese/partial Japanese names) is matched against every player
name. The **Refresh from Data Dragon** button (Data tab, needs internet)
downloads the official names for 13 locales, which also covers brand-new
champions your bundled copy may not know yet.

**Power Score.** Each metric is first shrunk toward the global mean with a
prior of 3 games (so a 1-game fluke doesn't top the chart), then min-max
normalised across the field:

| Component | Weight |
|---|---|
| Kill participation — avg (kills + assists) | 35 % |
| Damage — avg total damage to champions | 30 % |
| Gold — avg gold earned | 20 % |
| Win rate — bot's team result | 15 % |

**Accuracy.** The parser is heuristic; the **OCR Debug** tab shows exactly
what was read for any screenshot, and the **Records** tab lets you correct
any misread value (the leaderboard recalculates instantly). Low-confidence
rows are flagged in the Records table.

**Vision API option.** Instead of local OCR you can point the app at any
OpenAI-compatible vision endpoint (OpenAI, DashScope/Qwen, vLLM, Ollama…):
sidebar → *Vision API settings* → base URL + API key + model → engine
“Vision API (LLM)”. The model returns structured JSON, which is usually the
most accurate route.

## 5. Project layout

```
├── run.bat                 one-click Windows launcher
├── requirements.txt
├── app.py                  Streamlit dashboard
├── botrank/                core package (stdlib-only, fully unit-tested)
│   ├── champions.py        champion DB, bot matching, Data Dragon refresh
│   ├── ocr.py              PaddleOCR / RapidOCR engines
│   ├── vision_api.py       optional LLM vision engine
│   ├── parser.py           scoreboard parsing (rows, stats, outcome, bots)
│   ├── scoring.py          Power Score + leaderboard aggregation
│   ├── storage.py          SQLite persistence
│   ├── pipeline.py         folder scanning, hashing, progress callbacks
│   └── demo.py             demo-data seeder (Data tab)
├── data/
│   ├── champions.json      bundled champion names (offline fallback)
│   └── botrank.db          created on first run
├── tests/test_core.py      25 unit tests:  python -m unittest discover -s tests
├── tests/test_apptest.py   Streamlit render smoke tests (AppTest)
└── tools/
    ├── debug_boxes.py      dump raw OCR boxes for one screenshot
    └── full_scan.py        scan the whole folder and print a quality report
```

The parser handles the layouts seen in the wild: the default and
*advanced details* scoreboards (zh / en / ja / ko clients), the Japanese
**データ** tab (CS + gold columns, no damage), split KDA cells (`8` +
`/14/17`), MVP stars glued to damage (`★135,034`), misread team labels
(`チーム 1` → `于一么1`), and duplicate bot names in one match (two Veigars).

## 6. Troubleshooting

| Symptom | Fix |
|---|---|
| `paddlepaddle` install fails | You're on Python 3.13/3.14. Use Python 3.9–3.12, or switch the engine to **RapidOCR** and remove `paddle*` from `requirements.txt`. |
| App is slow | PaddleOCR on CPU ≈ 2–4 s/screenshot. Switch to RapidOCR, or use the Vision API engine. |
| "No scoreboard rows detected" | The screenshot isn't a post-game scoreboard (lobby, loading screen, in-game HUD). Check the OCR Debug tab. |
| A bot is classified as human (or vice versa) | Add an alias in Data → *Custom bot-name aliases* (`Veigar = 小法师`), refresh Data Dragon, or correct the row in the Records tab. |
| Scan does nothing | Every file is already processed — uncheck *Skip already-processed screenshots*. |
| Browse button does nothing | Paste the folder path manually (no desktop/Tkinter in some environments). |
