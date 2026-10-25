"""LoL Bot Heroes Power Rank — Streamlit dashboard.

Run locally with one click (``run.bat``) or manually::

    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageDraw

from botrank.champions import (CUSTOM_CHAMPION_PATH, DEFAULT_CHAMPION_PATH,
                               ChampionDB, save_custom_aliases)
from botrank.demo import seed_demo_data
from botrank.ocr import create_engine
from botrank.parser import (KDA_RE, NUM_THOUSANDS_RE, PLAIN_NUM_RE, Box,
                            _TEAM_PATTERNS)
from botrank.pipeline import collect_images, sha1_of
from botrank.scoring import build_leaderboard, human_vs_bot_summary, side_split
from botrank.storage import Store
from botrank.vision_api import VisionEngine

# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="LoL Bot Heroes · Power Rank",
                   page_icon="🏆", layout="wide", initial_sidebar_state="expanded")

DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "botrank.db"
DEFAULT_FOLDER = ROOT / "league_of_legends_bot_matches_records"

MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}

CUSTOM_CSS = """
<style>
    .block-container { padding-top: 1.2rem; max-width: 1400px; }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(255,255,255,.05), rgba(255,255,255,.02));
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 12px;
        padding: .6rem .9rem;
    }
    div[data-testid="stMetric"] label { font-size: .78rem; opacity: .75; }
    .scan-box {
        border: 1px dashed rgba(255,255,255,.25);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
    }
    section[data-testid="stSidebar"] { background: rgba(0,0,0,.25); }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# cached resources
# --------------------------------------------------------------------------- #

@st.cache_resource(show_spinner=False)
def get_store() -> Store:
    return Store(DB_PATH)


@st.cache_resource(show_spinner=False)
def get_champions() -> ChampionDB:
    return ChampionDB.load()


@st.cache_resource(show_spinner=False)
def get_engine(kind: str, lang: str, base_url: str, api_key: str, model: str):
    if kind == "vision":
        if not (base_url and api_key and model):
            raise RuntimeError(
                "Vision API needs base URL, API key and model name.")
        return VisionEngine(base_url, api_key, model)
    return create_engine(kind, lang)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def pick_folder() -> Optional[str]:
    """Native folder picker (Windows/macOS/Linux desktops)."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(parent=root,
                                       title="Select screenshots folder")
        root.destroy()
        return path or None
    except Exception as exc:                                  # pragma: no cover
        st.warning("Folder picker unavailable (%s) — paste the path instead."
                   % exc)
        return None


def fmt_int(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    try:
        return "{:,}".format(int(value))
    except (TypeError, ValueError):
        return str(value)


def medal(rank: int) -> str:
    return MEDALS.get(rank, "")


def records_df() -> pd.DataFrame:
    records = get_store().all_records()
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["is_bot"] = frame["is_bot"].astype(bool)
    return frame


def leaderboard_frame(min_games: int = 1) -> pd.DataFrame:
    records = get_store().all_records()
    rows = build_leaderboard(records, min_games=min_games)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame["medal"] = frame["rank"].map(medal)
    frame["w_l"] = frame.apply(lambda r: "%d-%d" % (r["wins"], r["losses"]),
                               axis=1)
    frame["kda"] = frame.apply(
        lambda r: "%.1f / %.1f / %.1f" % (r["avg_kills"], r["avg_deaths"],
                                          r["avg_assists"]), axis=1)
    return frame


def score_chart(frame: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = frame.head(top_n).iloc[::-1]
    labels = ["%s %s" % (medal(int(r["rank"])), r["champion"])
              for _, r in top.iterrows()]
    scores = top["power_score"].tolist()
    figure = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker=dict(color=scores, showscale=False,
                    colorscale=[[0, "#3d5a80"], [0.5, "#3fa7a0"],
                                [1, "#f2c14e"]]),
        text=["%.1f" % s for s in scores], textposition="outside",
        textfont=dict(color="white"),
        hovertemplate="<b>%{y}</b><br>Power Score %{x:.1f}<extra></extra>",
    ))
    figure.update_layout(
        template="plotly_dark", height=max(280, 34 * len(top) + 90),
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(range=[0, 108], title="Power Score"),
        yaxis=dict(title=""), bargap=0.35,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    return figure


def render_leaderboard(min_games: int, top_n: int = 15,
                       key_prefix: str = "tab") -> None:
    """Render the ranked leaderboard.

    ``key_prefix`` keeps element IDs unique when this is rendered repeatedly
    inside one run (the live scan updates call it with ``live<n>`` prefixes).
    """
    frame = leaderboard_frame(min_games)
    if frame.empty:
        st.info("No bot records yet — run a scan from the sidebar.")
        return
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(score_chart(frame, top_n), width="stretch",
                        key="%s_chart" % key_prefix)
    with right:
        st.dataframe(
            frame[["medal", "rank", "champion", "games", "w_l", "win_rate",
                   "kda", "kda_ratio", "avg_damage", "avg_gold",
                   "power_score"]],
            column_config={
                "medal": st.column_config.TextColumn("", width="small"),
                "rank": st.column_config.NumberColumn("Rank", width="small"),
                "champion": st.column_config.TextColumn("Champion", width="medium"),
                "games": st.column_config.NumberColumn("Games", width="small"),
                "w_l": st.column_config.TextColumn("W-L", width="small"),
                "win_rate": st.column_config.ProgressColumn(
                    "Win %", format="percent", min_value=0, max_value=1),
                "kda": st.column_config.TextColumn("Avg K/D/A", width="medium"),
                "kda_ratio": st.column_config.NumberColumn(
                    "KDA", format="%.2f", width="small"),
                "avg_damage": st.column_config.NumberColumn(
                    "Avg Damage", format="%,d"),
                "avg_gold": st.column_config.NumberColumn(
                    "Avg Gold", format="%,d"),
                "power_score": st.column_config.ProgressColumn(
                    "Power Score", format="%.1f", min_value=0, max_value=100),
            },
            hide_index=True, width="stretch", height=430,
            key="%s_table" % key_prefix)

    # --- champion drill-down ------------------------------------------------
    champions = frame["champion"].tolist()
    choice = st.selectbox("Champion detail", champions,
                          key="%s_detail" % key_prefix)
    if choice:
        row = frame[frame["champion"] == choice].iloc[0]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Power Score", "%.1f" % row["power_score"])
        c2.metric("Games / W-L", "%d · %s" % (row["games"], row["w_l"]))
        c3.metric("Win rate", "%.0f%%" % (100 * row["win_rate"]))
        c4.metric("Avg KDA", "%.2f" % row["kda_ratio"])
        c5.metric("Avg Damage", fmt_int(row["avg_damage"]))
        history = records_df()
        history = history[(history["is_bot"]) &
                          (history["champion_name"] == choice)]
        history = history.sort_values(["match_time", "id"])
        if not history.empty:
            chart_data = history[["damage", "gold"]].astype(float).tail(30)
            st.caption("Damage & gold over the last %d bot games"
                       % len(chart_data))
            trend = go.Figure()
            trend.add_trace(go.Scatter(
                y=chart_data["damage"], name="Damage",
                line=dict(color="#4cc9f0", width=2)))
            trend.add_trace(go.Scatter(
                y=chart_data["gold"], name="Gold",
                line=dict(color="#f2c14e", width=2)))
            trend.update_layout(
                template="plotly_dark", height=220,
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(trend, width="stretch",
                            key="%s_trend" % key_prefix)
            st.dataframe(
                history[["match_time", "file_name", "team", "kills", "deaths",
                         "assists", "damage", "gold", "outcome", "confidence"]],
                column_config={
                    "match_time": st.column_config.DatetimeColumn("Time"),
                    "confidence": st.column_config.ProgressColumn(
                        "OCR conf.", format="%.2f", min_value=0, max_value=1),
                },
                hide_index=True, width="stretch",
                key="%s_history" % key_prefix)


def render_insights() -> None:
    frame = records_df()
    if frame.empty:
        st.info("No records yet — run a scan from the sidebar.")
        return
    summary = human_vs_bot_summary(frame.to_dict("records"))
    st.subheader("Bots vs Humans")
    cols = st.columns(2)
    for column, (label, data) in zip(cols, [("🤖 Bots", summary["bot"]),
                                            ("🧑 Humans", summary["human"])]):
        with column:
            st.markdown("**%s**" % label)
            m1, m2, m3 = st.columns(3)
            m1.metric("Games", data["games"])
            m2.metric("Win rate", "%.0f%%" % (100 * data["win_rate"]))
            m3.metric("KDA", "%.2f" % data["kda_ratio"])
            m4, m5 = st.columns(2)
            m4.metric("Avg damage", fmt_int(data["avg_damage"]))
            m5.metric("Avg gold", fmt_int(data["avg_gold"]))

    st.subheader("Bot win rate by side")
    split = side_split(frame.to_dict("records"))
    side_frame = pd.DataFrame({
        "side": ["Team 1 (blue)", "Team 2 (red)"],
        "win_rate": [split["team1"]["win_rate"], split["team2"]["win_rate"]],
        "games": [split["team1"]["games"], split["team2"]["games"]],
    })
    figure = go.Figure(go.Bar(
        x=side_frame["side"], y=side_frame["win_rate"],
        text=["%.0f%% (%d games)" % (100 * r.win_rate, r.games)
              for r in side_frame.itertuples()],
        textposition="outside", marker_color=["#4c8dff", "#ff5c5c"]))
    figure.update_layout(template="plotly_dark", height=260,
                         yaxis=dict(range=[0, 1], tickformat="%"),
                         margin=dict(l=10, r=10, t=10, b=10),
                         paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(figure, width="stretch")

    st.subheader("Gold vs Damage (bot champions)")
    board = build_leaderboard(frame.to_dict("records"))
    if board:
        scatter = pd.DataFrame(board)
        scatter = scatter[scatter["games"] >= 2]
        if not scatter.empty:
            figure = go.Figure(go.Scatter(
                x=scatter["avg_gold"], y=scatter["avg_damage"],
                mode="markers+text",
                marker=dict(size=scatter["games"] * 2.2,
                            color=scatter["win_rate"],
                            colorscale=[[0, "#ff5c5c"], [1, "#3fa7a0"]],
                            showscale=True, colorbar=dict(title="Win rate")),
                text=scatter["champion"], textposition="top center",
                hovertemplate=("<b>%{text}</b><br>Gold %{x:,.0f}<br>"
                               "Damage %{y:,.0f}<extra></extra>")))
            figure.update_layout(template="plotly_dark", height=460,
                                 xaxis_title="Average gold",
                                 yaxis_title="Average damage",
                                 margin=dict(l=10, r=10, t=30, b=10),
                                 paper_bgcolor="rgba(0,0,0,0)",
                                 plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(figure, width="stretch")


def render_records() -> None:
    frame = records_df()
    if frame.empty:
        st.info("No records yet — run a scan from the sidebar.")
        return
    st.caption("Fix OCR mistakes inline, then save — the leaderboard updates "
               "immediately.")

    with st.expander("Filters", expanded=True):
        f1, f2, f3, f4 = st.columns(4)
        champions = sorted(frame.loc[frame["is_bot"], "champion_name"]
                           .dropna().unique().tolist())
        pick = f1.multiselect("Champion", champions)
        side = f2.multiselect("Team", [1, 2], format_func=lambda t: "Team %d" % t)
        outcome = f3.multiselect("Outcome", ["VICTORY", "DEFEAT", "UNKNOWN"])
        only_bots = f4.checkbox("Bots only", value=True)

    view = frame.copy()
    if pick:
        view = view[view["champion_name"].isin(pick)]
    if side:
        view = view[view["team"].isin(side)]
    if outcome:
        view = view[view["outcome"].isin(outcome)]
    if only_bots:
        view = view[view["is_bot"]]

    columns = ["id", "match_time", "file_name", "champion_name", "raw_name",
               "team", "kills", "deaths", "assists", "damage", "gold",
               "outcome", "confidence"]
    view = view[columns].sort_values("id", ascending=False)

    edited = st.data_editor(
        view,
        column_config={
            "id": st.column_config.NumberColumn("ID", disabled=True),
            "match_time": st.column_config.DatetimeColumn("Time", disabled=True),
            "file_name": st.column_config.TextColumn("File", disabled=True),
            "champion_name": st.column_config.TextColumn("Champion"),
            "raw_name": st.column_config.TextColumn("OCR name", disabled=True),
            "team": st.column_config.NumberColumn("Team"),
            "kills": st.column_config.NumberColumn("K", min_value=0),
            "deaths": st.column_config.NumberColumn("D", min_value=0),
            "assists": st.column_config.NumberColumn("A", min_value=0),
            "damage": st.column_config.NumberColumn("Damage", min_value=0),
            "gold": st.column_config.NumberColumn("Gold", min_value=0),
            "outcome": st.column_config.SelectboxColumn(
                "Outcome", options=["VICTORY", "DEFEAT", "UNKNOWN"]),
            "confidence": st.column_config.ProgressColumn(
                "OCR conf.", format="%.2f", min_value=0, max_value=1),
        },
        disabled=["id", "match_time", "file_name", "raw_name", "confidence"],
        num_rows="fixed",
        hide_index=True, width="stretch", height=520,
        key="records_editor")

    save_col, del_col, export_col = st.columns(3)
    if save_col.button("💾 Save corrections", width="stretch"):
        changed = 0
        for _, row in edited.iterrows():
            original = view[view["id"] == row["id"]]
            if original.empty:
                continue
            original = original.iloc[0]
            diff = {}
            for column in ("champion_name", "team", "kills", "deaths",
                           "assists", "damage", "gold", "outcome"):
                if str(original[column]) != str(row[column]):
                    diff[column] = row[column]
            if diff:
                get_store().update_record(int(row["id"]), diff)
                changed += 1
        if changed:
            st.success("Saved %d corrected record(s)." % changed)
            st.rerun()
        else:
            st.info("Nothing to save.")

    doomed = del_col.multiselect("Delete rows (by ID)", view["id"].tolist())
    if del_col.button("🗑 Delete selected", width="stretch") and doomed:
        get_store().delete_records(doomed)
        st.success("Deleted %d record(s)." % len(doomed))
        st.rerun()

    csv = view.drop(columns=["id"]).to_csv(index=False).encode("utf-8")
    export_col.download_button("⬇ Download CSV", csv,
                               file_name="botrank_records.csv",
                               mime="text/csv",
                               width="stretch")


def render_debug_tab(folder: str, engine_kind: str, lang: str) -> None:
    st.caption("Inspect what the OCR engine actually reads — useful when a "
               "row looks wrong in the leaderboard.")
    if not folder or not Path(folder).is_dir():
        st.warning("Select a valid screenshots folder in the sidebar.")
        return
    images = collect_images(folder)
    if not images:
        st.warning("No images found in %s" % folder)
        return
    names = [p.name for p in images[:500]]
    choice = st.selectbox("Screenshot", names)
    path = images[names.index(choice)]

    cache_key = "%s|%s|%s" % (path, engine_kind, lang)
    if st.button("🔍 Run OCR") or cache_key in st.session_state:
        try:
            with st.spinner("Running OCR…"):
                if cache_key not in st.session_state:
                    engine = get_engine(engine_kind, lang, "", "", "")
                    st.session_state[cache_key] = engine.extract(
                        path, get_champions())
                match = st.session_state[cache_key]
            annotated, _boxes = annotate_image(path, engine_kind, lang)
            st.image(annotated, width="stretch")
            st.caption("Yellow = KDA · Cyan = damage/gold · Purple = team "
                       "header · Blue/Red = outcome banner · Grey = other text")
            if match.players:
                players = pd.DataFrame([{
                    "name": p.name, "bot": "🤖" if p.is_bot else "🧑",
                    "champion": p.champion_name or "—", "match": p.name_match,
                    "team": p.team, "lvl": p.level,
                    "K": p.kills, "D": p.deaths, "A": p.assists,
                    "damage": p.damage, "gold": p.gold, "gpm": p.gold_per_min,
                    "outcome": p.outcome, "conf": p.confidence,
                } for p in match.players])
                st.dataframe(players, hide_index=True, width="stretch")
            meta1, meta2 = st.columns(2)
            meta1.metric("Detected outcome", match.team1_outcome)
            meta2.metric("Match length",
                         "%d:%02d" % divmod(match.duration_sec or 0, 60))
            for warning in match.warnings:
                st.warning(warning)
        except Exception as exc:
            st.error("OCR failed: %s" % exc)
            st.info("Install the engine dependencies with:  "
                    "pip install -r requirements.txt")


@st.cache_data(show_spinner=False)
def annotate_image(path: str, engine_kind: str, lang: str):
    """OCR + box overlay (cached per file)."""
    engine = get_engine(engine_kind, lang, "", "", "")
    image = Image.open(path).convert("RGB")
    boxes = engine.ocr_boxes(path)
    draw = ImageDraw.Draw(image)
    for box in boxes:
        text = box.text.strip()
        if KDA_RE.match(text):
            color = (255, 230, 80)
        elif NUM_THOUSANDS_RE.match(text) or PLAIN_NUM_RE.match(text):
            color = (80, 220, 255)
        elif any(pattern.search(text) for _, pattern in _TEAM_PATTERNS):
            color = (200, 120, 255)
        else:
            color = (170, 170, 170)
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=color,
                       width=2)
    return image, boxes


def render_data_tab() -> None:
    store = get_store()
    champions = get_champions()

    st.subheader("Champion database")
    c1, c2, c3 = st.columns(3)
    c1.metric("Champions", len(champions))
    c2.metric("Source", champions.source or "—")
    if DEFAULT_CHAMPION_PATH.exists():
        stamp = pd.to_datetime(DEFAULT_CHAMPION_PATH.stat().st_mtime,
                               unit="s").strftime("%Y-%m-%d %H:%M")
    else:
        stamp = "—"
    c3.metric("Last update", stamp)

    if st.button("🔄 Refresh from Data Dragon (needs internet)"):
        with st.spinner("Downloading official champion names…"):
            try:
                payload = champions.refresh_from_ddragon()
                st.success("Updated to %d champions (patch %s)."
                           % (len(payload["champions"]), payload["version"]))
                st.cache_resource.clear()
                st.rerun()
            except Exception as exc:
                st.error("Refresh failed: %s" % exc)

    st.markdown("**Custom bot-name aliases** — one per line, "
                "`Champion = alias1, alias2 …`")
    current = ""
    if CUSTOM_CHAMPION_PATH.exists():
        import json
        data = json.loads(CUSTOM_CHAMPION_PATH.read_text(encoding="utf-8"))
        current = "\n".join("%s = %s" % (k, ", ".join(v))
                            for k, v in data.items())
    aliases_text = st.text_area("Aliases", current, height=110,
                                key="aliases_editor",
                                placeholder="Veigar = 小法师, 大魔王")
    if st.button("💾 Save aliases"):
        mapping: Dict[str, List[str]] = {}
        for line in aliases_text.splitlines():
            if "=" not in line:
                continue
            canonical, rest = line.split("=", 1)
            names = [n.strip() for n in re_split_aliases(rest) if n.strip()]
            if canonical.strip() and names:
                mapping[canonical.strip()] = names
        save_custom_aliases(mapping)
        st.cache_resource.clear()
        st.success("Aliases saved.")
        st.rerun()

    st.subheader("Database")
    stats = store.stats()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Screenshots processed", stats["files_ok"])
    m2.metric("Player records", stats["records"])
    m3.metric("Bot records", stats["bot_records"])
    m4.metric("Bot champions seen", stats["bot_champions"])

    failed = store.failed_files()
    if failed:
        with st.expander("⚠ %d file(s) failed to parse" % len(failed)):
            st.dataframe(pd.DataFrame(failed), hide_index=True,
                         width="stretch")

    st.subheader("Export")
    e1, e2 = st.columns(2)
    if e1.button("⬇ Export records CSV"):
        frame = records_df()
        if frame.empty:
            st.warning("Nothing to export.")
        else:
            st.download_button("Download records CSV",
                               frame.to_csv(index=False).encode("utf-8"),
                               file_name="botrank_records.csv",
                               mime="text/csv")
    if e2.button("⬇ Export leaderboard CSV"):
        frame = leaderboard_frame()
        if frame.empty:
            st.warning("Nothing to export.")
        else:
            st.download_button("Download leaderboard CSV",
                               frame.to_csv(index=False).encode("utf-8"),
                               file_name="botrank_leaderboard.csv",
                               mime="text/csv")

    st.subheader("Danger zone")
    d1, d2 = st.columns(2)
    if d1.button("🧪 Seed demo data", width="stretch"):
        inserted = seed_demo_data(store)
        st.success("Inserted %d demo records." % inserted)
        st.rerun()
    if d2.button("🧹 Clear all data", width="stretch"):
        store.clear()
        st.success("All records cleared.")
        st.rerun()


def re_split_aliases(text: str) -> List[str]:
    import re
    return re.split(r"[,，;；]", text)


# --------------------------------------------------------------------------- #
# sidebar
# --------------------------------------------------------------------------- #

def render_sidebar() -> dict:
    with st.sidebar:
        st.title("🏆 Bot Heroes")
        st.caption("Power Rank for LoL bot-game screenshots")

        folder = st.text_input(
            "Screenshots folder", st.session_state.get("folder", ""),
            key="folder_input",
            help="Folder containing your post-game screenshots.")
        buttons = st.columns(2)
        if buttons[0].button("📂 Browse", width="stretch"):
            picked = pick_folder()
            if picked:
                st.session_state.folder = picked
                st.rerun()
        if buttons[1].button("↺ Default", width="stretch"):
            if DEFAULT_FOLDER.is_dir():
                st.session_state.folder = str(DEFAULT_FOLDER)
                st.rerun()
            else:
                st.warning("Default folder not found.")

        st.markdown("---")
        st.markdown("**OCR engine**")
        engine_kind = st.selectbox(
            "Engine", ["paddle", "rapid", "vision"],
            format_func=lambda k: {"paddle": "PaddleOCR (offline, accurate)",
                                   "rapid": "RapidOCR (offline, light)",
                                   "vision": "Vision API (LLM)"}[k],
            key="engine_kind")
        lang = st.selectbox(
            "OCR language", ["ch", "en", "japan", "korean"],
            format_func=lambda l: {"ch": "Chinese client (中文)",
                                   "en": "English client",
                                   "japan": "Japanese client (日本語)",
                                   "korean": "Korean client (한국어)"}[l],
            key="ocr_lang")

        with st.expander("Vision API settings", expanded=False):
            st.text_input("Base URL", key="vision_base",
                          placeholder="https://api.openai.com/v1")
            st.text_input("API key", key="vision_key", type="password")
            st.text_input("Model", key="vision_model",
                          placeholder="gpt-4o-mini / qwen-vl-max …")

        with st.expander("Scan options", expanded=False):
            skip_ok = st.checkbox("Skip already-processed screenshots",
                                  value=True, key="skip_ok")
            recursive = st.checkbox("Scan sub-folders too", value=False,
                                    key="recursive")
            max_files = st.number_input("Max screenshots per scan (0 = all)",
                                        min_value=0, value=0, step=10)
            min_games = st.slider("Minimum games on leaderboard", 1, 20, 1,
                                  key="min_games")

        st.markdown("---")
        scan = st.button("🚀 Scan & Rank", type="primary",
                         width="stretch")
        st.caption("First run downloads the OCR models (~10–30 MB).")

        return {"folder": folder, "engine_kind": engine_kind, "lang": lang,
                "skip_ok": skip_ok, "recursive": recursive,
                "max_files": int(max_files) if max_files else None,
                "min_games": min_games, "scan": scan}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def run_scan(options: dict) -> None:
    folder = Path(options["folder"])
    if not folder.is_dir():
        st.error("Folder not found: %s" % folder)
        return
    try:
        engine = get_engine(options["engine_kind"], options["lang"],
                            st.session_state.get("vision_base", ""),
                            st.session_state.get("vision_key", ""),
                            st.session_state.get("vision_model", ""))
    except Exception as exc:
        st.error("Could not start the OCR engine: %s" % exc)
        st.info("Install the dependencies with:  pip install -r requirements.txt")
        return

    images = collect_images(folder, recursive=options["recursive"])
    if not images:
        st.warning("No screenshots found in %s" % folder)
        return

    store = get_store()
    champions = get_champions()
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    live = st.empty()

    def on_progress(done: int, total: int, name: str, stats: dict) -> None:
        fraction = done / total if total else 1.0
        progress_bar.progress(fraction)
        status_text.caption("OCR %d / %d · %s" % (done, total, name))
        if done % 3 == 0 or done == total:
            with live.container():
                st.markdown("#### Live leaderboard")
                render_leaderboard(options["min_games"],
                                   key_prefix="live%d" % done)

    from botrank.pipeline import process_folder
    stats = process_folder(
        folder, engine, store, champions,
        progress_cb=on_progress,
        skip_ok=options["skip_ok"],
        max_files=options["max_files"])
    progress_bar.empty()
    status_text.empty()
    if stats["processed"] or stats["failed"] or stats["skipped"]:
        message = ("Scan finished — **%d** new screenshot(s) parsed, "
                   "**%d** skipped, **%d** failed · **%d** player rows "
                   "(**%d** bots)"
                   % (stats["processed"], stats["skipped"], stats["failed"],
                      stats["new_records"], stats["bot_records"]))
        (st.success if not stats["failed"] else st.warning)(message)
    else:
        st.info("Nothing to do — every screenshot is already processed. "
                "Uncheck *Skip already-processed* to force a re-scan.")


def main() -> None:
    if "folder" not in st.session_state:
        st.session_state.folder = (str(DEFAULT_FOLDER)
                                   if DEFAULT_FOLDER.is_dir() else "")
    options = render_sidebar()

    st.markdown("## 🏆 LoL Bot Heroes · Power Rank")
    st.caption("OCR every post-game screenshot, separate bots from humans, "
               "and rank the strongest bot heroes.")

    if options["scan"]:
        run_scan(options)

    # --- header metrics -----------------------------------------------------
    store = get_store()
    stats = store.stats()
    folder_path = Path(options["folder"]) if options["folder"] else None
    found = len(collect_images(folder_path)) if folder_path and \
        folder_path.is_dir() else 0
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Screenshots in folder", found)
    m2.metric("Processed", stats["files_ok"])
    m3.metric("Bot rows", stats["bot_records"])
    m4.metric("Bot champions", stats["bot_champions"])
    m5.metric("Last scan", (stats["last_scan"] or "—").replace("T", " "))

    tabs = st.tabs(["🏆 Leaderboard", "📊 Insights", "📋 Records",
                    "🔍 OCR Debug", "⚙️ Data"])
    with tabs[0]:
        render_leaderboard(options["min_games"])
    with tabs[1]:
        render_insights()
    with tabs[2]:
        render_records()
    with tabs[3]:
        render_debug_tab(options["folder"], options["engine_kind"],
                         options["lang"])
    with tabs[4]:
        render_data_tab()


if __name__ == "__main__":
    main()
