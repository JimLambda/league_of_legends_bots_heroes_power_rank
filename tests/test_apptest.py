"""Smoke-test the Streamlit app with AppTest (renders every tab)."""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest

from botrank.demo import seed_demo_data
from botrank.storage import Store

REAL_FOLDER = ROOT / "league_of_legends_bot_matches_records"


def run_app(timeout: int = 90) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=timeout)
    at.run()
    return at


def check(at: AppTest, label: str) -> None:
    problems = [e for e in at.exception if e]
    if problems:
        for problem in problems:
            print("EXCEPTION (%s): %s" % (label, problem))
        raise SystemExit(1)
    print("OK: %s" % label)


def main() -> None:
    # 1. empty database — default render of all five tabs
    at = run_app()
    check(at, "empty database renders")
    metrics = at.get("metric")
    assert metrics, "header metrics missing"

    # 2. populated database (demo data) — leaderboard / insights / records
    store = Store(ROOT / "data" / "botrank.db")
    seed_demo_data(store)
    store.close()
    at = run_app()
    check(at, "demo data renders")
    check(at, "tabs present: %d" % len(at.tabs))
    tables = (at.get("arrow_data_frame") or at.get("dataframe"))
    assert tables, "leaderboard table missing"
    charts = (at.get("plotly_chart") or at.get("arrow_vega_lite_chart")
              or at.get("line_chart"))
    assert charts, "no charts rendered"

    # 3. scan path with an unavailable engine (paddle not installed here)
    with tempfile.TemporaryDirectory() as tmp:
        if REAL_FOLDER.is_dir():
            for image in list(REAL_FOLDER.glob("*.png"))[:2]:
                shutil.copy(image, tmp)
        at = run_app()
        at.text_input(key="folder_input").set_value(tmp).run()
        at.button(key=None) if False else None
        # click the primary scan button
        scan_buttons = [b for b in at.button if "Scan" in str(b.label)]
        assert scan_buttons, "scan button not found"
        scan_buttons[0].click().run(timeout=120)
        check(at, "scan with missing paddle engine fails gracefully")

    # 4. clear the database again for a clean handover
    store = Store(ROOT / "data" / "botrank.db")
    store.clear()
    store.close()

    # 5. OCR Debug tab: switch to the light engine and run a real OCR pass
    at = run_app()
    at.selectbox(key="engine_kind").set_value("rapid").run()
    check(at, "engine switch renders")
    ocr_buttons = [b for b in at.button if "Run OCR" in b.label]
    assert ocr_buttons, "OCR debug button not found"
    ocr_buttons[0].click().run(timeout=180)
    check(at, "OCR debug tab runs a real screenshot")
    images = at.get("imgs")
    assert images, "annotated screenshot not rendered"

    store = Store(ROOT / "data" / "botrank.db")
    store.clear()
    store.close()
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
