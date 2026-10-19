"""The phone app, driven headlessly with Streamlit's AppTest."""
from datetime import date

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import kelly as ks
from conftest import ROOT


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)                 # no views.csv / journal.csv preloaded
    return AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def size(at, tickers=("NVDA", "ASML", "KO", "MSFT", "VOO")):
    at.multiselect[0].set_value(list(tickers)).run()
    at.button[0].click().run()
    assert not at.exception, at.exception
    return at


def metrics(at):
    return {m.label: m.value for m in at.metric}


def test_capm_only_basket(app):
    at = size(app)
    assert any("every name uses CAPM" in i.value for i in at.info)
    labels = [m.label for m in at.metric]
    assert labels[:3] == ["Chance of loss (3y)", "Worst case, 1-in-20 (3y)", "Exp. growth"]
    assert len(at.toggle) == 0                   # no backtest
    assert any("Add your research to see how sensitive" in c.value for c in at.caption)


def test_max_per_name_capped_at_40(app):
    sl = next(s for s in app.slider if s.label == "Max per name")
    assert sl.max == 0.40 and sl.value == 0.25


def test_views_drive_sizing_and_match_the_cli_helpers(app):
    app.session_state["views"] = {
        "NVDA": {"p_bear": .25, "bear": .5, "p_base": .35, "base": .15, "bull": 1.0, "value": None},
        "ASML": {"p_bear": .2, "bear": .35, "p_base": .4, "base": .2, "bull": None, "value": 2000.0},
    }
    at = size(app)
    assert any("Sized on your research for 2 of 5" in s.value for s in at.success)
    alloc = at.dataframe[1].value
    assert {"ticker", "weight", "dollars", "edge", "margin_of_safety", "solo_kelly",
            "swing"} <= set(alloc.columns)
    assert set(alloc.edge.dropna()) - {""} == {"views", "CAPM"}
    wrong = next(d.value for d in at.dataframe if "Verdict" in d.value.columns)
    assert set(wrong.Verdict) <= {"Robust", "Sensitive", "Fragile", "Not held", ""}
    assert any(h.value.startswith(":red[What if") for h in at.subheader)


def test_journal_log_and_calibration(app):
    app.session_state["views"] = {
        "NVDA": {"p_bear": .25, "bear": .5, "p_base": .35, "base": .15, "bull": 1.0, "value": None}}
    at = size(app)
    next(b for b in at.button if b.label.startswith("Log today")).click().run()
    assert metrics(at)["Calls"] == "1"
    old = pd.DataFrame([{"date": date(2022, 1, 3), "ticker": f"T{i}", "horizon_years": 1,
                         "entry_price": 10, "p_bear": .2, "bear": .4, "p_base": .2,
                         "base": .1, "p_bull": .6, "bull": .5,
                         "outcome": ["bull", "base", "bear"][i % 3]} for i in range(12)])
    at.session_state["journal"] = ks.append_journal(at.session_state["journal"], old)
    at.run()
    assert metrics(at)["Resolved"] == "12"
    assert any("run hot" in i.value for i in at.info)
    assert {d.label for d in at.get("download_button")} >= {"Save views CSV",
                                                           "Save journal CSV",
                                                           "Save these results (CSV)"}
