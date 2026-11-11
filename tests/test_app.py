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
    assert not any("backtest" in t.label.lower() for t in at.toggle)   # no backtest
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
    # Fractions, not percents, so the table's CSV download matches the other exports
    names = alloc[alloc.ticker != "CASH"]
    assert alloc.weight.between(0, 1).all() and alloc.weight.sum() == pytest.approx(1)
    assert pd.api.types.is_numeric_dtype(alloc.solo_kelly)
    assert names.swing.dropna().lt(5).all()
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


def test_loaded_views_join_the_basket(app):
    app.session_state["_add_picks"] = ["TSM", "SNOW"]           # what an upload queues
    at = app.run()
    picks = at.multiselect[0].value
    assert picks[:2] == ["TSM", "SNOW"] and "NVDA" in picks      # added, defaults kept
    assert not at.exception


def test_loaded_views_past_the_cap_are_named(app):
    many = [f"T{i:02d}" for i in range(14)]
    app.session_state["_add_picks"] = many
    at = app.run()
    assert at.multiselect[0].value == many + ["NVDA"]            # 15: the file first
    assert any("didn't fit" in w.value and "ASML" in w.value for w in at.warning)


def test_neutral_mode_warns_that_it_is_a_test(app):
    app.session_state["views"] = {
        "NVDA": {"p_bear": .25, "bear": .5, "p_base": .35, "base": .15, "bull": 1.0, "value": None}}
    next(t for t in app.toggle if t.label.startswith("Neutral")).set_value(True)
    at = size(app)
    assert any("Neutral inputs is on" in w.value for w in at.warning)


def test_untouched_view_keeps_its_price_and_horizon(app):
    at = size(app)
    at.session_state["views"] = {        # as an upload stores it; the table leaves it be
        "NVDA": {"p_bear": .25, "bear": .5, "p_base": .35, "base": .15, "bull": 1.0,
                 "value": None, "as_of_price": 123.0, "horizon": 1.0}}
    at.run()
    assert at.session_state["views"]["NVDA"]["as_of_price"] == 123.0


def test_market_odds_stay_hidden_until_you_have_odds(app):
    at = size(app)
    assert not any(t.label.startswith("Show the market") for t in at.toggle)
    assert any("stay hidden until you've committed" in c.value for c in at.caption)


def test_market_check_shows_beside_your_views(app):
    app.session_state["views"] = {
        "NVDA": {"p_bear": .25, "bear": .5, "p_base": .35, "base": .15, "bull": 1.0, "value": None}}
    at = size(app)
    next(t for t in at.toggle if t.label.startswith("Show the market")).set_value(True).run()
    assert not at.exception
    check = next(d.value for d in at.dataframe if "market_bear" in d.value.columns)
    assert list(check.ticker) == ["NVDA"] and check.your_bear.iloc[0] == pytest.approx(.25)


def test_book_mode_toggle_puts_cash_at_zero(app):
    next(t for t in app.toggle if t.label.startswith("Book mode")).set_value(True)
    at = size(app)
    assert any("Risk-free 0.00% (book mode" in c.value for c in at.caption)


# ---- saved data: a fake private repo in place of GitHub
import github_store as gs  # noqa: E402

SAVED_VIEWS = "ticker,p_bear,bear,p_base,base,bull,value\nTSM,0.2,0.4,0.4,0.15,0.8,\n"


class MemoryStore:
    repo = "me/kelly-data"

    def __init__(self, files=None):
        self.files = {p: (t, f"v{i}") for i, (p, t) in enumerate((files or {}).items())}
        self.writes = []

    def read(self, path):
        return self.files.get(path, (None, None))

    def write(self, path, text, message, sha):
        if sha != self.files.get(path, (None, None))[1]:
            raise gs.Conflict(path)
        new = f"v{len(self.writes) + 10}"
        self.files[path] = (text, new)
        self.writes.append(path)
        return new


@pytest.fixture
def saved(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    repo = MemoryStore({gs.VIEWS_PATH: SAVED_VIEWS})
    monkeypatch.setattr(gs, "from_secrets", lambda secrets: (repo, "open sesame"))
    return repo, AppTest.from_file(str(ROOT / "app.py"), default_timeout=180).run()


def unlock(at, pw):
    at.text_input[0].input(pw)
    next(b for b in at.button if b.label == "Unlock").click().run()
    return at


def test_nothing_loads_until_the_passphrase(saved):
    repo, at = saved
    assert any("Unlock your saved" in e.label for e in at.expander)
    assert at.session_state["views"] == {}
    at = unlock(at, "wrong")
    assert any("Wrong passphrase" in e.value for e in at.error)
    assert at.session_state["views"] == {} and not repo.writes


def test_unlock_loads_your_views_into_the_basket(saved):
    repo, at = saved
    at = unlock(at, "open sesame")
    assert not at.exception
    assert "TSM" in at.session_state["views"] and "TSM" in at.multiselect[0].value
    assert any("save to your private repo (me/kelly-data)" in c.value for c in at.caption)
    assert not repo.writes                          # loading alone doesn't write


def test_changes_save_themselves(saved):
    repo, at = saved
    at = unlock(at, "open sesame")
    at.session_state["views"]["KO"] = {"p_bear": .1, "bear": .15, "p_base": .6, "base": .12,
                                      "bull": .3, "value": None}
    at.run()
    assert gs.VIEWS_PATH in repo.writes and "KO" in repo.files[gs.VIEWS_PATH][0]


def test_two_devices_journal_entries_merge(saved):
    repo, at = saved
    at = unlock(at, "open sesame")
    other = ks.journal_entries(ks.resolve_views(ks.normalize_views(pd.DataFrame(
        {"ticker": ["MU"], "p_bear": [.3], "bear": [.5], "bull": [1.0]})), {"MU": 100.0}), 2.0,
        on=date(2026, 11, 1))
    repo.files[gs.JOURNAL_PATH] = (other.to_csv(index=False), "phone")   # saved elsewhere
    mine = ks.journal_entries(ks.resolve_views(ks.normalize_views(pd.DataFrame(
        {"ticker": ["KO"], "p_bear": [.1], "bear": [.15], "bull": [.3]})), {"KO": 60.0}), 2.0,
        on=date(2026, 11, 2))
    at.session_state["journal"] = ks.append_journal(at.session_state["journal"], mine)
    at.run()
    saved_tickers = set(ks.normalize_journal(pd.read_csv(
        __import__("io").StringIO(repo.files[gs.JOURNAL_PATH][0]))).ticker)
    assert saved_tickers == {"MU", "KO"}


def test_street_button_fills_only_blank_rows(app, monkeypatch):
    class T:
        def __init__(self, t):
            self.info = {"currentPrice": 100.0, "targetLowPrice": 70.0,
                         "targetMeanPrice": 120.0, "targetHighPrice": 150.0}
    monkeypatch.setattr("yfinance.Ticker", T)
    app.session_state["views"] = {
        "NVDA": {"p_bear": .25, "bear": .5, "p_base": .35, "base": .15, "bull": 1.0, "value": None}}
    at = app.run()
    next(b for b in at.button if b.label.startswith("Start blank rows")).click().run()
    views = at.session_state["views"]
    assert views["NVDA"]["bear"] == .5                     # your view untouched
    assert views["ASML"]["horizon"] == 1 and views["ASML"]["p_bear"] == pytest.approx(.2)
    assert any("not your edge" in w.value for w in at.warning)
