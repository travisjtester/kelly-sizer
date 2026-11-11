from datetime import date

import pandas as pd

import kelly as ks


def _calls(n, p_bull, hits):
    return pd.DataFrame([{"date": date(2021, 1, 4), "ticker": f"T{i}", "horizon_years": 1,
                          "entry_price": 10, "p_bear": .1, "bear": .5, "p_base": .9 - p_bull,
                          "base": .1, "p_bull": p_bull, "bull": .6,
                          "outcome": "bull" if i < hits else "base"} for i in range(n)])


def test_log_dedupes_same_day_and_marks_due():
    v = ks.resolve_views(ks.normalize_views(pd.DataFrame(
        {"ticker": ["A"], "p_bear": [.2], "bear": [.4], "bull": [.8]})), {"A": 50.0})
    j = ks.append_journal(ks.journal_entries(v, 3, on=date(2020, 1, 2)),
                          ks.journal_entries(v, 3, on=date(2020, 1, 2)))
    assert len(j) == 1
    entries, s, _ = ks.journal_report(j, {"A": 60.0}, today=date(2026, 10, 18))
    assert s["due"] == 1 and abs(entries.at[0, "return_since_entry"] - .2) < 1e-9


def test_calibration_flags_overconfidence():
    _, s, calib = ks.journal_report(_calls(20, .7, 10), today=date(2026, 10, 18))
    assert s["resolved"] == 20 and "run hot" in s["verdict"]
    assert calib.iloc[0]["calls"] == 20 and calib.iloc[0]["actual_bull"] == .5


def test_calibration_recognizes_good_odds():
    _, s, _ = ks.journal_report(_calls(20, .6, 12), today=date(2026, 10, 18))
    assert "Well calibrated" in s["verdict"]


def test_too_few_calls_to_judge():
    _, s, _ = ks.journal_report(_calls(4, .6, 2), today=date(2026, 10, 18))
    assert "Too few" in s["verdict"]


def test_journal_round_trips_through_csv(tmp_path):
    p = tmp_path / "journal.csv"
    j = ks.normalize_journal(_calls(3, .6, 1))
    j.to_csv(p, index=False)
    back = ks.load_journal(p)
    assert len(back) == 3 and list(back.outcome) == ["bull", "base", "base"]


def _resolved(n, you_bear, mkt_bear, outcome="bear"):
    """n resolved calls where you and the market gave the bear case these odds."""
    return ks.normalize_journal(pd.DataFrame({
        "date": [date(2023, 1, 1)] * n, "ticker": [f"T{i}" for i in range(n)],
        "horizon_years": 1, "p_bear": you_bear, "bear": .4, "p_base": 0.0,
        "p_bull": 1 - you_bear, "bull": .5, "outcome": outcome,
        "market_bear": mkt_bear, "market_bull": 1 - mkt_bear}))


def test_older_journals_without_market_columns_still_load():
    old = pd.DataFrame({"date": ["2025-01-02"], "ticker": ["NVDA"], "p_bear": [.2],
                        "bear": [.5], "p_bull": [.8], "bull": [1.0]})
    j = ks.normalize_journal(old)
    assert {"market_bear", "market_bull", "market_expiry"} <= set(j.columns)
    assert j.market_bear.isna().all() and j.market_expiry.iloc[0] == ""


def test_logging_records_the_markets_odds_beside_yours():
    views = ks.resolve_views(ks.normalize_views(pd.DataFrame(
        {"ticker": ["AAA", "BBB"], "p_bear": [.2, .3], "bear": [.4, .5], "bull": [.6, .9]})),
        {"AAA": 50.0, "BBB": 20.0})
    market = pd.DataFrame({"ticker": ["AAA", "BBB"], "market_bear": [.31, float("nan")],
                           "market_bull": [.22, float("nan")],
                           "expiry": ["2028-01-21", None]})
    e = ks.journal_entries(views, 2.0, on=date(2026, 11, 11), market=market).set_index("ticker")
    assert e.at["AAA", "market_bear"] == .31 and e.at["AAA", "market_expiry"] == "2028-01-21"
    assert pd.isna(e.at["BBB", "market_bear"]) and e.at["BBB", "market_expiry"] == ""


def test_you_beat_the_market_when_your_odds_were_better():
    _, s, _ = ks.journal_report(_resolved(12, you_bear=.8, mkt_bear=.2), today=date(2026, 1, 1))
    assert s["market_calls"] == 12 and s["your_brier_vs"] < s["market_brier"]
    assert "Your odds beat the market's on 12 calls" in s["market_verdict"]


def test_the_market_beats_you_and_says_bet_smaller():
    _, s, _ = ks.journal_report(_resolved(12, you_bear=.1, mkt_bear=.7), today=date(2026, 1, 1))
    assert "The market's odds beat yours" in s["market_verdict"]
    assert "bet a smaller fraction" in s["market_verdict"]


def test_market_comparison_waits_for_enough_calls():
    _, s, _ = ks.journal_report(_resolved(4, .8, .2), today=date(2026, 1, 1))
    assert "needs 10+" in s["market_verdict"]
    _, s2, _ = ks.journal_report(_resolved(12, .8, float("nan")), today=date(2026, 1, 1))
    assert s2["market_calls"] == 0 and "market_verdict" not in s2   # no market odds logged
