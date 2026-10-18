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
