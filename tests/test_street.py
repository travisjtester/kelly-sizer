"""The Street-target starter: analyst targets -> a labeled consensus view."""
import pandas as pd
import pytest

import kelly as ks

INFO = {
    "AAA": {"currentPrice": 100.0, "targetLowPrice": 70.0, "targetMeanPrice": 125.0,
            "targetHighPrice": 160.0, "numberOfAnalystOpinions": 30},
    "BBB": {"currentPrice": 100.0, "targetLowPrice": 110.0, "targetMeanPrice": 140.0,
            "targetHighPrice": 180.0, "numberOfAnalystOpinions": 12},   # low above price
    "CCC": {"currentPrice": 100.0},                                        # no coverage
    "DDD": {"currentPrice": 200.0, "targetLowPrice": 120.0, "targetMeanPrice": 150.0,
            "targetHighPrice": 190.0},                                     # above the high
}


@pytest.fixture
def yahoo(monkeypatch):
    class T:
        def __init__(self, t):
            self.info = INFO.get(t, {})
    monkeypatch.setattr("yfinance.Ticker", T)


def test_targets_become_bear_base_bull_with_placeholder_odds(yahoo):
    v, notes = ks.street_views(ks.street_targets(["AAA"]))
    r = v.loc["AAA"]
    assert (r.p_bear, r.p_base, r.p_bull) == pytest.approx((.2, .5, .3))
    assert (r.bear, r.base, r.bull) == pytest.approx((.30, .25, .60))
    assert r.horizon == 1 and r.as_of_price == 100          # 12-month targets, anchored
    assert "Street consensus starter" in r.notes and "30 analysts" in r.notes
    assert not notes


def test_a_low_target_above_the_price_is_not_a_bear_case(yahoo):
    v, notes = ks.street_views(ks.street_targets(["BBB"]))
    assert v.at["BBB", "bear"] == pytest.approx(ks.STREET_BEAR_FLOOR)
    assert any("BBB: lowest target is above the price" in n for n in notes)


def test_names_without_usable_targets_are_skipped_with_a_reason(yahoo):
    v, notes = ks.street_views(ks.street_targets(["CCC", "DDD"]))
    assert v.empty
    assert any("CCC: skipped, no analyst price targets" in n for n in notes)
    assert any("DDD: skipped, already above the highest target" in n for n in notes)


def test_starter_views_load_like_any_other(yahoo):
    v, _ = ks.street_views(ks.street_targets(["AAA", "BBB"]))
    again = ks.normalize_views(v.reset_index())
    assert set(again.index) == {"AAA", "BBB"}
    with ks.captured_log() as logs:
        ks._warn_horizon_mismatch(again, 3.0)                 # a 3-year run flags them
    assert any("AAA (1y)" in m for m in logs)
