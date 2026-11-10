"""
Market-implied odds, checked against answers known in closed form: fake option
chains priced with Black-Scholes at a chosen volatility, so the engine has to
recover that volatility from the quotes and turn it back into odds.
"""
from datetime import date, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import kelly as ks

S, T, RF = 100.0, 1.5, 0.04
R = np.log1p(RF)
STRIKES = np.arange(40.0, 205.0, 5.0)


def chain(vol_at=lambda K: 0.40, spread=0.02, no_bid_below=None):
    """Calls and puts quoted at Black-Scholes prices for vol_at(K), with a
    relative bid/ask spread around that mid."""
    def side(call):
        mid = np.array([ks._bs_price(S, K, T, R, vol_at(K), call) for K in STRIKES])
        bid, ask = mid * (1 - spread / 2), mid * (1 + spread / 2)
        if no_bid_below is not None:
            bid = np.where(STRIKES < no_bid_below, 0.0, bid)
        return pd.DataFrame({"strike": STRIKES, "bid": bid, "ask": ask})
    return side(True), side(False)


def closed_form(K, vol, below):
    d2 = (np.log(S / K) + (R - vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    return norm.cdf(-d2) if below else norm.cdf(d2)


def test_flat_volatility_recovers_the_lognormal_odds():
    calls, puts = chain()
    smile = ks.option_smile(calls, puts, S, T, R)
    assert np.allclose(smile.iv, 0.40, atol=1e-4)               # vol recovered from mids
    pb, pu, note = ks.implied_odds(smile, S, T, R, bear_price=70.0, bull_price=150.0)
    assert pb == pytest.approx(closed_form(70.0, 0.40, True), abs=1e-4)
    assert pu == pytest.approx(closed_form(150.0, 0.40, False), abs=1e-4)
    assert note == ""


def test_skew_raises_the_bear_odds():
    """Far-down puts carry more volatility (crash protection costs more), so
    the bear odds must come out higher than with a flat at-the-money vol."""
    skew = lambda K: 0.40 + 0.30 * max(0.0, (S - K) / S)       # noqa: E731
    calls, puts = chain(vol_at=skew)
    smile = ks.option_smile(calls, puts, S, T, R)
    pb, _, _ = ks.implied_odds(smile, S, T, R, 70.0, 150.0)
    assert pb > closed_form(70.0, 0.40, True) + 0.02
    assert pb == pytest.approx(closed_form(70.0, skew(70.0), True), abs=1e-3)


def test_between_strikes_interpolates():
    calls, puts = chain()
    smile = ks.option_smile(calls, puts, S, T, R)
    pb, _, _ = ks.implied_odds(smile, S, T, R, 72.5, 150.0)     # strikes are every $5
    assert pb == pytest.approx(closed_form(72.5, 0.40, True), abs=1e-4)


def test_illiquid_or_untraded_prices_say_so_instead_of_guessing():
    calls, puts = chain(no_bid_below=80.0)                       # no bids under $80
    smile = ks.option_smile(calls, puts, S, T, R)
    pb, pu, note = ks.implied_odds(smile, S, T, R, 70.0, 150.0)
    assert np.isnan(pb) and pu > 0
    assert "no reliable quotes near the bear price $70.00" in note
    _, pu2, note2 = ks.implied_odds(smile, S, T, R, 85.0, 400.0)  # beyond the top strike
    assert np.isnan(pu2) and "bull price $400.00" in note2


def test_wide_spreads_are_skipped():
    calls, puts = chain(spread=0.8)                              # 80% of mid: junk quotes
    assert ks.option_smile(calls, puts, S, T, R).empty


@pytest.mark.parametrize("horizon, want", [(3.0, 2.0), (1.0, 1.0), (0.5, 0.5)])
def test_expiry_is_the_longest_at_or_under_the_horizon(horizon, want):
    today = date(2026, 11, 10)
    exps = [(today + timedelta(days=round(y * 365.25))).isoformat()
            for y in (0.05, 0.5, 1.0, 2.0)]
    e, years = ks._pick_expiry(exps, horizon, today)
    assert years == pytest.approx(want, abs=0.01)


def test_market_check_lines_up_your_odds_with_the_markets(monkeypatch):
    calls, puts = chain()
    expiry = (date.today() + timedelta(days=round(T * 365.25))).isoformat()

    class FakeTicker:
        def __init__(self, t):
            self.options = (expiry,)

        def option_chain(self, e):
            return SimpleNamespace(calls=calls, puts=puts)
    monkeypatch.setattr("yfinance.Ticker", FakeTicker)
    views = ks.normalize_views(pd.DataFrame({"ticker": ["AAA"], "p_bear": [.10], "bear": [.30],
                                             "p_base": [.5], "base": [.1], "bull": [.5]}))
    check = ks.market_check(views, {"AAA": S}, horizon=1.5, rf=RF).set_index("ticker")
    row = check.loc["AAA"]
    assert row.your_bear == pytest.approx(.10) and row.your_bull == pytest.approx(.40)
    assert row.bear_price == pytest.approx(70.0) and row.bull_price == pytest.approx(150.0)
    assert row.market_bear == pytest.approx(closed_form(70.0, 0.40, True), abs=2e-3)
    gaps = ks.market_gaps(check.reset_index())
    assert any("less downside than the market" in g for g in gaps)     # 10% vs ~22%


def test_no_options_never_raises(monkeypatch):
    class Broken:
        def __init__(self, t):
            raise RuntimeError("yahoo said no")
    monkeypatch.setattr("yfinance.Ticker", Broken)
    sm = ks.fetch_option_smile("ZZZ", 100.0, 3.0, RF)
    assert sm["smile"].empty and "unavailable" in sm["note"]
