"""
Live checks against the real data sources (no fakes). Run with `pytest smoke`.

The offline suite fakes Yahoo so it's fast and repeatable, which also means it
can't notice a source changing shape or going away. These checks catch that:
each asserts the source answers, and with numbers in a sane range.
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import kelly as ks  # noqa: E402

BASKET = ["AAPL", "MSFT", "KO"]


@pytest.mark.parametrize("years", [1, 3, 10])
def test_fred_treasury_yields(years):
    assert 0 < ks._fred_yield(ks.FRED_TREASURIES[years]) < 0.20


@pytest.mark.parametrize("years", [1, 3, 10])
def test_treasury_gov_yields(years):
    assert 0 < ks._treasury_yield(years) < 0.20


@pytest.mark.parametrize("horizon, years", [(2, 2), (3, 3)])
def test_risk_free_is_horizon_matched(horizon, years):
    """Whichever source answers, the rate must match the horizon: the Yahoo
    fallback only has 5- and 10-year yields."""
    rf, source = ks.get_risk_free_rate(horizon)
    assert source in (f"FRED {ks.FRED_TREASURIES[years]}", f"Treasury.gov {years} Yr"), source
    assert 0 < rf < 0.20


def test_sp500_constituents():
    meta = ks.get_sp500_constituents()
    assert 490 <= len(meta) <= 520
    assert meta[["ticker", "name", "sector"]].notna().all().all()
    assert "Information Technology" in set(meta["sector"])


def test_yahoo_monthly_history():
    prices = ks.download_monthly_closes(BASKET + [ks.BENCHMARK], ks.VARIANCE_YEARS)
    assert set(BASKET + [ks.BENCHMARK]) <= set(prices.columns)
    assert len(prices.dropna()) >= ks.MIN_MONTHS


def test_yahoo_latest_prices():
    px = ks.download_latest_prices(BASKET)
    assert set(px) == set(BASKET) and all(p > 0 for p in px.values())


def test_yahoo_name_and_sector():
    meta = ks.lookup_extra_meta(["ASML"]).iloc[0]
    assert meta["name"] and meta["sector"] == "Information Technology"


def test_end_to_end_sizing():
    rf, _ = ks.get_risk_free_rate(3)
    prices = ks.download_monthly_closes(BASKET + [ks.BENCHMARK], ks.VARIANCE_YEARS)
    views = ks.normalize_views(pd.DataFrame({"ticker": ["KO"], "p_bear": [.1], "bear": [.15],
                                             "bull": [.3]}))
    _, weights, summary = ks.size_basket(prices, BASKET, rf, views=views,
                                         prices_now=ks.download_latest_prices(["KO"]))
    assert summary["solver_ok"].all()
    assert abs(weights.filter(like="weight_half").sum().iloc[0] - 1) < 1e-9
