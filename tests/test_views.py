import numpy as np
import pandas as pd
import pytest

import kelly as ks


def _v(**cols):
    return ks.normalize_views(pd.DataFrame({"ticker": ["A"], **cols}))


@pytest.mark.parametrize("cols", [
    dict(p_bear=[.4], bear=[.4], p_base=[.2], base=[.1], bull=[1.2]),
    dict(p_bear=[40], bear=[40], p_base=[20], base=[10], bull=[120]),
    dict(p_bear=["40%"], bear=["40%"], p_base=["20%"], base=["10%"], bull=["120%"]),
])
def test_decimals_percents_and_percent_signs_agree(cols):
    v = _v(**cols)
    assert np.allclose(v.loc["A", ["p_bear", "bear", "p_base", "base", "bull", "p_bull"]]
                       .astype(float), [.4, .4, .2, .1, 1.2, .4])


def test_returns_over_100_percent_are_not_mistaken_for_percents():
    assert _v(p_bear=[.3], bear=[.9], bull=[4.0]).loc["A", "bull"] == 4.0
    assert _v(p_bear=[30], bear=[90], bull=[400]).loc["A", "bull"] == 4.0


def test_small_returns_follow_the_files_units():
    assert abs(_v(p_bear=[40], bear=[40], p_base=[20], base=[5], bull=[80])
               .loc["A", "base"] - .05) < 1e-12
    assert abs(_v(p_bear=[.4], bear=[.4], p_base=[.2], base=[.05], bull=[.8])
               .loc["A", "base"] - .05) < 1e-12


def test_legacy_binary_format_loads_as_bear_bull():
    old = ks.normalize_views(pd.DataFrame({"ticker": ["A"], "p_win": [60],
                                           "upside": [80], "downside": [40]}))
    assert np.allclose(old.loc["A", ["p_bear", "bear", "p_bull", "bull"]].astype(float),
                       [.4, .4, .6, .8])


def test_intrinsic_value_sets_bull_and_margin_of_safety():
    v = ks.normalize_views(pd.DataFrame({"ticker": ["A", "B"], "p_bear": [.2, .2],
                                         "bear": [.3, .3], "value": [150, 80]}))
    rv = ks.resolve_views(v, {"A": 100.0, "B": 100.0})
    assert list(rv.index) == ["A"]                       # value below price: no upside
    assert abs(rv.at["A", "bull"] - .5) < 1e-12
    assert abs(rv.at["A", "margin_of_safety"] - 1 / 3) < 1e-12


@pytest.mark.parametrize("cols", [
    dict(p_bear=[.6], bear=[.3], p_base=[.5], base=[.1], bull=[.5]),   # odds > 100%
    dict(p_bear=[.2], bear=[150], bull=[.5]),                          # loss > 100%
    dict(p_bear=[.2], bear=[.3], p_base=[.3], bull=[.5]),              # base odds, no base
])
def test_invalid_views_rejected(cols):
    with pytest.raises(ValueError):
        _v(**cols)


def test_short_history_name_kept_only_with_a_view(prices):
    views = pd.DataFrame({"ticker": ["NEWCO"], "p_bear": [.3], "bear": [.5], "bull": [1.0]})
    res = ks.compute_kelly(prices, .045, .1068, views=views)
    w, _ = ks.portfolio_kelly(prices, res, ["NEWCO", "CCC", "DDD"], .045)
    assert "NEWCO" in set(w.ticker)
    res2 = ks.compute_kelly(prices, .045, .1068)
    w2, _ = ks.portfolio_kelly(prices, res2, ["NEWCO", "CCC", "DDD"], .045)
    assert "NEWCO" not in set(w2.ticker)


def test_no_unfaithful_columns(prices):
    res = ks.compute_kelly(prices, .045, .1068)
    _, s = ks.portfolio_kelly(prices, res, ["CCC", "DDD"], .045)
    for gone in ["cagr_gap", "performance_category", "sharpe_capm"]:
        assert gone not in res.columns
    assert "sharpe" not in s.columns
