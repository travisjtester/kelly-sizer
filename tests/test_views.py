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


def _bears(*bears):
    return pd.DataFrame({"ticker": [f"T{i}" for i in range(len(bears))],
                         "p_bear": [.2] * len(bears), "bear": list(bears),
                         "bull": [.5] * len(bears)})


def test_mixed_bear_signs_warn_and_name_the_odd_rows():
    with ks.captured_log() as notes:
        v = ks.normalize_views(_bears(-.3, -.4, .2))
    assert any("bear signs are mixed" in n and "T2" in n and "T0" not in n for n in notes)
    assert (v.bear > 0).all()                      # still read as a loss: a warning, not an error


@pytest.mark.parametrize("bears", [(-.3, -.4, -.2), (.3, .4, .2)])
def test_consistent_bear_signs_stay_quiet(bears):
    with ks.captured_log() as notes:
        ks.normalize_views(_bears(*bears))
    assert not any("bear signs" in n for n in notes)


def _anchored(**extra):
    return ks.normalize_views(pd.DataFrame({"ticker": ["A"], "p_bear": [.2], "bear": [.3],
                                            "p_base": [.5], "base": [.1], "bull": [.5],
                                            "as_of_price": ["$100"], **extra}))


def test_context_columns_parse_and_blank_out_junk():
    v = _anchored(horizon=[0])
    assert v.at["A", "as_of_price"] == 100 and np.isnan(v.at["A", "horizon"])
    assert {"as_of_price", "horizon"} <= set(ks.normalize_views(_bears(-.3)).columns)


def test_stale_view_reanchors_to_todays_price():
    # Set at $100: bear $70, base $110, bull $150. At $90 those prices are
    # a 22% loss, a 22% gain and a 67% gain.
    with ks.captured_log() as notes:
        r = ks.resolve_views(_anchored(), {"A": 90.0}).loc["A"]
    assert r.bear == pytest.approx(1 - 70 / 90)
    assert r.base == pytest.approx(110 / 90 - 1) and r.bull == pytest.approx(150 / 90 - 1)
    assert any("moved -10%" in n for n in notes)


def test_small_moves_reanchor_quietly_and_no_price_means_no_change():
    with ks.captured_log() as notes:
        r = ks.resolve_views(_anchored(), {"A": 98.0}).loc["A"]
    assert r.bear == pytest.approx(1 - 70 / 98) and not notes
    plain = ks.resolve_views(_bears(.3), {"T0": 50.0}).loc["T0"]
    assert plain.bear == pytest.approx(.3) and plain.bull == pytest.approx(.5)


@pytest.mark.parametrize("price, words", [(65.0, "at or below your bear price $70.00"),
                                          (160.0, "already past your bull case")])
def test_views_the_price_has_run_through_fall_back_to_capm(price, words):
    with ks.captured_log() as notes:
        v = ks.resolve_views(_anchored(), {"A": price})
    assert v.empty and any(words in n for n in notes)


def test_horizon_mismatch_is_flagged():
    with ks.captured_log() as notes:
        ks._warn_horizon_mismatch(_anchored(horizon=[1]), 3.0)
        ks._warn_horizon_mismatch(_anchored(horizon=[3]), 3.0)
        ks._warn_horizon_mismatch(_anchored(), 3.0)
    assert len(notes) == 1 and "A (1y)" in notes[0] and "This run is 3 years" in notes[0]


def test_neutralize_keeps_order_and_hits_the_target():
    v = _anchored().loc["A"]
    n = ks.neutralize_view(v, 0.25)
    probs, rets = ks.view_outcomes(n)
    assert float(probs @ rets) == pytest.approx(0.25)
    assert -n.bear < n.base < n.bull


def test_neutral_mode_sizes_views_at_their_capm_return(prices):
    views = ks.normalize_views(pd.DataFrame({"ticker": ["AAA"], "p_bear": [.2], "bear": [.2],
                                             "p_base": [.5], "base": [.8], "bull": [2.0]}))
    with ks.captured_log():
        edge = ks.compute_kelly(prices, .045, .1, views=views).set_index("ticker").loc["AAA"]
        flat = ks.compute_kelly(prices, .045, .1, views=views,
                                neutral=True).set_index("ticker").loc["AAA"]
    capm_h = (1 + flat.expected_return) ** 3 - 1
    assert flat.view_expected_return == pytest.approx(capm_h)
    assert edge.full_kelly > flat.full_kelly


@pytest.mark.parametrize("horizon, maturity", [(1, 1), (2, 2), (2.5, 3), (3, 3), (4, 5),
                                               (6, 7), (10, 10), (20, 10)])
def test_risk_free_maturity_matches_the_horizon(horizon, maturity):
    assert ks._nearest_maturity(horizon, ks.FRED_TREASURIES) == maturity
