"""'What if I'm wrong?': Thorp's warning (p. 105) made visible."""
import numpy as np
import pandas as pd

import kelly as ks
from conftest import fake_download

EXAMPLE = ks.load_views(ks.Path(__file__).resolve().parents[1] / "views.example.csv")
BASKET = ["NVDA", "ASML", "KO", "VOO"]


def _prices():
    return fake_download(["^GSPC", *BASKET])["Close"]


def test_stress_moves_odds_from_bull_and_keeps_them_valid():
    v = ks.stress_views(EXAMPLE, 0.10, 1.0)
    assert np.allclose(v.p_bear - EXAMPLE.p_bear, 0.10)
    assert np.allclose(v.p_base, EXAMPLE.p_base)
    assert np.allclose(v.p_bear + v.p_base + v.p_bull, 1)
    tight = ks.normalize_views(pd.DataFrame({"ticker": ["A"], "p_bear": [.5], "bear": [.3],
                                             "p_base": [.45], "base": [.1], "bull": [.5]}))
    t = ks.stress_views(tight, 0.10, 1.0)            # only 5 pts of bull to take
    assert t.p_bull.iloc[0] >= 0.0099 and abs(t.p_bear.iloc[0] - .6) < 1e-9


def test_deeper_loss_is_capped_at_total_loss():
    v = ks.stress_views(EXAMPLE.assign(bear=.8), 0.0, 1.5)
    assert (v.bear == 1.0).all()


def test_nvda_example_is_fragile_and_being_wrong_costs_you():
    prices = _prices()
    _, w, _ = ks.size_basket(prices, BASKET, .045, views=EXAMPLE, prices_now={"ASML": 900.0})
    by_name, held = ks.what_if_wrong(prices, BASKET, .045, EXAMPLE, w, "half",
                                     prices_now={"ASML": 900.0})
    nvda = by_name.set_index("ticker").loc["NVDA"]
    assert nvda.verdict == "Fragile" and nvda.iloc[2] < nvda.mine * 0.4
    assert by_name.set_index("ticker").at["VOO", "verdict"] == ""     # CAPM: no view
    assert (held.prob_loss.iloc[1:] > held.prob_loss.iloc[0]).all()
    assert "Kelly would cut NVDA" in ks.what_if_message(by_name, held)


def test_a_low_risk_view_is_robust():
    prices = _prices()
    safe = ks.normalize_views(pd.DataFrame({"ticker": ["KO"], "p_bear": [.02], "bear": [.05],
                                            "bull": [.60]}))
    _, w, _ = ks.size_basket(prices, ["KO", "VOO"], .045, views=safe)
    by_name, _ = ks.what_if_wrong(prices, ["KO", "VOO"], .045, safe, w, "half")
    assert by_name.set_index("ticker").at["KO", "verdict"] == "Robust"


def test_no_views_means_nothing_to_be_wrong_about():
    prices = _prices()
    _, w, _ = ks.size_basket(prices, BASKET, .045)
    assert ks.what_if_wrong(prices, BASKET, .045, None, w, "half") is None
