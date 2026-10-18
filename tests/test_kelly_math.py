"""The book's Kelly, checked: Concentrated Investing ch. 3-4."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

import kelly as ks

RF = 0.045


def test_books_worked_example():
    # p. 76: even money, 51% to win, lose the stake otherwise -> bet 2%
    assert abs(ks.binary_kelly(0.51, 1.0, 1.0, 0.0) - 0.02) < 1e-12
    assert abs(ks.discrete_kelly(np.array([.49, .51]), np.array([-1., 1.]), 0.0) - 0.02) < 1e-6


def test_no_edge_no_bet():
    assert ks.discrete_kelly(np.array([.5, .5]), np.array([-.2, .2]), 0.05) == 0
    assert ks.binary_kelly(0.5, 0.2, 0.2, 0.1) == 0


def test_solver_matches_brute_force_and_closed_form():
    cases = [([.25, .35, .40], [-.5, .15, 1.0], .14), ([.1, .6, .3], [-.15, .12, .3], .14),
             ([.3, .7], [-.4, .8], .1)]
    for probs, rets, rf in cases:
        p, r = np.array(probs), np.array(rets)
        brute = minimize_scalar(lambda f: -(p @ np.log(1 + rf + f * (r - rf))),
                                bounds=(0, (1 + rf) / (rf - r.min()) * 0.9999),
                                method="bounded").x
        assert abs(ks.discrete_kelly(p, r, rf) - brute) < 1e-4
    assert abs(ks.discrete_kelly(np.array([.3, .7]), np.array([-.4, .8]), .1)
               - ks.binary_kelly(.7, .8, .4, .1)) < 1e-6


def test_never_risks_ruin():
    # a total-loss bear case: the bet must stay below 100%
    f = ks.discrete_kelly(np.array([.05, .95]), np.array([-1.0, 1.0]), 0.0)
    assert 0 < f < 1


def test_thorps_joint_sizing_rule(prices):
    # p. 105: two bets that each warrant >50% alone can't both get >50%
    views = pd.DataFrame({"ticker": ["AAA", "BBB"], "p_bear": [.2, .2],
                          "bear": [1., 1.], "bull": [1., 1.]})
    res = ks.compute_kelly(prices, RF, .1068, views=views, horizon=1)
    solo = res.set_index("ticker").loc[["AAA", "BBB"], "full_kelly"]
    w, _ = ks.portfolio_kelly(prices, res, ["AAA", "BBB"], RF, max_weight=1,
                              max_leverage=1, shrink=1, horizon=1)
    joint = w.set_index("ticker").loc[["AAA", "BBB"], "weight_full_kelly"]
    assert (solo > .5).all() and (joint < .5).all()


def test_weights_sum_to_one_with_cash_and_are_deterministic(prices):
    res = ks.compute_kelly(prices, RF, .1068)
    w1, _ = ks.portfolio_kelly(prices, res, ["AAA", "CCC", "DDD"], RF)
    w2, _ = ks.portfolio_kelly(prices, res, ["AAA", "CCC", "DDD"], RF)
    assert np.allclose(w1.filter(like="weight_").sum(), 1)
    assert w1.ticker.iloc[-1] == "CASH"
    assert np.allclose(w1.weight_full_kelly, w2.weight_full_kelly)


def test_caps_respected(prices):
    res = ks.compute_kelly(prices, RF, .1068)
    w, s = ks.portfolio_kelly(prices, res, ["AAA", "CCC", "DDD"], RF, max_weight=.25)
    names = w[w.ticker != "CASH"].filter(like="weight_")
    assert (names.to_numpy() <= .25 + 1e-6).all()
    assert (s.invested <= 1 + 1e-6).all()


def test_fractional_kelly_is_fraction_of_the_kelly_bet_then_capped(prices):
    # uncapped: half Kelly is exactly half of full Kelly
    res = ks.compute_kelly(prices, RF, .1068, horizon=1)
    w, _ = ks.portfolio_kelly(prices, res, ["CCC", "DDD"], RF, max_weight=5,
                              max_leverage=5, horizon=1)
    names = w[w.ticker != "CASH"]
    assert np.allclose(names.weight_half_kelly, names.weight_full_kelly / 2, atol=2e-3)


def test_copula_reproduces_your_scenario_odds(prices):
    views = pd.DataFrame({"ticker": ["AAA"], "p_bear": [.25], "bear": [.5],
                          "p_base": [.35], "base": [.15], "bull": [1.0]})
    ok = ks.compute_kelly(prices, RF, .1068, views=views, horizon=3).set_index("ticker")
    R = ks.simulate_outcomes(ok, ["AAA", "CCC"], np.eye(2), 3, 200_000)
    freq = [(R[:, 0] == -.5).mean(), (R[:, 0] == .15).mean(), (R[:, 0] == 1.0).mean()]
    assert np.allclose(freq, [.25, .35, .40], atol=.005)


def test_short_horizon_converges_to_continuous_kelly(prices):
    # Buy-and-hold over one month ~ continuous (Merton) Kelly w = Sigma^-1 (alpha - r)
    basket = ["CCC", "DDD"]
    res = ks.compute_kelly(prices, RF, .1068, horizon=1 / 12).set_index("ticker")
    w, _ = ks.portfolio_kelly(prices, res.reset_index(), basket, RF, max_weight=10,
                              max_leverage=10, shrink=0, horizon=1 / 12,
                              n_scenarios=200_000)
    vol = res.loc[basket, "annual_volatility"].to_numpy()
    corr = prices[basket].pct_change(fill_method=None).corr().to_numpy()
    cov = corr * np.outer(vol, vol)
    alpha = np.log1p(res.loc[basket, "expected_return"].to_numpy())
    merton = np.linalg.solve(cov, alpha - np.log1p(RF))
    got = w.set_index("ticker").loc[basket, "weight_full_kelly"].to_numpy()
    assert np.allclose(got, merton, rtol=0.08, atol=0.05), (got, merton)
