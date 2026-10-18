"""
Kelly Criterion screener for S&P 500 constituents.

Pipeline
  1. Constituents  : Wikipedia (browser UA) -> GitHub datasets CSV fallback
  2. Risk-free rate: FRED DGS10 (no API key) -> Yahoo ^TNX fallback -> constant
  3. Prices        : ONE bulk yfinance download of monthly closes (auto-adjusted)
                     for every ticker + ^GSPC, in chunks, with retries
  4. Per ticker    : beta (regressed vs ^GSPC, 5y monthly), CAPM expected
                     return, annualized variance (10y monthly), Kelly fractions
  5. Output        : CSV (Desktop by default) sorted by full Kelly
  6. Optional      : multi-asset Kelly weights for a basket (--portfolio /
                     --portfolio-top), accounting for correlations
  7. Optional      : views mode (--views views.csv). Your own edge per name:
                     probability of winning, upside, permanent downside.
                     Sized by maximizing expected log wealth over correlated
                     scenarios, following Concentrated Investing (Benello,
                     van Biema, Carlisle) ch. 3-4.

Kelly (continuous / Merton form):  f* = (E[R] - Rf) / sigma^2
With CAPM E[R] = Rf + beta * MRP, this reduces to  f* = beta * MRP / sigma^2.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------- config ---
# Long-run nominal S&P 500 annual return. Update yearly:
# https://www.officialdata.org/us/stocks/s-p-500/1924?amount=100&endYear=2024
MARKET_RETURN = 0.1068
FALLBACK_RF = 0.045

BENCHMARK = "^GSPC"
VARIANCE_YEARS = 10      # lookback for variance
BETA_YEARS = 5           # lookback for beta (matches Yahoo's 5y-monthly convention)
MIN_MONTHS = 60          # require 5y of history, same rule as the original
CHUNK = 100              # tickers per bulk download request

# hist_cagr minus volatility-adjusted CAPM CAGR, in annual decimal terms.
# (upper_bound_exclusive, label), checked in order.
PERFORMANCE_BUCKETS = [
    (-0.06, "Strong underperformer"),
    (-0.02, "Underperformer"),
    (0.02, "In line"),
    (0.06, "Outperformer"),
    (np.inf, "Strong outperformer"),
]

# Portfolio (multi-asset) Kelly defaults
PORTFOLIO_MAX_WEIGHT = 0.25   # per-name cap on the full-Kelly book
PORTFOLIO_MAX_LEVERAGE = 1.0  # sum of weights; 1.0 = no margin
COV_SHRINK = 0.20             # blend sample covariance toward its diagonal

# Views mode (Concentrated Investing: edge from your own research)
DEFAULT_HORIZON = 3.0         # years for a thesis to play out
N_SCENARIOS = 20_000          # joint outcomes simulated for expected-log Kelly
VIEW_MIN_MONTHS = 12          # min shared months to estimate a correlation
SCENARIO_SEED = 7             # fixed so reruns give identical weights

UA = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/130.0 Safari/537.36"
}
WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
GITHUB_URL = ("https://raw.githubusercontent.com/datasets/"
              "s-and-p-500-companies/main/data/constituents.csv")
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10"


def log(msg: str) -> None:
    print(msg, flush=True)


# ----------------------------------------------------------- data sources ---
def _yahoo_symbol(s: str) -> str:
    return s.strip().replace(".", "-")  # BRK.B -> BRK-B


def get_sp500_constituents() -> pd.DataFrame:
    """Return DataFrame[ticker, name, sector]."""
    try:
        html = requests.get(WIKI_URL, headers=UA, timeout=20).text
        df = pd.read_html(io.StringIO(html))[0]
        out = pd.DataFrame({
            "ticker": df["Symbol"].map(_yahoo_symbol),
            "name": df["Security"],
            "sector": df["GICS Sector"],
        })
        log(f"Constituents: {len(out)} from Wikipedia")
        return out
    except Exception as e:
        log(f"Wikipedia failed ({e}); trying GitHub dataset...")

    df = pd.read_csv(io.StringIO(requests.get(GITHUB_URL, timeout=20).text))
    sector_col = "GICS Sector" if "GICS Sector" in df.columns else "Sector"
    name_col = "Security" if "Security" in df.columns else "Name"
    out = pd.DataFrame({
        "ticker": df["Symbol"].map(_yahoo_symbol),
        "name": df[name_col],
        "sector": df[sector_col],
    })
    log(f"Constituents: {len(out)} from GitHub dataset")
    return out


def get_risk_free_rate() -> tuple[float, str]:
    """10-Year Treasury yield as a decimal, plus the source used."""
    try:
        r = requests.get(FRED_URL, headers=UA, timeout=8)
        r.raise_for_status()
        s = pd.read_csv(io.StringIO(r.text)).iloc[:, 1]
        s = pd.to_numeric(s, errors="coerce").dropna()
        return float(s.iloc[-1]) / 100, "FRED DGS10"
    except Exception as e:
        log(f"FRED failed ({e}); trying Yahoo ^TNX...")
    try:
        import yfinance as yf
        h = yf.Ticker("^TNX").history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]) / 100, "Yahoo ^TNX"
    except Exception as e:
        log(f"^TNX failed ({e}); using fallback {FALLBACK_RF:.2%}")
    return FALLBACK_RF, "fallback constant"


def download_monthly_closes(tickers: list[str], years: int,
                            retries: int = 3) -> pd.DataFrame:
    """
    Month-end adjusted closes, one column per ticker.
    Bulk-downloads in chunks: ~6 requests for the whole index instead of
    ~1,500 one-at-a-time calls (which is what got the old script throttled).
    """
    import yfinance as yf

    frames = []
    for i in range(0, len(tickers), CHUNK):
        chunk = tickers[i:i + CHUNK]
        log(f"Downloading prices {i + 1}-{i + len(chunk)} of {len(tickers)}...")
        for attempt in range(retries):
            try:
                raw = yf.download(
                    chunk, period=f"{years}y", interval="1mo",
                    auto_adjust=True, group_by="column",
                    threads=True, progress=False,
                )
                close = raw["Close"]
                if isinstance(close, pd.Series):        # single-ticker edge case
                    close = close.to_frame(chunk[0])
                frames.append(close)
                break
            except Exception as e:
                wait = 5 * 2 ** attempt
                log(f"  attempt {attempt + 1} failed ({e}); retrying in {wait}s")
                time.sleep(wait)
        time.sleep(2)

    if not frames:
        return pd.DataFrame()
    prices = pd.concat(frames, axis=1)
    prices = prices.loc[:, ~prices.columns.duplicated()]
    # Interval=1mo can include a partial current-month bar; drop it.
    today = pd.Timestamp(date.today())
    prices = prices[prices.index < today.to_period("M").to_timestamp()]
    return prices.sort_index()


# ----------------------------------------------------------------- math ---
def categorize_gap(gap: float) -> str:
    for bound, label in PERFORMANCE_BUCKETS:
        if gap < bound:
            return label
    return PERFORMANCE_BUCKETS[-1][1]


# ---------------------------------------------------------------- views ---
VIEW_COLS = ["p_win", "upside", "downside"]


# A column is read as percents if any plain number in it exceeds this
# (p_win, downside: 1; upside can legitimately be >100%, e.g. 1.5 = +150%).
_PCT_THRESHOLD = {"p_win": 1.0, "upside": 10.0, "downside": 1.0}


def _column_to_fraction(col: pd.Series, name: str) -> pd.Series:
    """
    Parse one views column. A '%' sign always means percent. Otherwise the
    column is read consistently: all percents if any value exceeds the
    threshold for that column (60 -> 0.60), else all decimals (0.6 -> 0.60).
    """
    txt = col.astype(str).str.strip()
    has_pct = txt.str.endswith("%")
    num = pd.to_numeric(txt.str.rstrip("%"), errors="coerce")
    plain = num[~has_pct].dropna()
    as_pct = bool((plain.abs() > _PCT_THRESHOLD[name]).any())
    out = num.where(~has_pct, num / 100)
    if as_pct:
        out = out.where(has_pct, num / 100)
    return out


def normalize_views(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Clean a views table to index=ticker, columns p_win/upside/downside as
    fractions. Rows missing any of the three are ignored (that name uses CAPM).

      p_win    : probability the thesis plays out over the horizon (0-1)
      upside   : gain if it does, e.g. 0.60 for +60%
      downside : PERMANENT loss if it doesn't, e.g. 0.30 for -30% (0-1]
    """
    empty = pd.DataFrame(columns=VIEW_COLS + ["notes"]).rename_axis("ticker")
    if df is None or df.empty:
        return empty
    d = df.copy()
    if d.index.name == "ticker" and "ticker" not in d.columns:
        d = d.reset_index()                      # already normalized: idempotent
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "ticker" not in d.columns or not set(VIEW_COLS) <= set(d.columns):
        raise ValueError("Views need columns: ticker, p_win, upside, downside")
    d["ticker"] = d["ticker"].astype(str).map(_yahoo_symbol).str.upper()
    for c in VIEW_COLS:
        d[c] = _column_to_fraction(d[c], c)
    d["downside"] = d["downside"].abs()
    d = d.dropna(subset=VIEW_COLS)
    bad = d[(d.p_win <= 0) | (d.p_win >= 1) | (d.upside <= 0)
            | (d.downside <= 0) | (d.downside > 1)]
    if not bad.empty:
        raise ValueError("Invalid views (need 0<p_win<1, upside>0, 0<downside<=1): "
                         + ", ".join(bad["ticker"]))
    tiny = d[d.upside < 0.02]
    if not tiny.empty:
        log("Views: upside under 2% for " + ", ".join(tiny["ticker"])
            + " - check units (use 1.5 or 150 for +150%)")
    if "notes" not in d.columns:
        d["notes"] = ""
    return d.drop_duplicates("ticker", keep="last").set_index("ticker")[VIEW_COLS + ["notes"]]


def load_views(path: Path | str) -> pd.DataFrame:
    return normalize_views(pd.read_csv(path))


def binary_kelly(p: float, upside: float, downside: float, rf_h: float) -> float:
    """
    Exact Kelly fraction for one bet that returns +upside with prob p or
    -downside with prob 1-p, versus cash earning rf_h over the same horizon.

        f* = (1 + rf) * ( p / (downside + rf) - q / (upside - rf) )

    With rf = 0 this is p/a - q/b; with a total-loss downside it is the
    book's (bp - q) / b. Negative means "no edge, don't bet" (floored at 0).
    """
    q = 1 - p
    if upside <= rf_h:
        return 0.0
    f = (1 + rf_h) * (p / (downside + rf_h) - q / (upside - rf_h))
    return max(f, 0.0)


def view_stats(v: pd.Series, rf: float, horizon: float) -> dict:
    rf_h = (1 + rf) ** horizon - 1
    exp_h = v.p_win * v.upside - (1 - v.p_win) * v.downside
    f = binary_kelly(v.p_win, v.upside, v.downside, rf_h)
    return {
        "edge_source": "views",
        "p_win": v.p_win, "upside": v.upside, "downside": v.downside,
        "horizon_years": horizon,
        "view_expected_return": exp_h,
        "view_annualized_return": (1 + exp_h) ** (1 / horizon) - 1 if exp_h > -1 else -1.0,
        "full_kelly": f, "half_kelly": f / 2,
        "quarter_kelly": f / 4, "eighth_kelly": f / 8,
    }


def compute_kelly(prices: pd.DataFrame, rf: float, market_return: float,
                  benchmark: str = BENCHMARK,
                  views: pd.DataFrame | None = None,
                  horizon: float = DEFAULT_HORIZON) -> pd.DataFrame:
    """
    Per-name stats. CAPM supplies the edge by default; a name with a view
    gets its Kelly columns from your view instead (edge_source = 'views'),
    keeping the CAPM figure in capm_full_kelly for comparison. Names with a
    view are kept even with short price history.
    """
    views = normalize_views(views) if views is not None else normalize_views(None)
    mrp = market_return - rf
    rets = prices.pct_change(fill_method=None)
    beta_window = rets.iloc[-BETA_YEARS * 12:]
    mkt_b = beta_window[benchmark]

    rows = []
    for t in prices.columns:
        if t == benchmark:
            continue
        r = rets[t].dropna()
        has_view = t in views.index
        if len(r) < MIN_MONTHS:
            if has_view:
                rows.append({"ticker": t, "months_of_data": len(r),
                             **view_stats(views.loc[t], rf, horizon),
                             "status": "ok"})
            else:
                rows.append({"ticker": t, "edge_source": "CAPM",
                             "status": f"insufficient history ({len(r)} mo)"})
            continue

        pair = pd.concat([beta_window[t], mkt_b], axis=1).dropna()
        beta = pair.iloc[:, 0].cov(pair.iloc[:, 1]) / pair.iloc[:, 1].var()
        var_annual = r.var() * 12
        vol_annual = np.sqrt(var_annual)
        exp_ret = rf + beta * mrp
        kelly = (exp_ret - rf) / var_annual if var_annual > 0 else np.nan
        hist_cagr = (1 + r).prod() ** (12 / len(r)) - 1
        # CAPM gives an arithmetic mean; compounding loses ~variance/2 a year.
        expected_cagr = exp_ret - var_annual / 2
        cagr_gap = hist_cagr - expected_cagr

        row = {
            "ticker": t,
            "edge_source": "CAPM",
            "beta": beta,
            "expected_return": exp_ret,
            "annual_variance": var_annual,
            "annual_volatility": vol_annual,
            "sharpe_capm": (exp_ret - rf) / vol_annual,
            "hist_cagr": hist_cagr,
            "expected_cagr": expected_cagr,
            "cagr_gap": cagr_gap,
            "performance_category": categorize_gap(cagr_gap),
            "months_of_data": len(r),
            "capm_full_kelly": kelly,
            "full_kelly": kelly,
            "half_kelly": kelly / 2,
            "quarter_kelly": kelly / 4,
            "eighth_kelly": kelly / 8,
            "status": "ok",
        }
        if has_view:
            row.update(view_stats(views.loc[t], rf, horizon))
        rows.append(row)
    return pd.DataFrame(rows)


FRACTIONS = {"full": 1.0, "half": 0.5, "quarter": 0.25, "eighth": 0.125}


def _basket_filter(results: pd.DataFrame, basket: list[str]) -> tuple[pd.DataFrame, list[str]]:
    ok = results.set_index("ticker")
    dropped = [t for t in basket
               if t not in ok.index or ok.at[t, "status"] != "ok"]
    for t in dropped:
        reason = ok.at[t, "status"] if t in ok.index else "not found"
        log(f"Portfolio: dropping {t} ({reason})")
    basket = [t for t in basket if t not in dropped]
    if len(basket) < 2:
        raise ValueError("Portfolio needs at least 2 tickers with valid data.")
    return ok, basket


def _solve(neg_obj, neg_grad, n: int, max_weight: float, max_leverage: float):
    from scipy.optimize import minimize
    res = minimize(
        neg_obj, x0=np.full(n, min(max_leverage, 1.0) / n / 2), jac=neg_grad,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "ineq",
                      "fun": lambda w: max_leverage - w.sum(),
                      "jac": lambda w: -np.ones(n)}],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    w[w < 1e-5] = 0.0
    return w, bool(res.success)


def _assemble(ok: pd.DataFrame, basket: list[str], weights: dict,
              summary: list[dict], extra: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = pd.DataFrame({"ticker": basket, **weights})
    meta_cols = ["name", "sector", "universe", "edge_source", "p_win", "upside",
                 "downside", "beta", "annual_volatility", "expected_return",
                 "full_kelly"]
    meta_cols = [c for c in meta_cols if c in ok.columns]
    out = out.merge(ok.loc[basket, meta_cols].reset_index(), on="ticker")
    out = out.rename(columns={"full_kelly": "standalone_full_kelly"})
    out = out.sort_values("weight_full_kelly", ascending=False)
    cash = {"ticker": "CASH", **{k: 1 - v.sum() for k, v in weights.items()}}
    out = pd.concat([out, pd.DataFrame([cash])], ignore_index=True)
    summ = pd.DataFrame(summary)
    for i, (k, v) in enumerate(extra.items(), start=1):
        summ.insert(i, k, v)
    return out, summ


def _portfolio_kelly_capm(prices, ok, basket, rf, max_weight, max_leverage, shrink):
    """
    Continuous (Merton) Kelly on CAPM inputs, rebalanced:
        maximize  w'(mu - rf) - (1 / 2c) * w' Sigma w
    solved separately for each Kelly fraction c (uncapped, this equals
    scaling full Kelly by c). Sigma: annualized covariance over the months
    all names share, shrunk toward its diagonal.
    """
    from scipy.stats import norm

    rets = prices[basket].pct_change(fill_method=None).dropna()
    shortest = prices[basket].notna().sum().idxmin()
    log(f"Portfolio: {len(rets)} overlapping months (limited by {shortest})")
    if len(rets) < MIN_MONTHS:
        raise ValueError(f"Only {len(rets)} months of overlapping history "
                         f"across the basket (need {MIN_MONTHS}).")
    sample = rets.cov().values * 12
    cov = (1 - shrink) * sample + shrink * np.diag(np.diag(sample))
    excess = ok.loc[basket, "expected_return"].values - rf
    n = len(basket)

    weights, summary = {}, []
    for label, c in FRACTIONS.items():
        w, success = _solve(
            lambda w, c=c: -(w @ excess - 0.5 / c * w @ cov @ w),
            lambda w, c=c: -(excess - (1 / c) * cov @ w),
            n, max_weight, max_leverage)
        weights[f"weight_{label}_kelly"] = w
        port_excess, port_var = w @ excess, w @ cov @ w
        vol = np.sqrt(port_var)
        g = rf + port_excess - 0.5 * port_var          # log growth / yr
        summary.append({
            "kelly_fraction": label,
            "invested": w.sum(), "cash": 1 - w.sum(),
            "holdings": int((w > 0).sum()),
            "expected_return": rf + port_excess,
            "volatility": vol,
            "sharpe": port_excess / vol if vol > 0 else np.nan,
            "expected_growth_rate": g,
            # 1-year horizon, lognormal approximation
            "prob_loss": float(norm.cdf(-g / vol)) if vol > 0 else 0.0,
            "worst_5pct": float(np.expm1(g + norm.ppf(0.05) * vol)),
            "solver_ok": success,
        })
    return _assemble(ok, basket, weights, summary,
                     {"engine": "CAPM", "basket_size": n,
                      "months_used": len(rets), "horizon_years": 1.0})


def _correlation(prices: pd.DataFrame, basket: list[str], shrink: float) -> tuple[np.ndarray, int]:
    """Pairwise correlations of monthly returns (each pair uses the months it
    shares, min VIEW_MIN_MONTHS; otherwise 0), shrunk toward independence and
    repaired to be positive semi-definite."""
    rets = prices[basket].pct_change(fill_method=None)
    corr = rets.corr(min_periods=VIEW_MIN_MONTHS).fillna(0.0).to_numpy(copy=True)
    np.fill_diagonal(corr, 1.0)
    corr = (1 - shrink) * corr + shrink * np.eye(len(basket))
    vals, vecs = np.linalg.eigh(corr)
    corr = vecs @ np.diag(np.clip(vals, 1e-8, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    missing = [t for t in basket
               if rets[t].notna().sum() < VIEW_MIN_MONTHS]
    for t in missing:
        log(f"Portfolio: {t} has <{VIEW_MIN_MONTHS} months of prices; "
            f"treated as uncorrelated with the rest")
    return corr, int(rets.dropna(how="all").shape[0])


def simulate_outcomes(ok: pd.DataFrame, basket: list[str], corr: np.ndarray,
                      horizon: float, n_scenarios: int = N_SCENARIOS,
                      seed: int = SCENARIO_SEED) -> np.ndarray:
    """
    Joint horizon returns, shape (n_scenarios, n_names), via a Gaussian copula
    on historical correlations:
      - name with a view : +upside with prob p_win, else -downside (permanent)
      - name without one : lognormal with CAPM mean and historical volatility
    """
    from scipy.stats import norm

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_scenarios, len(basket))) @ L.T
    out = np.empty_like(z)
    for j, t in enumerate(basket):
        row = ok.loc[t]
        if row.get("edge_source") == "views":
            win = norm.cdf(z[:, j]) > 1 - row["p_win"]
            out[:, j] = np.where(win, row["upside"], -row["downside"])
        else:
            vol = row["annual_volatility"]
            mu_log = np.log1p(row["expected_return"]) - vol ** 2 / 2
            out[:, j] = np.expm1(mu_log * horizon + vol * np.sqrt(horizon) * z[:, j])
    return out


def _portfolio_kelly_views(prices, ok, basket, rf, max_weight, max_leverage,
                           shrink, horizon, n_scenarios):
    """
    Kelly as the book defines it: maximize the expected LOG of wealth at the
    thesis horizon over the joint outcomes of all positions at once (Thorp's
    "joint properties" rule), long-only, no more than max_leverage invested.
    A position whose loss could wipe out the portfolio gets log(0) = -inf, so
    the solution never risks ruin. Fractional Kelly = the full-Kelly bets
    scaled by c (half Kelly = bet half).
    """
    corr, months = _correlation(prices, basket, shrink)
    R = simulate_outcomes(ok, basket, corr, horizon, n_scenarios)
    rf_h = (1 + rf) ** horizon - 1
    X = R - rf_h
    n = len(basket)

    def neg_obj(w):
        W = np.maximum(1 + rf_h + X @ w, 1e-12)
        return -np.mean(np.log(W))

    def neg_grad(w):
        W = np.maximum(1 + rf_h + X @ w, 1e-12)
        return -(X / W[:, None]).mean(axis=0)

    w_full, success = _solve(neg_obj, neg_grad, n, max_weight, max_leverage)

    weights, summary = {}, []
    for label, c in FRACTIONS.items():
        w = w_full * c
        weights[f"weight_{label}_kelly"] = w
        W = 1 + rf_h + X @ w
        logW = np.log(np.maximum(W, 1e-12))
        exp_ann = np.mean(W) ** (1 / horizon) - 1
        vol = np.std(logW) / np.sqrt(horizon)
        summary.append({
            "kelly_fraction": label,
            "invested": w.sum(), "cash": 1 - w.sum(),
            "holdings": int((w > 0).sum()),
            "expected_return": exp_ann,
            "volatility": vol,
            "sharpe": (exp_ann - rf) / vol if vol > 0 else np.nan,
            "expected_growth_rate": np.expm1(np.mean(logW) / horizon),
            "prob_loss": float(np.mean(W < 1)),
            "worst_5pct": float(np.quantile(W, 0.05) - 1),
            "solver_ok": success,
        })
    n_views = int((ok.loc[basket, "edge_source"] == "views").sum())
    log(f"Portfolio: views mode, {n_views} of {n} names with views, "
        f"{horizon:g}-year horizon, {n_scenarios:,} scenarios")
    return _assemble(ok, basket, weights, summary,
                     {"engine": "views", "basket_size": n,
                      "months_used": months, "horizon_years": horizon})


def portfolio_kelly(prices: pd.DataFrame, results: pd.DataFrame,
                    basket: list[str], rf: float,
                    max_weight: float = PORTFOLIO_MAX_WEIGHT,
                    max_leverage: float = PORTFOLIO_MAX_LEVERAGE,
                    shrink: float = COV_SHRINK,
                    horizon: float = DEFAULT_HORIZON,
                    n_scenarios: int = N_SCENARIOS,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Joint Kelly weights for a basket. If any name carries a view, uses the
    views engine (expected log wealth over correlated scenarios at the thesis
    horizon); otherwise the continuous CAPM engine. Returns (weights, summary).
    """
    ok, basket = _basket_filter(results, basket)
    has_views = ("edge_source" in ok.columns
                 and (ok.loc[basket, "edge_source"] == "views").any())
    if has_views:
        return _portfolio_kelly_views(prices, ok, basket, rf, max_weight,
                                      max_leverage, shrink, horizon, n_scenarios)
    return _portfolio_kelly_capm(prices, ok, basket, rf, max_weight,
                                 max_leverage, shrink)


# ------------------------------------------------------------------ main ---
def default_output() -> Path:
    desktop = Path.home() / "Desktop"
    folder = desktop if desktop.exists() else Path.cwd()
    return folder / f"kelly_sp500_{date.today():%Y-%m-%d}.csv"


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write CSV; if the file is locked (open in Excel), use a timestamped name."""
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%H%M%S}{path.suffix}")
        log(f"{path.name} is open/locked (Excel?). Writing {alt.name} instead.")
        df.to_csv(alt, index=False)
        return alt


def lookup_extra_meta(tickers: list[str]) -> pd.DataFrame:
    """Name/sector for non-S&P tickers (a few .info calls, best effort)."""
    import yfinance as yf

    rows = []
    for t in tickers:
        name = sector = None
        try:
            info = yf.Ticker(t).info
            name = info.get("shortName") or info.get("longName")
            sector = info.get("sector")
        except Exception:
            pass
        rows.append({"ticker": t, "name": name, "sector": sector})
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Kelly Criterion screener for the S&P 500")
    p.add_argument("--tickers", nargs="+",
                   help="Run ONLY these tickers instead of the full index")
    p.add_argument("--extra", nargs="+", default=[],
                   help="Add non-S&P tickers on top of the index (e.g. TSM ASML)")
    p.add_argument("--limit", type=int,
                   help="Only process the first N constituents (quick test)")
    p.add_argument("--market-return", type=float, default=MARKET_RETURN,
                   help=f"Long-run market return (default {MARKET_RETURN})")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path")

    g = p.add_argument_group("portfolio Kelly")
    g.add_argument("--portfolio", nargs="+",
                   help="Optimize joint Kelly weights for these tickers")
    g.add_argument("--portfolio-top", type=int,
                   help="Optimize joint Kelly weights for the top N by standalone Kelly")
    g.add_argument("--max-weight", type=float, default=PORTFOLIO_MAX_WEIGHT,
                   help=f"Per-name cap (default {PORTFOLIO_MAX_WEIGHT})")
    g.add_argument("--max-leverage", type=float, default=PORTFOLIO_MAX_LEVERAGE,
                   help=f"Max total invested; >1 allows margin (default {PORTFOLIO_MAX_LEVERAGE})")
    g.add_argument("--shrink", type=float, default=COV_SHRINK,
                   help=f"Covariance shrinkage toward diagonal, 0-1 (default {COV_SHRINK})")

    v = p.add_argument_group("views mode (your own edge)")
    v.add_argument("--views", type=Path,
                   help="CSV with ticker,p_win,upside,downside[,notes]; see views.example.csv")
    v.add_argument("--horizon", type=float, default=DEFAULT_HORIZON,
                   help=f"Years for each thesis to play out (default {DEFAULT_HORIZON:g})")
    v.add_argument("--scenarios", type=int, default=N_SCENARIOS,
                   help=f"Simulated joint outcomes (default {N_SCENARIOS:,})")
    args = p.parse_args(argv)

    views = load_views(args.views) if args.views else normalize_views(None)
    if len(views):
        log(f"Views loaded for {len(views)} names: {', '.join(views.index)}")

    rf, rf_source = get_risk_free_rate()
    log(f"Risk-free rate: {rf:.3%} ({rf_source})  |  "
        f"Market premium: {args.market_return - rf:.3%}")

    meta = get_sp500_constituents()
    meta["universe"] = "S&P 500"
    if args.tickers:
        tickers = [_yahoo_symbol(t) for t in args.tickers]
    else:
        tickers = meta["ticker"].tolist()
        if args.limit:
            tickers = tickers[:args.limit]

    # Extras + any portfolio names not already in the run
    wanted = [_yahoo_symbol(t) for t in args.extra + (args.portfolio or [])
              + list(views.index)]
    add = [t for t in dict.fromkeys(wanted) if t not in tickers]
    tickers += add
    non_sp = [t for t in tickers if t not in set(meta["ticker"])]
    if non_sp:
        log(f"Non-S&P tickers: {', '.join(non_sp)}")
        extra_meta = lookup_extra_meta(non_sp)
        extra_meta["universe"] = "Extra"
        meta = pd.concat([meta, extra_meta], ignore_index=True)

    prices = download_monthly_closes(tickers + [BENCHMARK], VARIANCE_YEARS)
    if prices.empty or BENCHMARK not in prices.columns:
        log("ERROR: no price data returned. Check network / try again later.")
        return 1

    missing = sorted(set(tickers) - set(prices.columns))
    res = compute_kelly(prices, rf, args.market_return,
                        views=views, horizon=args.horizon)
    if missing:
        res = pd.concat([res, pd.DataFrame({"ticker": missing,
                                            "status": "no price data"})])

    res = meta.merge(res, on="ticker", how="right")
    res.insert(0, "run_date", date.today().isoformat())
    res["risk_free_rate"] = rf
    res["market_return"] = args.market_return
    res = res.sort_values("full_kelly", ascending=False, na_position="last")

    ok = res[res["status"] == "ok"]
    log(f"\nComputed {len(ok)} of {len(tickers)} tickers.")
    show = ["ticker", "edge_source", "sector", "beta", "annual_volatility",
            "cagr_gap", "performance_category", "full_kelly", "half_kelly"]
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 160):
        log("\nTop 20 by full Kelly:\n" + ok[show].head(20).to_string(index=False))

    out = args.out or default_output()
    out.parent.mkdir(parents=True, exist_ok=True)
    out = safe_to_csv(res, out)
    log(f"\nSaved: {out}")

    # ---- optional portfolio Kelly
    basket = None
    if args.portfolio:
        basket = [_yahoo_symbol(t) for t in args.portfolio]
    elif args.portfolio_top:
        basket = ok["ticker"].head(args.portfolio_top).tolist()
    if basket:
        try:
            weights, summary = portfolio_kelly(
                prices, res, basket, rf,
                max_weight=args.max_weight,
                max_leverage=args.max_leverage,
                shrink=args.shrink,
                horizon=args.horizon,
                n_scenarios=args.scenarios,
            )
        except ValueError as e:
            log(f"\nPortfolio Kelly skipped: {e}")
            return 0
        stamp = f"{date.today():%Y-%m-%d}"
        w_path = out.parent / f"kelly_portfolio_{stamp}.csv"
        s_path = out.parent / f"kelly_portfolio_summary_{stamp}.csv"
        w_path = safe_to_csv(weights, w_path)
        s_path = safe_to_csv(summary, s_path)
        wcols = ["ticker", "edge_source", "weight_full_kelly", "weight_half_kelly",
                 "weight_quarter_kelly", "standalone_full_kelly"]
        wcols = [c for c in wcols if c in weights.columns]
        with pd.option_context("display.float_format", "{:.3f}".format,
                               "display.width", 160):
            log("\nPortfolio Kelly weights:\n" + weights[wcols].to_string(index=False))
            log("\nPortfolio summary:\n" + summary.drop(columns=["solver_ok"]).to_string(index=False))
        if not summary["solver_ok"].all():
            log("WARNING: optimizer did not fully converge for some fractions.")
        log(f"\nSaved: {w_path}\nSaved: {s_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
