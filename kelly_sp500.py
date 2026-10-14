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

Kelly (continuous / Merton form):  f* = (E[R] - Rf) / sigma^2
With CAPM E[R] = Rf + beta * MRP, this reduces to  f* = beta * MRP / sigma^2.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from datetime import date
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
        r = requests.get(FRED_URL, headers=UA, timeout=20)
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


def compute_kelly(prices: pd.DataFrame, rf: float, market_return: float,
                  benchmark: str = BENCHMARK) -> pd.DataFrame:
    mrp = market_return - rf
    rets = prices.pct_change(fill_method=None)
    beta_window = rets.iloc[-BETA_YEARS * 12:]
    mkt_b = beta_window[benchmark]

    rows = []
    for t in prices.columns:
        if t == benchmark:
            continue
        r = rets[t].dropna()
        if len(r) < MIN_MONTHS:
            rows.append({"ticker": t, "status": f"insufficient history ({len(r)} mo)"})
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

        rows.append({
            "ticker": t,
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
            "full_kelly": kelly,
            "half_kelly": kelly / 2,
            "quarter_kelly": kelly / 4,
            "eighth_kelly": kelly / 8,
            "status": "ok",
        })
    return pd.DataFrame(rows)


def portfolio_kelly(prices: pd.DataFrame, results: pd.DataFrame,
                    basket: list[str], rf: float,
                    max_weight: float = PORTFOLIO_MAX_WEIGHT,
                    max_leverage: float = PORTFOLIO_MAX_LEVERAGE,
                    shrink: float = COV_SHRINK,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Multi-asset Kelly: maximize expected log growth of the whole basket.

        maximize  w'(mu - rf) - (1 / 2c) * w' Sigma w
        s.t.      0 <= w_i <= max_weight,  sum(w) <= max_leverage

    c is the Kelly fraction (1, 1/2, 1/4, 1/8). Solving per fraction is the
    proper fractional Kelly (higher risk aversion), not just scaling weights.
    Whatever isn't invested sits in cash at rf.

    mu    : per-stock CAPM expected return (same as the screener)
    Sigma : annualized covariance of monthly returns over the window all
            names share, shrunk toward its diagonal to tame estimation noise.
    """
    from scipy.optimize import minimize

    ok = results.set_index("ticker")
    basket = [t for t in basket if t in ok.index and ok.at[t, "status"] == "ok"]
    if len(basket) < 2:
        raise ValueError("Portfolio needs at least 2 tickers with valid data.")

    rets = prices[basket].pct_change(fill_method=None).dropna()
    if len(rets) < MIN_MONTHS:
        raise ValueError(f"Only {len(rets)} months of overlapping history "
                         f"across the basket (need {MIN_MONTHS}).")

    sample = rets.cov().values * 12
    cov = (1 - shrink) * sample + shrink * np.diag(np.diag(sample))
    excess = ok.loc[basket, "expected_return"].values - rf
    n = len(basket)

    fractions = {"full": 1.0, "half": 0.5, "quarter": 0.25, "eighth": 0.125}
    weights, summary = {}, []
    for label, c in fractions.items():
        def neg_obj(w, c=c):
            return -(w @ excess - 0.5 / c * w @ cov @ w)

        def neg_grad(w, c=c):
            return -(excess - (1 / c) * cov @ w)

        res = minimize(
            neg_obj, x0=np.full(n, min(max_leverage, 1.0) / n), jac=neg_grad,
            method="SLSQP",
            bounds=[(0.0, max_weight)] * n,
            constraints=[{"type": "ineq",
                          "fun": lambda w: max_leverage - w.sum(),
                          "jac": lambda w: -np.ones(n)}],
            options={"maxiter": 500, "ftol": 1e-12},
        )
        w = np.clip(res.x, 0, None)
        w[w < 1e-5] = 0.0
        weights[f"weight_{label}_kelly"] = w

        port_excess = w @ excess
        port_var = w @ cov @ w
        summary.append({
            "kelly_fraction": label,
            "invested": w.sum(),
            "cash": 1 - w.sum(),
            "holdings": int((w > 0).sum()),
            "expected_return": rf + port_excess,
            "volatility": np.sqrt(port_var),
            "sharpe": port_excess / np.sqrt(port_var) if port_var > 0 else np.nan,
            "expected_growth_rate": rf + port_excess - 0.5 * port_var,
            "solver_ok": bool(res.success),
        })

    out = pd.DataFrame({"ticker": basket, **weights})
    meta_cols = ["name", "sector", "universe", "beta", "annual_volatility",
                 "expected_return", "full_kelly"]
    meta_cols = [c for c in meta_cols if c in ok.columns]
    out = out.merge(ok.loc[basket, meta_cols].reset_index(), on="ticker")
    out = out.rename(columns={"full_kelly": "standalone_full_kelly"})
    out = out.sort_values("weight_full_kelly", ascending=False)

    cash = {"ticker": "CASH",
            **{k: 1 - v.sum() for k, v in weights.items()}}
    out = pd.concat([out, pd.DataFrame([cash])], ignore_index=True)

    summ = pd.DataFrame(summary)
    summ.insert(1, "basket_size", n)
    summ.insert(2, "months_used", len(rets))
    return out, summ


# ------------------------------------------------------------------ main ---
def default_output() -> Path:
    desktop = Path.home() / "Desktop"
    folder = desktop if desktop.exists() else Path.cwd()
    return folder / f"kelly_sp500_{date.today():%Y-%m-%d}.csv"


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
    args = p.parse_args(argv)

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
    wanted = [_yahoo_symbol(t) for t in args.extra + (args.portfolio or [])]
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
    res = compute_kelly(prices, rf, args.market_return)
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
    show = ["ticker", "sector", "beta", "annual_volatility", "cagr_gap",
            "performance_category", "full_kelly", "half_kelly"]
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 160):
        log("\nTop 20 by full Kelly:\n" + ok[show].head(20).to_string(index=False))

    out = args.out or default_output()
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
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
            )
        except ValueError as e:
            log(f"\nPortfolio Kelly skipped: {e}")
            return 0
        stamp = f"{date.today():%Y-%m-%d}"
        w_path = out.parent / f"kelly_portfolio_{stamp}.csv"
        s_path = out.parent / f"kelly_portfolio_summary_{stamp}.csv"
        weights.to_csv(w_path, index=False)
        summary.to_csv(s_path, index=False)
        wcols = ["ticker", "weight_full_kelly", "weight_half_kelly",
                 "weight_quarter_kelly", "standalone_full_kelly"]
        with pd.option_context("display.float_format", "{:.3f}".format,
                               "display.width", 160):
            log("\nPortfolio Kelly weights:\n" + weights[wcols].to_string(index=False))
            log("\nPortfolio summary:\n" + summary.drop(columns="solver_ok").to_string(index=False))
        if not summary["solver_ok"].all():
            log("WARNING: optimizer did not fully converge for some fractions.")
        log(f"\nSaved: {w_path}\nSaved: {s_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
