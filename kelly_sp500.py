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

        rows.append({
            "ticker": t,
            "beta": beta,
            "expected_return": exp_ret,
            "annual_variance": var_annual,
            "annual_volatility": vol_annual,
            "sharpe_capm": (exp_ret - rf) / vol_annual,
            "hist_cagr": hist_cagr,
            "months_of_data": len(r),
            "full_kelly": kelly,
            "half_kelly": kelly / 2,
            "quarter_kelly": kelly / 4,
            "eighth_kelly": kelly / 8,
            "status": "ok",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ main ---
def default_output() -> Path:
    desktop = Path.home() / "Desktop"
    folder = desktop if desktop.exists() else Path.cwd()
    return folder / f"kelly_sp500_{date.today():%Y-%m-%d}.csv"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Kelly Criterion screener for the S&P 500")
    p.add_argument("--tickers", nargs="+", help="Run on specific tickers instead of the full index")
    p.add_argument("--limit", type=int, help="Only process the first N constituents (quick test)")
    p.add_argument("--market-return", type=float, default=MARKET_RETURN,
                   help=f"Long-run market return (default {MARKET_RETURN})")
    p.add_argument("--out", type=Path, default=None, help="Output CSV path")
    args = p.parse_args(argv)

    rf, rf_source = get_risk_free_rate()
    log(f"Risk-free rate: {rf:.3%} ({rf_source})  |  "
        f"Market premium: {args.market_return - rf:.3%}")

    meta = get_sp500_constituents()
    if args.tickers:
        tickers = [_yahoo_symbol(t) for t in args.tickers]
    else:
        tickers = meta["ticker"].tolist()
        if args.limit:
            tickers = tickers[:args.limit]

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
    show = ["ticker", "sector", "beta", "annual_volatility",
            "full_kelly", "half_kelly", "quarter_kelly"]
    with pd.option_context("display.float_format", "{:.3f}".format,
                           "display.width", 140):
        log("\nTop 20 by full Kelly:\n" + ok[show].head(20).to_string(index=False))

    out = args.out or default_output()
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    log(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
