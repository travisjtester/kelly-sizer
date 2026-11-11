"""
Kelly Sizer engine and command line: Kelly position sizing after
Concentrated Investing (Benello, van Biema & Carlisle, Wiley 2016), ch. 3-4.
The phone app (app.py) runs the same engine.

    python kelly.py NVDA ASML KO MSFT VOO      (see --help)

Pipeline
  1. Constituents  : Wikipedia (browser UA) -> GitHub datasets CSV fallback
  2. Risk-free rate: FRED Treasury matched to the horizon (DGS1..DGS10, no API
                     key) -> Yahoo ^FVX/^TNX fallback -> constant
  3. Prices        : ONE bulk yfinance download of monthly closes (auto-adjusted)
                     for every ticker + ^GSPC, in chunks, with retries
  4. Your views    : optional bear / base / bull scenarios per name (bear =
                     permanent loss), bull given directly or from an intrinsic
                     value estimate (margin of safety)
  5. Solo Kelly    : per name over a buy-and-hold horizon versus cash. Your
                     scenarios where you have a view; otherwise CAPM (expected
                     return Rf + beta * MRP, lognormal with historical vol)
  6. Portfolio     : joint weights maximizing expected log wealth over
                     correlated scenarios (Thorp's "joint properties" rule)
  7. Journal       : dated views + recorded outcomes -> calibration of your
                     probabilities
  8. Screen        : opt-in no-edge baseline for the whole S&P 500 (--screen)

Kelly: maximize E[log wealth]. For one win/lose bet this is the book's
(bp - q) / b; here it is solved for any outcome mix and for the basket.
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from contextlib import contextmanager
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

# Portfolio (multi-asset) Kelly defaults
PORTFOLIO_MAX_WEIGHT = 0.25   # per-name cap on the full-Kelly book
BOOK_MAX_WEIGHT = 0.40        # Buffett's largest ever (AmEx, 40%); app ceiling
PORTFOLIO_MAX_LEVERAGE = 1.0  # sum of weights; 1.0 = no margin
COV_SHRINK = 0.20             # blend sample covariance toward its diagonal

# Horizon engine (Concentrated Investing: buy and hold, edge from research)
DEFAULT_HORIZON = 3.0         # years for a thesis to play out (also CAPM names)
N_SCENARIOS = 20_000          # joint outcomes simulated for expected-log Kelly
VIEW_MIN_MONTHS = 12          # min shared months to estimate a correlation
SCENARIO_SEED = 7             # fixed so reruns give identical weights
# Joint outcomes are tied together with a Student-t copula: the same pairwise
# correlations as a Gaussian one, but bad outcomes land together more often,
# as they do in a crash. Lower = more crash clustering; None = Gaussian.
# Fitted t-copulas on stock returns typically land around 4-8.
COPULA_DOF = 5

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


# Cash over a buy-and-hold horizon should earn the Treasury that matures with
# it. (The book's Kelly formula takes cash at zero; a horizon-matched Treasury
# is the tool's extension.) Sources, in order: the Treasury's own daily par
# yield curve, then FRED's constant-maturity series (same maturities; FRED
# often doesn't answer cloud hosts such as GitHub's), then Yahoo, which
# carries only the 5- and 10-year, then a constant.
FRED_TREASURIES = {1: "DGS1", 2: "DGS2", 3: "DGS3", 5: "DGS5", 7: "DGS7", 10: "DGS10"}
TREASURY_URL = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve"
                "&field_tdr_date_value={year}&page&_format=csv")
YAHOO_TREASURIES = {5: "^FVX", 10: "^TNX"}
RATE_TIMEOUT = 20        # seconds; FRED can be slow to answer from cloud hosts
RATE_TRIES = 2


def _nearest_maturity(horizon: float, maturities) -> int:
    """The maturity closest to the horizon; a tie goes to the longer one."""
    return min(maturities, key=lambda m: (abs(m - horizon), -m))


def _get(url: str) -> str:
    """GET with a generous timeout and one retry."""
    for attempt in range(RATE_TRIES):
        try:
            r = requests.get(url, headers=UA, timeout=RATE_TIMEOUT)
            r.raise_for_status()
            return r.text
        except Exception:
            if attempt == RATE_TRIES - 1:
                raise
            time.sleep(2)


def _fred_yield(series: str) -> float:
    s = pd.read_csv(io.StringIO(_get(FRED_URL.replace("DGS10", series)))).iloc[:, 1]
    return float(pd.to_numeric(s, errors="coerce").dropna().iloc[-1]) / 100


def _treasury_yield(years: int) -> float:
    """Latest par yield for one maturity from home.treasury.gov (one CSV per
    year; early January falls back to last year's)."""
    col = f"{years} Yr"
    for year in (date.today().year, date.today().year - 1):
        df = pd.read_csv(io.StringIO(_get(TREASURY_URL.format(year=year))))
        if col not in df.columns or "Date" not in df.columns:
            continue
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce")
        s = pd.to_numeric(df.sort_values("Date")[col], errors="coerce").dropna()
        if len(s):
            return float(s.iloc[-1]) / 100
    raise ValueError(f"no {col} yields found")


BOOK_MODE_RF = (0.0, "book mode: cash earns 0%")


def risk_free_for(horizon: float, book_mode: bool = False) -> tuple[float, str]:
    """Book mode: cash earns nothing, as in the book's Kelly formula
    f* = (bp - q) / b (pp. 75-76), so a single bet's Kelly is exactly that
    formula. Otherwise the horizon-matched Treasury (the tool's extension)."""
    return BOOK_MODE_RF if book_mode else get_risk_free_rate(horizon)


def get_risk_free_rate(horizon: float = DEFAULT_HORIZON) -> tuple[float, str]:
    """Treasury yield matched to the horizon, as a decimal, plus the source used."""
    years = _nearest_maturity(horizon, FRED_TREASURIES)
    series = FRED_TREASURIES[years]
    try:
        return _treasury_yield(years), f"Treasury.gov {years} Yr"
    except Exception as e:
        log(f"Treasury.gov {years} Yr failed ({e}); trying FRED...")
    try:
        return _fred_yield(series), f"FRED {series}"
    except Exception as e:
        log(f"FRED {series} failed ({e}); trying Yahoo...")
    symbol = YAHOO_TREASURIES[_nearest_maturity(horizon, YAHOO_TREASURIES)]
    try:
        import yfinance as yf
        h = yf.Ticker(symbol).history(period="5d")["Close"].dropna()
        return float(h.iloc[-1]) / 100, f"Yahoo {symbol}"
    except Exception as e:
        log(f"{symbol} failed ({e}); using fallback {FALLBACK_RF:.2%}")
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
                frames.append(_refetch_blanks(close, chunk, retries,
                                              period=f"{years}y", interval="1mo"))
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



def _refetch_blanks(close: pd.DataFrame, tickers: list[str], retries: int,
                    **kw) -> pd.DataFrame:
    """
    A bulk download can come back with a ticker silently blank: yfinance
    logs the error (e.g. its cache's "database is locked" when threads collide)
    and leaves the column empty. Fetch those again one at a time, without
    threads, rather than let a name quietly drop out of the basket.
    """
    import yfinance as yf

    close = close.reindex(columns=list(dict.fromkeys([*close.columns, *tickers])))
    for t in [t for t in tickers if close[t].isna().all()]:
        for attempt in range(retries):
            try:
                one = yf.download(t, auto_adjust=True, threads=False, progress=False,
                                  **kw)["Close"]
                one = one.iloc[:, 0] if isinstance(one, pd.DataFrame) else one
                if one.notna().any():
                    close = close.reindex(close.index.union(one.index))
                    close[t] = one.reindex(close.index)
                    break
            except Exception:
                pass
            time.sleep(2 * 2 ** attempt)
        else:
            log(f"  {t}: no prices after {retries} retries")
    return close


def download_latest_prices(tickers: list[str]) -> dict[str, float]:
    """Most recent daily close per ticker (for margin of safety vs. value)."""
    import yfinance as yf

    if not tickers:
        return {}
    try:
        raw = yf.download(list(tickers), period="5d", interval="1d",
                          auto_adjust=True, group_by="column", threads=True,
                          progress=False)
        close = raw["Close"]
        if isinstance(close, pd.Series):
            close = close.to_frame(tickers[0])
        close = _refetch_blanks(close, list(tickers), 2, period="5d", interval="1d")
        last = close.ffill().iloc[-1].dropna()
        return {k: float(v) for k, v in last.items()}
    except Exception as e:
        log(f"Latest prices failed ({e}); using last month-end closes")
        return {}


# ---------------------------------------------------------- market odds ---
# The tote board. Munger: winning bettors "look and sift the world for a
# mispriced bet" (pp. 209-210). Option prices are the market's odds; your edge
# is where your odds differ from them. These numbers are shown BESIDE your
# odds, after you've entered them, and never feed the sizing: copying them in
# would make your edge zero by construction.
#
# Caveats, stated wherever they're shown: option-implied odds are risk-neutral
# (they carry a risk premium, so downside reads a little high and upside a
# little low), they price a share price on a date rather than a permanent
# loss, and listed options rarely run past ~2 years.
OPTION_MAX_SPREAD = 0.5     # skip quotes whose bid/ask spread is over 50% of the mid
OPTION_MIN_YEARS = 0.1      # ignore expiries closer than ~5 weeks


def _bs_price(S: float, K: float, T: float, r: float, vol: float, call: bool) -> float:
    """Black-Scholes price (European, no dividends); r is continuous."""
    from scipy.stats import norm
    sd = vol * np.sqrt(T)
    d1 = (np.log(S / K) + (r + vol ** 2 / 2) * T) / sd
    d2 = d1 - sd
    if call:
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def _implied_vol(price: float, S: float, K: float, T: float, r: float,
                 call: bool) -> float | None:
    """The volatility that reproduces a quoted price, or None if no volatility
    between 1% and 500% does (a stale or nonsensical quote)."""
    from scipy.optimize import brentq
    f = lambda v: _bs_price(S, K, T, r, v, call) - price   # noqa: E731
    lo, hi = 0.01, 5.0
    try:
        if f(lo) > 0 or f(hi) < 0:
            return None
        return float(brentq(f, lo, hi, xtol=1e-6))
    except (ValueError, ZeroDivisionError):
        return None


def option_smile(calls: pd.DataFrame, puts: pd.DataFrame, S: float, T: float,
                 r: float) -> pd.DataFrame:
    """
    Implied volatility by strike, from out-of-the-money quotes (puts below
    today's price, calls at or above: the liquid side of each strike). Each
    volatility is solved from the bid/ask MID, not Yahoo's own figure, which
    can be computed from a stale last trade. Quotes with no bid, a crossed
    market, or a spread over OPTION_MAX_SPREAD of the mid are skipped.
    """
    rows = []
    for side, call in ((puts, False), (calls, True)):
        if side is None or not len(side):
            continue
        q = side[["strike", "bid", "ask"]].apply(pd.to_numeric, errors="coerce").dropna()
        q = q[(q.strike < S) if not call else (q.strike >= S)]
        q = q[(q.bid > 0) & (q.ask >= q.bid)]
        mid = (q.bid + q.ask) / 2
        q = q[(q.ask - q.bid) <= OPTION_MAX_SPREAD * mid]
        for K, m in zip(q.strike, (q.bid + q.ask) / 2):
            iv = _implied_vol(float(m), S, float(K), T, r, call)
            if iv is not None:
                rows.append((float(K), iv))
    return pd.DataFrame(rows, columns=["strike", "iv"]).sort_values("strike").reset_index(drop=True)


def _iv_at(smile: pd.DataFrame, K: float) -> float | None:
    """Volatility at strike K, on a straight line between the nearest traded
    strikes. None outside the traded range: no extrapolating into thin air."""
    if smile.empty or not (smile.strike.min() <= K <= smile.strike.max()):
        return None
    return float(np.interp(K, smile.strike, smile.iv))


def implied_odds(smile: pd.DataFrame, S: float, T: float, r: float,
                 bear_price: float, bull_price: float) -> tuple[float, float, str]:
    """
    Market-implied (risk-neutral) chance of finishing below bear_price and
    above bull_price at the expiry: N(-d2) and N(d2), each with the volatility
    at its own strike (the skew matters most for the bear case: far-down puts
    carry more volatility than at-the-money ones). NaN when there's no
    reliable quote near that price; the note says which.
    """
    from scipy.stats import norm

    def d2(K, vol):
        return (np.log(S / K) + (r - vol ** 2 / 2) * T) / (vol * np.sqrt(T))
    notes = []
    p = {}
    for label, K, below in (("bear", bear_price, True), ("bull", bull_price, False)):
        vol = _iv_at(smile, K) if K and K > 0 else None
        if vol is None:
            p[label] = np.nan
            notes.append(f"no reliable quotes near the {label} price ${K:,.2f}")
        else:
            p[label] = float(norm.cdf(-d2(K, vol)) if below else norm.cdf(d2(K, vol)))
    return p["bear"], p["bull"], "; ".join(notes)


def _pick_expiry(expiries: list[str], horizon: float, today: date) -> tuple[str, float] | None:
    """The longest expiry at or under the horizon (the longest of all if every
    one is shorter), skipping any closer than OPTION_MIN_YEARS."""
    dated = []
    for e in expiries:
        try:
            years = (datetime.strptime(e, "%Y-%m-%d").date() - today).days / 365.25
        except ValueError:
            continue
        if years >= OPTION_MIN_YEARS:
            dated.append((years, e))
    if not dated:
        return None
    fits = [d for d in dated if d[0] <= horizon + 1 / 52]
    years, e = max(fits) if fits else min(dated)
    return e, years


def fetch_option_smile(ticker: str, spot: float, horizon: float, rf: float,
                       today: date | None = None) -> dict:
    """
    One option chain from Yahoo: the expiry closest to (at or under) the
    horizon, and its smile. Returns {expiry, years, smile, note}; never raises,
    since market odds are a reference and must not block sizing.
    """
    import yfinance as yf

    out = {"expiry": None, "years": np.nan, "smile": pd.DataFrame(columns=["strike", "iv"]),
           "note": ""}
    try:
        tk = yf.Ticker(ticker)
        pick = _pick_expiry(list(tk.options or []), horizon, today or date.today())
        if pick is None:
            out["note"] = "no listed options"
            return out
        out["expiry"], out["years"] = pick
        chain = tk.option_chain(out["expiry"])
        out["smile"] = option_smile(chain.calls, chain.puts, spot, out["years"],
                                    np.log1p(rf))
        if out["smile"].empty:
            out["note"] = "no reliable quotes"
    except Exception as e:
        out["note"] = f"option data unavailable ({type(e).__name__})"
    return out


def market_check(views: pd.DataFrame, prices_now: dict[str, float], horizon: float,
                 rf: float, smiles: dict[str, dict] | None = None) -> pd.DataFrame:
    """
    Your odds beside the market's, one row per view with a usable price. Bear
    and bull PRICES come from your view (re-anchored to today if it carries an
    as_of_price; the bull from your value where you gave one). smiles can be
    passed in (the app caches them); otherwise they're fetched.
    """
    cols = ["ticker", "your_bear", "market_bear", "bear_price", "your_bull", "market_bull",
            "bull_price", "expiry", "years", "note"]
    if views is None or not len(views):
        return pd.DataFrame(columns=cols)
    with captured_log():
        v = resolve_views(views, prices_now)
    rows = []
    r = np.log1p(rf)
    for t, x in v.iterrows():
        S = x["price"]
        bear_px, bull_px = S * (1 - x["bear"]), S * (1 + x["bull"])
        sm = (smiles or {}).get(t) or fetch_option_smile(t, S, horizon, rf)
        if sm["smile"].empty:
            pb, pu, note = np.nan, np.nan, sm["note"]
        else:
            pb, pu, note = implied_odds(sm["smile"], S, sm["years"], r, bear_px, bull_px)
            if sm["years"] < horizon - 0.25:
                note = "; ".join(filter(None, [f"longest options run {sm['years']:.1f}y, "
                                               f"shorter than your {horizon:g}y horizon", note]))
        rows.append({"ticker": t, "your_bear": x["p_bear"], "market_bear": pb,
                     "bear_price": bear_px, "your_bull": x["p_bull"], "market_bull": pu,
                     "bull_price": bull_px, "expiry": sm["expiry"], "years": sm["years"],
                     "note": note})
    return pd.DataFrame(rows, columns=cols)


MARKET_GAP = 0.10   # a gap of 10+ points between your odds and the market's gets a sentence


def market_gaps(check: pd.DataFrame) -> list[str]:
    """Plain-words gaps worth explaining to yourself (the mispriced-bet question)."""
    out = []
    for _, r in check.iterrows():
        if pd.notna(r.market_bear) and abs(r.your_bear - r.market_bear) >= MARKET_GAP:
            if r.your_bear < r.market_bear:
                ask = "You see less downside than the market. What do you know that it doesn't?"
            else:
                ask = ("You see more downside than the market. Is the bear case too heavy, "
                       "or is the stock cheap for a reason?")
            out.append(f"{r.ticker}: you give the bear case {r.your_bear:.0%}; the market "
                       f"prices {r.market_bear:.0%} below ${r.bear_price:,.2f}. {ask}")
        if pd.notna(r.market_bull) and abs(r.your_bull - r.market_bull) >= MARKET_GAP:
            more = r.your_bull > r.market_bull
            out.append(f"{r.ticker}: you give the bull case {r.your_bull:.0%}; the market "
                       f"prices {r.market_bull:.0%} above ${r.bull_price:,.2f}. You're "
                       f"{'more' if more else 'less'} optimistic than the market.")
    return out


# ------------------------------------------------------- street starter ---
# A consensus starting point, not an edge. Analysts' low / mean / high
# 12-month price targets become a bear / base / bull view with placeholder odds,
# for names you haven't researched yet, so there's something to edit rather
# than a blank row. Sell-side targets lean bullish (the first live test sized
# them at 41%/yr expected growth vs. 6.8% with no views), so every starter view
# is labeled, carries horizon = 1 (targets are 12-month, so a longer run flags
# the mismatch) and as_of_price (so it re-anchors as the price moves). The
# book's edge comes from your own research (p. 1, Preface p. xi): replace the
# odds and the bear case with yours.
STREET_ODDS = (0.20, 0.50)        # placeholder bear / base odds; bull gets the rest
STREET_BEAR_FLOOR = 0.20          # a low target isn't a broken thesis: at least a 20% loss
STREET_NOTE = "Street consensus starter: replace the odds and bear case with your research"


def street_targets(tickers: list[str]) -> pd.DataFrame:
    """Analyst price targets from Yahoo: low / mean / high, analyst count and
    the current price. Best effort: a name without targets gets a note."""
    import yfinance as yf

    rows = []
    for t in tickers:
        r = {"ticker": t, "price": np.nan, "low": np.nan, "mean": np.nan, "high": np.nan,
             "analysts": np.nan, "note": ""}
        try:
            info = yf.Ticker(t).info or {}
            r.update(price=info.get("currentPrice") or info.get("regularMarketPrice"),
                     low=info.get("targetLowPrice"), mean=info.get("targetMeanPrice"),
                     high=info.get("targetHighPrice"),
                     analysts=info.get("numberOfAnalystOpinions"))
        except Exception as e:
            r["note"] = f"targets unavailable ({type(e).__name__})"
        rows.append(r)
    out = pd.DataFrame(rows)
    num = ["price", "low", "mean", "high", "analysts"]
    out[num] = out[num].apply(pd.to_numeric, errors="coerce")
    return out


def street_views(targets: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Starter views from street_targets(), and notes on each name: which were
    skipped and why, and where the bear case was floored (a low target above
    the price, or barely below it, isn't a permanent loss).
    """
    p_bear, p_base = STREET_ODDS
    rows, notes = [], []
    for _, r in targets.iterrows():
        t = r["ticker"]
        if r["note"]:
            notes.append(f"{t}: skipped, {r['note']}")
            continue
        if not all(pd.notna(r[c]) and r[c] > 0 for c in ["price", "low", "mean", "high"]):
            notes.append(f"{t}: skipped, no analyst price targets")
            continue
        S = r["price"]
        bull = r["high"] / S - 1
        if bull <= 0:
            notes.append(f"{t}: skipped, already above the highest target (${r['high']:,.2f})")
            continue
        raw_bear = 1 - r["low"] / S
        bear = min(max(raw_bear, STREET_BEAR_FLOOR), 0.95)
        n = f"{r['analysts']:.0f} analysts; " if pd.notna(r["analysts"]) else ""
        rows.append({"ticker": t, "p_bear": p_bear, "bear": round(bear, 4), "p_base": p_base,
                     "base": round(r["mean"] / S - 1, 4), "bull": round(bull, 4),
                     "as_of_price": S, "horizon": 1.0,
                     "notes": f"{STREET_NOTE} ({n}low/mean/high ${r['low']:,.0f}/"
                              f"${r['mean']:,.0f}/${r['high']:,.0f})"})
        if raw_bear < STREET_BEAR_FLOOR:
            why = ("above the price" if raw_bear <= 0 else f"only {raw_bear:.0%} below it")
            notes.append(f"{t}: lowest target is {why}, so the bear case is set at a "
                         f"{STREET_BEAR_FLOOR:.0%} loss. Write your own.")
    views = normalize_views(pd.DataFrame(rows)) if rows else normalize_views(None)
    return views, notes


# ---------------------------------------------------------------- views ---
# A view is up to three outcomes for a thesis over the horizon:
#   bear : PERMANENT loss of capital (the book's definition of risk), prob p_bear
#   base : optional middle case, prob p_base
#   bull : the thesis plays out, prob p_bull = 1 - p_bear - p_base
# The bull return can be given directly or derived from an intrinsic value
# estimate: bull = value / price - 1 (margin of safety = 1 - price / value).
VIEW_COLS = ["p_bear", "bear", "p_base", "base", "bull", "value"]
# Optional context for a view. as_of_price: the share price the returns were
# measured from; the scenario PRICES are held fixed and the returns re-measured
# from today's price (stale views re-anchor themselves). horizon: the years the
# view assumes; a mismatch with the run's horizon is flagged, not rescaled.
VIEW_CONTEXT_COLS = ["as_of_price", "horizon"]
LEGACY_VIEW_COLS = ["p_win", "upside", "downside"]

UNIT_COLS = ["p_bear", "p_base", "p_bull", "bear", "p_win", "downside"]   # always <= 1 as decimals


def _file_uses_percents(d: pd.DataFrame) -> bool:
    """A views file uses one set of units. The odds and loss columns can't
    exceed 1 as decimals, so any plain value above 1 there means the whole
    file is in percents (40 = 40%); otherwise it's decimals (0.4 = 40%)."""
    for c in UNIT_COLS:
        if c in d.columns:
            txt = d[c].astype(str).str.strip()
            num = pd.to_numeric(txt[~txt.str.endswith("%")], errors="coerce").dropna()
            if (num.abs() > 1).any():
                return True
    return False


def _column_to_fraction(col: pd.Series, as_pct: bool) -> pd.Series:
    """Parse one column. A '%' sign always means percent; plain numbers follow
    the file's units."""
    txt = col.astype(str).str.strip().replace({"nan": "", "None": "", "<NA>": ""})
    has_pct = txt.str.endswith("%")
    num = pd.to_numeric(txt.str.rstrip("%"), errors="coerce")
    return num / 100 if as_pct else num.where(~has_pct, num / 100)


def _warn_mixed_bear_signs(d: pd.DataFrame) -> None:
    """Bear is read as the size of the loss, so its sign is dropped. That's
    fine while a file is consistent, but mixed signs usually mean some rows
    were computed as a return (low target / price - 1): a low target above the
    price then comes out positive and would be read as a loss. Flag the rows
    that disagree with the rest of the file."""
    neg, pos = d["bear"] < 0, d["bear"] > 0
    if not (neg.any() and pos.any()):
        return
    odd = pos if neg.sum() >= pos.sum() else neg
    log("Views: bear signs are mixed; check " + ", ".join(d.loc[odd, "ticker"])
        + ". Bear is read as the size of the loss whatever its sign, so a row worked "
          "out as low target / price − 1 with the low target above the price would "
          "become a loss that isn't there.")


def normalize_views(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    Clean a views table to index=ticker with VIEW_COLS (+ p_bull, notes).
    Accepts the scenario format (p_bear, bear, [p_base, base], bull and/or
    value) or the older binary format (p_win, upside, downside). Rows without
    a usable bear case and a bull return or value are ignored (CAPM is used).
    """
    cols = VIEW_COLS + ["p_bull", "notes"] + VIEW_CONTEXT_COLS
    empty = pd.DataFrame(columns=cols).rename_axis("ticker")
    if df is None or df.empty:
        return empty
    d = df.copy()
    if d.index.name == "ticker" and "ticker" not in d.columns:
        d = d.reset_index()                      # already normalized: idempotent
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "ticker" not in d.columns:
        raise ValueError("Views need a ticker column")
    d["ticker"] = d["ticker"].astype(str).map(_yahoo_symbol).str.upper()

    legacy = set(LEGACY_VIEW_COLS) <= set(d.columns) and "p_bear" not in d.columns
    if not legacy and not {"p_bear", "bear"} <= set(d.columns):
        raise ValueError("Views need p_bear and bear (or the older p_win, upside, downside)")
    as_pct = _file_uses_percents(d)
    if legacy:
        for c in LEGACY_VIEW_COLS:
            d[c] = _column_to_fraction(d[c], as_pct)
        d["p_bear"] = 1 - d["p_win"]
        d["bear"] = d["downside"]
        d["bull"] = d["upside"]
    for c in VIEW_COLS:
        if c not in d.columns:
            d[c] = np.nan
    done = {"p_bear", "bear", "bull"} if legacy else set()    # already converted above
    for c in ["p_bear", "bear", "p_base", "base", "bull"]:
        if c not in done:
            d[c] = _column_to_fraction(d[c], as_pct)
    d["value"] = pd.to_numeric(d["value"].astype(str).str.replace(r"[$,]", "", regex=True),
                               errors="coerce")
    for c in VIEW_CONTEXT_COLS:            # optional; blank, zero or junk = not given
        raw = d[c] if c in d.columns else pd.Series(np.nan, index=d.index)
        num = pd.to_numeric(raw.astype(str).str.replace(r"[$,]", "", regex=True),
                            errors="coerce")
        d[c] = num.where(num > 0)
    _warn_mixed_bear_signs(d)
    d["bear"] = d["bear"].abs()     # a loss either way: -0.30 and 0.30 both mean lose 30%
    d["p_base"] = d["p_base"].fillna(0.0)
    d.loc[d["p_base"] == 0, "base"] = np.nan

    d = d.dropna(subset=["p_bear", "bear"])
    d = d[d["bull"].notna() | d["value"].notna()]
    d["p_bull"] = 1 - d["p_bear"] - d["p_base"]

    bad = d[(d.p_bear <= 0) | (d.p_bear >= 1) | (d.bear <= 0) | (d.bear > 1)
            | (d.p_base < 0) | (d.p_bull <= 0)
            | ((d.p_base > 0) & (d.base.isna() | (d.base <= -1)))
            | (d.bull.notna() & (d.bull <= 0)) | (d.value.notna() & (d.value <= 0))]
    if not bad.empty:
        raise ValueError(
            "Invalid views for " + ", ".join(bad["ticker"]) + ". Need 0 < bear odds < 1, "
            "0 < bear loss <= 100%, bear + base odds < 100%, a base return when base "
            "odds are set, and a positive bull return or value.")
    odd = d[d.bull.notna() & ((d.bull < 0.02) | (d.bull > 10))]
    if not odd.empty:
        log("Views: check units for " + ", ".join(odd["ticker"]) + " (bull under +2% or "
            "over +1000%). Use the same units as your odds: 0.4 and 1.5, or 40 and 150")
    if "notes" not in d.columns:
        d["notes"] = ""
    d["notes"] = d["notes"].fillna("")
    return d.drop_duplicates("ticker", keep="last").set_index("ticker")[cols]


def load_views(path: Path | str) -> pd.DataFrame:
    return normalize_views(pd.read_csv(path))


def resolve_views(views: pd.DataFrame, prices_now: dict[str, float]) -> pd.DataFrame:
    """
    Attach the current price, derive the bull return from intrinsic value
    where it wasn't given, and compute margin of safety (1 - price / value).
    Views that need a price to work but have none are dropped with a note.
    """
    v = normalize_views(views).copy()
    v["price"] = [prices_now.get(t, np.nan) for t in v.index]
    breached = _reanchor(v)
    v["margin_of_safety"] = 1 - v["price"] / v["value"]
    need = v["bull"].isna()
    v.loc[need, "bull"] = v.loc[need, "value"] / v.loc[need, "price"] - 1
    unusable = v[v["bull"].isna() | (v["bull"] <= 0)].drop(index=list(breached), errors="ignore")
    for t, r in unusable.iterrows():
        if pd.isna(r["price"]):
            why = "no current price to compare with its value"
        elif need[t]:
            why = f"value ${r['value']:,.2f} is below the price ${r['price']:,.2f}: no upside"
        else:
            why = f"the price ${r['price']:,.2f} is already past your bull case: no upside"
        log(f"Views: ignoring {t} ({why}); it falls back to CAPM")
    for t, bear_px in breached.items():
        log(f"Views: ignoring {t} (the price ${v.at[t, 'price']:,.2f} is already at or "
            f"below your bear price ${bear_px:,.2f}; revisit the view); it falls back to CAPM")
    return v.drop(index=unusable.index.union(list(breached)))


REANCHOR_NOTE = 0.05    # mention a re-anchored view once the price has moved 5%+


def _reanchor(v: pd.DataFrame) -> dict[str, float]:
    """
    Views with an as_of_price: hold each scenario's PRICE fixed and re-measure
    its return from today's price, in place. A view written at $100 with a 30%
    bear (bear price $70) reads as a 22% bear once the stock is $90. A bull
    given as a return is re-anchored the same way; one given as value already
    uses today's price. Returns {ticker: bear price} for views whose bear price
    the stock has already fallen through, which can't be sized as written.
    """
    k = v["as_of_price"] / v["price"]
    breached = {}
    for t in v.index[k.notna() & np.isfinite(k)]:
        r, kk = v.loc[t], k[t]
        bear_px = r["as_of_price"] * (1 - r["bear"])
        v.at[t, "bear"] = 1 - kk * (1 - r["bear"])
        if pd.notna(r["base"]):
            v.at[t, "base"] = kk * (1 + r["base"]) - 1
        if pd.notna(r["bull"]):
            v.at[t, "bull"] = kk * (1 + r["bull"]) - 1
        if v.at[t, "bear"] <= 0:
            breached[t] = bear_px
        elif abs(kk - 1) >= REANCHOR_NOTE:
            log(f"Views: {t} has moved {1 / kk - 1:+.0%} since your view at "
                f"${r['as_of_price']:,.2f}; scenario prices held, returns re-measured "
                f"from ${r['price']:,.2f}")
    return breached


def neutralize_view(v: pd.Series, target_return: float) -> pd.Series:
    """
    Neutral mode, for testing inputs: keep a view's shape (the spread between
    bear, base and bull) but scale every outcome so its expected return over
    the horizon equals target_return (the name's CAPM return). With the edge
    gone, sizes should land near the CAPM baseline; what's left is the effect
    of the scenario shape alone. Gross returns are scaled, so outcomes keep
    their order and never go below -100%.
    """
    probs, rets = view_outcomes(v)
    m = (1 + target_return) / (1 + float(probs @ rets))
    v = v.copy()
    v["bear"] = 1 - (1 - v["bear"]) * m        # may turn into a small gain: that's the shape
    if v["p_base"] > 0:
        v["base"] = (1 + v["base"]) * m - 1
    v["bull"] = (1 + v["bull"]) * m - 1
    return v


def view_outcomes(v: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(probabilities, horizon returns), ordered bear -> base -> bull."""
    probs, rets = [v["p_bear"]], [-v["bear"]]
    if v["p_base"] > 0:
        probs.append(v["p_base"]); rets.append(v["base"])
    probs.append(v["p_bull"]); rets.append(v["bull"])
    return np.array(probs, float), np.array(rets, float)


def binary_kelly(p: float, upside: float, downside: float, rf_h: float) -> float:
    """
    Closed-form Kelly for one win/lose bet against cash over the horizon:
        f* = (1 + rf) * ( p / (downside + rf) - q / (upside - rf) )
    With rf = 0 and a total-loss downside it is the book's (bp - q) / b.
    Kept as a reference; discrete_kelly() solves the general case.
    """
    q = 1 - p
    if upside <= rf_h:
        return 0.0
    return max((1 + rf_h) * (p / (downside + rf_h) - q / (upside - rf_h)), 0.0)


def discrete_kelly(probs: np.ndarray, rets: np.ndarray, rf_h: float,
                   cap: float = 10.0) -> float:
    """
    Kelly fraction for one bet with any number of outcomes, versus cash at
    rf_h: maximize sum p * log(1 + rf + f (r - rf)). Concave, so no edge
    (expected return <= rf) means zero. The fraction is kept below the point
    where the worst outcome would wipe out the bankroll: never risks ruin.
    """
    from scipy.optimize import minimize_scalar

    x = rets - rf_h
    if probs @ x <= 0:
        return 0.0
    worst = x.min()
    f_ruin = (1 + rf_h) / -worst if worst < 0 else np.inf
    hi = min(cap, f_ruin * (1 - 1e-9))
    obj = lambda f: -(probs @ np.log(1 + rf_h + f * x))
    res = minimize_scalar(obj, bounds=(0.0, hi), method="bounded",
                          options={"xatol": 1e-10})
    return float(res.x)


def lognormal_outcomes(expected_return: float, vol: float, horizon: float,
                       nodes: int = 41) -> tuple[np.ndarray, np.ndarray]:
    """CAPM name over the horizon as a lognormal, discretized with
    Gauss-Hermite quadrature (deterministic, accurate to many decimals)."""
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    mu_log = np.log1p(expected_return) - vol ** 2 / 2
    return w, np.expm1(mu_log * horizon + vol * np.sqrt(horizon) * x)


def view_stats(v: pd.Series, rf: float, horizon: float) -> dict:
    rf_h = (1 + rf) ** horizon - 1
    probs, rets = view_outcomes(v)
    exp_h = float(probs @ rets)
    f = discrete_kelly(probs, rets, rf_h)
    return {
        "edge_source": "views",
        **{c: v.get(c, np.nan) for c in
           ["p_bear", "bear", "p_base", "base", "p_bull", "bull", "value",
            "price", "margin_of_safety"]},
        "horizon_years": horizon,
        "view_expected_return": exp_h,
        "view_annualized_return": (1 + exp_h) ** (1 / horizon) - 1 if exp_h > -1 else -1.0,
        "full_kelly": f, "half_kelly": f / 2,
        "quarter_kelly": f / 4, "eighth_kelly": f / 8,
    }


def compute_kelly(prices: pd.DataFrame, rf: float, market_return: float,
                  benchmark: str = BENCHMARK,
                  views: pd.DataFrame | None = None,
                  horizon: float = DEFAULT_HORIZON,
                  prices_now: dict[str, float] | None = None,
                  neutral: bool = False) -> pd.DataFrame:
    """
    Per-name solo Kelly over the horizon (buy and hold, versus cash).
    neutral=True reshapes each view to its CAPM expected return (see
    neutralize_view): a test mode that removes the edge but keeps the shape.

    CAPM supplies the edge by default: expected return = Rf + beta * MRP,
    outcomes lognormal with historical volatility. A name with a view uses
    your scenarios instead (edge_source = 'views'); its CAPM figure stays in
    capm_full_kelly for comparison. Names with a view are kept even with
    short price history. prices_now (latest prices) is needed only for views
    given as intrinsic value; it falls back to the last month-end close.
    """
    views = normalize_views(views) if views is not None else normalize_views(None)
    if len(views):
        last_close = prices.ffill().iloc[-1].to_dict() if len(prices) else {}
        views = resolve_views(views, {**last_close, **(prices_now or {})})
        _warn_horizon_mismatch(views, horizon)
    rf_h = (1 + rf) ** horizon - 1
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
                if neutral:
                    log(f"Neutral mode: {t} has too little history for a CAPM return, "
                        "so its view is sized as entered")
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
        kelly = discrete_kelly(*lognormal_outcomes(exp_ret, vol_annual, horizon), rf_h)

        row = {
            "ticker": t,
            "edge_source": "CAPM",
            "beta": beta,
            "expected_return": exp_ret,
            "annual_variance": var_annual,
            "annual_volatility": vol_annual,
            "hist_cagr": (1 + r).prod() ** (12 / len(r)) - 1,
            "months_of_data": len(r),
            "horizon_years": horizon,
            "capm_full_kelly": kelly,
            "full_kelly": kelly,
            "half_kelly": kelly / 2,
            "quarter_kelly": kelly / 4,
            "eighth_kelly": kelly / 8,
            "status": "ok",
        }
        if has_view:
            v = views.loc[t]
            if neutral:
                v = neutralize_view(v, (1 + exp_ret) ** horizon - 1)
            row.update(view_stats(v, rf, horizon))
        rows.append(row)
    return pd.DataFrame(rows)


def _warn_horizon_mismatch(views: pd.DataFrame, horizon: float) -> None:
    """Views carry no time scale of their own: a 12-month target run over a
    3-year hold compares one year of upside with three years of cash. Flag
    views whose stated horizon differs from the run's."""
    off = views[views["horizon"].notna() & ((views["horizon"] - horizon).abs() > 1e-9)]
    if len(off):
        names = ", ".join(f"{t} ({h:g}y)" for t, h in off["horizon"].items())
        log(f"Views: some views assume a different horizon: {names}. This run is {horizon:g} "
            f"years and their returns aren't rescaled; set the horizon to match or "
            f"restate those views.")


# ------------------------------------------------------------ portfolio ---
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


def _correlation(prices: pd.DataFrame, basket: list[str], shrink: float) -> tuple[np.ndarray, int]:
    """Pairwise correlations of monthly returns (each pair uses the months it
    shares, min VIEW_MIN_MONTHS), shrunk toward independence and repaired to be
    positive semi-definite. A pair without enough shared months (a recent
    listing) gets the basket's average correlation, not zero: treating a new
    name as independent would understate how concentrated the basket is."""
    rets = prices[basket].pct_change(fill_method=None)
    corr = rets.corr(min_periods=VIEW_MIN_MONTHS).to_numpy(copy=True)
    off = ~np.eye(len(basket), dtype=bool)
    known = corr[off & ~np.isnan(corr)]
    fill = max(float(known.mean()), 0.0) if known.size else 0.0
    corr = np.where(np.isnan(corr), fill, corr)
    np.fill_diagonal(corr, 1.0)
    corr = (1 - shrink) * corr + shrink * np.eye(len(basket))
    vals, vecs = np.linalg.eigh(corr)
    corr = vecs @ np.diag(np.clip(vals, 1e-8, None)) @ vecs.T
    d = np.sqrt(np.diag(corr))
    corr = corr / np.outer(d, d)
    for t in basket:
        if rets[t].notna().sum() < VIEW_MIN_MONTHS:
            log(f"Portfolio: {t} has <{VIEW_MIN_MONTHS} months of prices; "
                f"assumed to move with the basket average (correlation {fill:.2f})")
    return corr, int(rets.dropna(how="all").shape[0])


def simulate_outcomes(ok: pd.DataFrame, basket: list[str], corr: np.ndarray,
                      horizon: float, n_scenarios: int = N_SCENARIOS,
                      seed: int = SCENARIO_SEED) -> np.ndarray:
    """
    Joint horizon returns, shape (n_scenarios, n_names), via a Student-t
    copula (COPULA_DOF) on historical correlations. Each name keeps its own
    outcomes; the copula only sets how they line up, with joint extremes
    (crashes) more common than a Gaussian copula allows:
      - name with a view : bear / base / bull with your probabilities, ordered
                           so correlated names tend to land on the same side
      - name without one : lognormal with CAPM mean and historical volatility
    """
    from scipy.stats import norm, t as student_t

    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_scenarios, len(basket))) @ L.T
    if COPULA_DOF:
        # One shared shock scale per scenario: when it's large, every name is
        # pushed toward its tails at once.
        s = np.sqrt(rng.chisquare(COPULA_DOF, size=(n_scenarios, 1)) / COPULA_DOF)
        u = student_t.cdf(z / s, COPULA_DOF)
        z = norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))    # normal scores for lognormal names
    else:
        u = norm.cdf(z)
    out = np.empty_like(z)
    for j, t in enumerate(basket):
        row = ok.loc[t]
        if row.get("edge_source") == "views":
            probs, rets = view_outcomes(row)
            idx = np.searchsorted(np.cumsum(probs)[:-1], u[:, j], side="right")
            out[:, j] = rets[idx]
        else:
            vol = row["annual_volatility"]
            mu_log = np.log1p(row["expected_return"]) - vol ** 2 / 2
            out[:, j] = np.expm1(mu_log * horizon + vol * np.sqrt(horizon) * z[:, j])
    return out


def _solve(neg_obj, neg_grad, n: int, max_weight: float, max_leverage: float):
    from scipy.optimize import minimize
    res = minimize(
        neg_obj, x0=np.full(n, min(max_leverage, 1.0) / n / 2), jac=neg_grad,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=[{"type": "ineq",
                      "fun": lambda w: max_leverage - w.sum(),
                      "jac": lambda w: -np.ones(n)}],
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    w = np.clip(res.x, 0, None)
    w[w < 1e-5] = 0.0
    return w, bool(res.success)


_RUIN_EPS = 1e-3


def _smooth_log(W: np.ndarray) -> np.ndarray:
    """log(W), continued below a tiny wealth level as a steep downward parabola
    instead of a flat floor. Ruin stays catastrophic, but the optimizer always
    has a slope pointing back to safety (a flat floor made SLSQP stall)."""
    e = _RUIN_EPS
    d = W - e
    return np.where(W >= e, np.log(np.maximum(W, e)), np.log(e) + d / e - d * d / (2 * e * e))


def _smooth_log_grad(W: np.ndarray) -> np.ndarray:
    e = _RUIN_EPS
    return np.where(W >= e, 1 / np.maximum(W, e), 1 / e - (W - e) / (e * e))


def basket_stats(w: np.ndarray, X: np.ndarray, rf_h: float, horizon: float) -> dict:
    """Outcome stats for weights w over scenario excess returns X (horizon)."""
    W = 1 + rf_h + X @ w
    logW = np.log(np.maximum(W, 1e-12))
    return {
        "invested": w.sum(), "cash": 1 - w.sum(),
        "holdings": int((w > 0).sum()),
        "prob_loss": float(np.mean(W < 1)),
        "worst_5pct": float(np.quantile(W, 0.05) - 1),
        "expected_growth_rate": np.expm1(np.mean(logW) / horizon),
        "expected_return": np.mean(W) ** (1 / horizon) - 1,
        "volatility": np.std(logW) / np.sqrt(horizon),
    }


def portfolio_kelly(prices: pd.DataFrame, results: pd.DataFrame,
                    basket: list[str], rf: float,
                    max_weight: float = PORTFOLIO_MAX_WEIGHT,
                    max_leverage: float = PORTFOLIO_MAX_LEVERAGE,
                    shrink: float = COV_SHRINK,
                    horizon: float = DEFAULT_HORIZON,
                    n_scenarios: int = N_SCENARIOS,
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Kelly as the book defines it: choose the basket's weights to maximize the
    expected LOG of wealth at the horizon, over the joint outcomes of every
    position at once (Thorp's "joint properties" rule). Buy and hold for the
    horizon, long only, at most max_leverage invested; the rest earns the
    risk-free rate. Any mix that could wipe out the portfolio scores log(0)
    = -inf, so the result never risks ruin. Fractional Kelly = the full-Kelly
    bet scaled by c (half Kelly = bet half), then capped by your position
    limits. Returns (weights, summary).
    """
    ok, basket = _basket_filter(results, basket)
    corr, months = _correlation(prices, basket, shrink)
    R = simulate_outcomes(ok, basket, corr, horizon, n_scenarios)
    rf_h = (1 + rf) ** horizon - 1
    X = R - rf_h
    n = len(basket)

    def neg_obj(w):
        return -np.mean(_smooth_log(1 + rf_h + X @ w))

    def neg_grad(w):
        return -(X * _smooth_log_grad(1 + rf_h + X @ w)[:, None]).mean(axis=0)

    # Fractional Kelly c means "bet c times what Kelly says", then apply your
    # position limits. Solve for the Kelly bet u with limits scaled by 1/c,
    # and hold w = c * u. Uncapped this is exactly c * full Kelly; when a cap
    # binds, half Kelly is half the Kelly bet trimmed to the cap, not half of
    # an already-capped bet.
    weights, summary = {}, []
    for label, c in FRACTIONS.items():
        u, success = _solve(neg_obj, neg_grad, n, max_weight / c, max_leverage / c)
        w = c * u
        w[w < 1e-5] = 0.0
        weights[f"weight_{label}_kelly"] = w
        summary.append({"kelly_fraction": label, **basket_stats(w, X, rf_h, horizon),
                        "solver_ok": success})
    n_views = int((ok.loc[basket, "edge_source"] == "views").sum())
    log(f"Portfolio: {n_views} of {n} names sized on your views, the rest on CAPM; "
        f"{horizon:g}-year horizon, {n_scenarios:,} scenarios")

    out = pd.DataFrame({"ticker": basket, **weights})
    meta_cols = ["name", "sector", "universe", "edge_source", "p_bear", "bear",
                 "p_base", "base", "p_bull", "bull", "value", "price",
                 "margin_of_safety", "beta", "annual_volatility", "expected_return",
                 "full_kelly"]
    meta_cols = [c for c in meta_cols if c in ok.columns]
    out = out.merge(ok.loc[basket, meta_cols].reset_index(), on="ticker")
    out = out.rename(columns={"full_kelly": "standalone_full_kelly"})
    out = out.sort_values(["weight_full_kelly", "ticker"], ascending=[False, True])
    cash = {"ticker": "CASH", **{k: 1 - v.sum() for k, v in weights.items()}}
    out = pd.concat([out, pd.DataFrame([cash])], ignore_index=True)
    summ = pd.DataFrame(summary)
    summ.insert(1, "views_used", n_views)
    summ.insert(2, "basket_size", n)
    summ.insert(3, "months_used", months)
    summ.insert(4, "horizon_years", horizon)
    return out, summ


# ------------------------------------------------------- what if wrong ---
# Thorp (p. 105): misjudging the downside, or a Black Swan, means the computed
# Kelly bet is too big. The stress test makes that fragility visible: the same
# basket re-sized with every bear case a little worse, and your current
# weights scored as if that were the truth.
STRESS_ODDS = 0.10        # bear case 10 points more likely (taken from bull)
STRESS_LOSS = 1.5         # bear case 50% deeper


def stress_views(views: pd.DataFrame, odds_shift: float = 0.0,
                 loss_mult: float = 1.0) -> pd.DataFrame:
    """Every view's bear case made worse: odds up by odds_shift (taken from the
    bull case, then the base case), loss multiplied by loss_mult (max 100%)."""
    v = normalize_views(views).copy()
    take_bull = np.minimum(odds_shift, (v["p_bull"] - 0.01).clip(lower=0))
    take_base = np.minimum(odds_shift - take_bull, v["p_base"])
    v["p_bear"] = v["p_bear"] + take_bull + take_base
    v["p_base"] = v["p_base"] - take_base
    v["p_bull"] = 1 - v["p_bear"] - v["p_base"]
    v["bear"] = (v["bear"] * loss_mult).clip(upper=1.0)
    return v


def _verdict(mine: float, worst: float) -> str:
    if mine <= 1e-4:
        return "Not held"
    ratio = worst / mine
    return "Robust" if ratio >= 0.75 else "Sensitive" if ratio >= 0.40 else "Fragile"


def what_if_wrong(prices: pd.DataFrame, tickers: list[str], rf: float,
                  views: pd.DataFrame | None, weights: pd.DataFrame, fraction: str,
                  prices_now: dict[str, float] | None = None,
                  market_return: float = MARKET_RETURN,
                  horizon: float = DEFAULT_HORIZON,
                  max_weight: float = PORTFOLIO_MAX_WEIGHT,
                  max_leverage: float = PORTFOLIO_MAX_LEVERAGE,
                  shrink: float = COV_SHRINK,
                  n_scenarios: int = N_SCENARIOS,
                  neutral: bool = False,
                  ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """
    Returns (by_name, held) or None when no name has a view.
      by_name : your weight vs. what Kelly would bet under each stress, and a
                verdict (Robust / Sensitive / Fragile) for names with a view
      held    : your current weights, as sized, scored under each stress
    """
    if views is None or not len(views):
        return None
    col = f"weight_{fraction}_kelly"
    mine = weights[weights.ticker != "CASH"].set_index("ticker")[col]
    basket = list(mine.index)
    stresses = {
        f"Bear odds +{STRESS_ODDS * 100:.0f} pts": stress_views(views, STRESS_ODDS, 1.0),
        f"Bear loss {STRESS_LOSS:g}x deeper": stress_views(views, 0.0, STRESS_LOSS),
    }
    kw = dict(market_return=market_return, horizon=horizon, max_weight=max_weight,
              max_leverage=max_leverage, shrink=shrink, n_scenarios=n_scenarios,
              neutral=neutral)
    rf_h = (1 + rf) ** horizon - 1
    corr, _ = _correlation(prices, basket, shrink)

    by_name = pd.DataFrame({"ticker": basket, "mine": mine.values})
    held = []
    with captured_log():
        res0 = compute_kelly(prices, rf, market_return, views=views, horizon=horizon,
                             prices_now=prices_now, neutral=neutral).set_index("ticker")
        X0 = simulate_outcomes(res0, basket, corr, horizon, n_scenarios) - rf_h
        held.append({"scenario": "As you entered it",
                     **basket_stats(mine.values, X0, rf_h, horizon)})
        for name, sv in stresses.items():
            res, w, _ = size_basket(prices, tickers, rf, views=sv, prices_now=prices_now, **kw)
            by_name[name] = w.set_index("ticker")[col].reindex(basket).fillna(0).values
            ok = res.set_index("ticker")
            X = simulate_outcomes(ok, basket, corr, horizon, n_scenarios) - rf_h
            held.append({"scenario": name, **basket_stats(mine.values, X, rf_h, horizon)})

    viewed = set(normalize_views(views).index)
    stress_cols = list(stresses)
    by_name["edge"] = ["views" if t in viewed else "CAPM" for t in basket]
    by_name["verdict"] = [
        _verdict(r["mine"], min(r[c] for c in stress_cols)) if r["edge"] == "views" else ""
        for _, r in by_name.iterrows()]
    by_name = by_name[["ticker", "edge", "mine", *stress_cols, "verdict"]]
    return by_name, pd.DataFrame(held)


def what_if_message(by_name: pd.DataFrame, held: pd.DataFrame) -> str:
    """One plain sentence: the most fragile bet, and what being wrong costs."""
    stress = by_name.columns[3]
    v = by_name[(by_name.edge == "views") & (by_name.mine > 1e-4)].copy()
    lines = []
    if len(v):
        v["drop"] = 1 - v[stress] / v["mine"]
        worst = v.sort_values("drop", ascending=False).iloc[0]
        if worst["drop"] > 0.25:
            lines.append(f"If every bear case is {STRESS_ODDS * 100:.0f} points more likely, "
                         f"Kelly would cut {worst['ticker']} from {worst['mine']:.0%} to "
                         f"{worst[stress]:.0%}.")
        else:
            lines.append("Your bets hold up: being somewhat too optimistic on the bear "
                         "cases barely changes the sizing.")
    base, bad = held.iloc[0], held.iloc[1]
    lines.append(f"Held as sized, your chance of loss would go from {base['prob_loss']:.0%} "
                 f"to {bad['prob_loss']:.0%} and expected growth from "
                 f"{base['expected_growth_rate']:.1%} to {bad['expected_growth_rate']:.1%}/yr.")
    return " ".join(lines)


# -------------------------------------------------------------- journal ---
# Kelly is only as good as your probabilities. The journal records each view
# with the date and entry price; later you record which scenario actually
# played out (judge the thesis, not the price: a drop isn't a permanent loss).
# Calibration then shows whether your 60% calls come true about 60% of the time.
#
# Each entry also records the market's odds at the time (market check), when
# options were available: the chance of finishing below your bear price and
# above your bull price at market_expiry. Once enough calls resolve, the
# journal scores your odds and the market's on the same calls. If the market
# keeps scoring better, your edge may not be real (Munger, pp. 209-210: bet
# only on a mispriced bet). Older journals without these columns still load.
JOURNAL_COLS = ["date", "ticker", "horizon_years", "entry_price", "value",
                "p_bear", "bear", "p_base", "base", "p_bull", "bull",
                "notes", "outcome", "outcome_date",
                "market_bear", "market_bull", "market_expiry"]
MIN_RESOLVED = 10    # resolved calls before calibration verdicts mean anything
OUTCOMES = ["bear", "base", "bull"]


def normalize_journal(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=JOURNAL_COLS)
    j = df.copy()
    j.columns = [str(c).strip().lower() for c in j.columns]
    for c in JOURNAL_COLS:
        if c not in j.columns:
            j[c] = np.nan
    j["ticker"] = j["ticker"].astype(str).map(_yahoo_symbol).str.upper()
    j["date"] = pd.to_datetime(j["date"], errors="coerce").dt.date
    j["outcome_date"] = pd.to_datetime(j["outcome_date"], errors="coerce").dt.date
    j["outcome"] = (j["outcome"].astype(str).str.strip().str.lower()
                    .where(lambda s: s.isin(OUTCOMES), ""))
    num = ["horizon_years", "entry_price", "value", "p_bear", "bear", "p_base",
           "base", "p_bull", "bull", "market_bear", "market_bull"]
    j[num] = j[num].apply(pd.to_numeric, errors="coerce")
    j["market_expiry"] = j["market_expiry"].fillna("").astype(str).replace("nan", "")
    j[["market_bear", "market_bull"]] = j[["market_bear", "market_bull"]].round(6)
    j["notes"] = j["notes"].fillna("").astype(str).replace("nan", "")
    j[["p_bear", "bear", "p_base", "base", "p_bull", "bull"]] = (
        j[["p_bear", "bear", "p_base", "base", "p_bull", "bull"]].round(6))
    j[["entry_price", "value"]] = j[["entry_price", "value"]].round(4)
    return j[JOURNAL_COLS].dropna(subset=["date"]).reset_index(drop=True)


def load_journal(path: Path | str) -> pd.DataFrame:
    p = Path(path)
    return normalize_journal(pd.read_csv(p)) if p.exists() else normalize_journal(None)


def journal_entries(resolved_views: pd.DataFrame, horizon: float,
                    on: date | None = None,
                    market: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rows for today's views (output of resolve_views), ready to append.
    market: a market_check table; its odds are recorded beside yours."""
    on = on or date.today()
    v = resolved_views.reset_index()
    rows = pd.DataFrame({
        "date": on, "ticker": v["ticker"], "horizon_years": horizon,
        "entry_price": v.get("price"), "value": v.get("value"),
        **{c: v[c] for c in ["p_bear", "bear", "p_base", "base", "p_bull", "bull"]},
        "notes": v.get("notes", ""), "outcome": "", "outcome_date": pd.NaT,
    })
    if market is not None and len(market):
        m = market.set_index("ticker")
        rows["market_bear"] = rows["ticker"].map(m["market_bear"])
        rows["market_bull"] = rows["ticker"].map(m["market_bull"])
        rows["market_expiry"] = rows["ticker"].map(m["expiry"]).fillna("")
    return normalize_journal(rows)


def _brier(P: np.ndarray, outcomes: pd.Series) -> float:
    """Mean Brier score over bear/base/bull (0 = perfect, 2 = certain and wrong)."""
    Y = np.stack([(outcomes == o).to_numpy(float) for o in OUTCOMES], axis=1)
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def market_vs_you(resolved: pd.DataFrame) -> dict:
    """
    Your odds and the market's, scored on the same resolved calls (those that
    have market odds). The market's base odds are what's left after its bear
    and bull odds. Caveat, stated wherever this is shown: the market's odds
    are about the price at the option expiry and are risk-neutral, while your
    outcomes judge the thesis, so the two answer slightly different questions.
    """
    both = resolved.dropna(subset=["market_bear", "market_bull"])
    out = {"market_calls": len(both)}
    if not len(both):
        return out
    mb, mu = both["market_bear"].to_numpy(), both["market_bull"].to_numpy()
    P_mkt = np.stack([mb, np.clip(1 - mb - mu, 0, 1), mu], axis=1)
    P_you = both[["p_bear", "p_base", "p_bull"]].fillna(0).to_numpy()
    out["your_brier_vs"] = _brier(P_you, both["outcome"])
    out["market_brier"] = _brier(P_mkt, both["outcome"])
    if len(both) < MIN_RESOLVED:
        out["market_verdict"] = (f"{len(both)} resolved calls have the market's odds beside "
                                 f"yours; the comparison needs {MIN_RESOLVED}+.")
    elif out["your_brier_vs"] < out["market_brier"] - 0.02:
        out["market_verdict"] = (
            f"Your odds beat the market's on {len(both)} calls (Brier "
            f"{out['your_brier_vs']:.2f} vs {out['market_brier']:.2f}). Evidence your edge "
            f"is real.")
    elif out["your_brier_vs"] > out["market_brier"] + 0.02:
        out["market_verdict"] = (
            f"The market's odds beat yours on {len(both)} calls (Brier "
            f"{out['market_brier']:.2f} vs {out['your_brier_vs']:.2f}). Your edge may not be "
            f"real: bet a smaller fraction until that changes.")
    else:
        out["market_verdict"] = (
            f"Your odds and the market's score about the same on {len(both)} calls (Brier "
            f"{out['your_brier_vs']:.2f} vs {out['market_brier']:.2f}). No clear edge yet.")
    return out


def append_journal(journal: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Add entries; re-logging the same ticker on the same day replaces it."""
    j = pd.concat([normalize_journal(journal), normalize_journal(new)], ignore_index=True)
    return j.drop_duplicates(["date", "ticker"], keep="last").reset_index(drop=True)


def journal_report(journal: pd.DataFrame, prices_now: dict[str, float] | None = None,
                   today: date | None = None) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    Returns (entries with status and return since entry, summary dict,
    calibration table by the bull probability you gave).

    status: 'resolved' (outcome recorded), 'due' (horizon has passed, record
    the outcome), or 'open'.
    """
    today = today or date.today()
    j = normalize_journal(journal).copy()
    prices_now = prices_now or {}
    due_date = [d + pd.DateOffset(days=int(round(h * 365.25)))
                if pd.notna(h) else pd.NaT
                for d, h in zip(pd.to_datetime(j["date"]), j["horizon_years"])]
    j["horizon_ends"] = pd.to_datetime(due_date).date if len(j) else []
    j["status"] = np.where(j["outcome"] != "", "resolved",
                           np.where(pd.to_datetime(j["horizon_ends"]) <= pd.Timestamp(today),
                                    "due", "open"))
    j["price_now"] = [prices_now.get(t, np.nan) for t in j["ticker"]]
    j["return_since_entry"] = j["price_now"] / j["entry_price"] - 1
    j["expected_return"] = (j["p_bear"] * -j["bear"] + j["p_base"].fillna(0) * j["base"].fillna(0)
                            + j["p_bull"] * j["bull"])

    res = j[j["status"] == "resolved"]
    summary = {"entries": len(j), "resolved": len(res),
               "due": int((j["status"] == "due").sum()),
               "open": int((j["status"] == "open").sum())}
    if len(res):
        P = res[["p_bear", "p_base", "p_bull"]].fillna(0).to_numpy()
        Y = np.stack([(res["outcome"] == o).to_numpy(float) for o in OUTCOMES], axis=1)
        summary["brier"] = _brier(P, res["outcome"])
        summary["avg_prob_of_actual"] = float(np.mean(np.sum(P * Y, axis=1)))
        summary.update(market_vs_you(res))
        for o, col in zip(OUTCOMES, ["p_bear", "p_base", "p_bull"]):
            summary[f"predicted_{o}"] = float(res[col].fillna(0).mean())
            summary[f"actual_{o}"] = float((res["outcome"] == o).mean())
        gap = summary["predicted_bull"] - summary["actual_bull"]
        summary["verdict"] = (
            f"Too few resolved calls to judge calibration yet (aim for {MIN_RESOLVED}+)."
            if len(res) < MIN_RESOLVED else
            f"Your bull calls run hot: you predicted {summary['predicted_bull']:.0%}, "
            f"{summary['actual_bull']:.0%} came true. Lower your odds or bet a smaller fraction."
            if gap > 0.10 else
            f"You're underconfident: {summary['actual_bull']:.0%} of bull calls came true "
            f"vs {summary['predicted_bull']:.0%} predicted."
            if gap < -0.10 else
            "Well calibrated so far: your bull odds match how often they came true.")
    bins = pd.cut(res["p_bull"], [0, 0.4, 0.6, 0.8, 1.0001],
                  labels=["<40%", "40-60%", "60-80%", "80%+"], right=False)
    calib = (res.assign(bucket=bins, hit=(res["outcome"] == "bull").astype(float))
             .groupby("bucket", observed=True)
             .agg(calls=("hit", "size"), predicted_bull=("p_bull", "mean"),
                  actual_bull=("hit", "mean"))
             .reset_index()) if len(res) else pd.DataFrame(
        columns=["bucket", "calls", "predicted_bull", "actual_bull"])
    return j, summary, calib


# --------------------------------------------------------- one workflow ---
# The phone app and the command line both run these, so they always show the
# same numbers, in the same order, with the same words.
@contextmanager
def captured_log():
    """Collect engine messages (dropped names, view notes) instead of printing."""
    global log
    msgs: list[str] = []
    original, log = log, msgs.append
    try:
        yield msgs
    finally:
        log = original


FRACTION_NAMES = {"full": "Full", "half": "Half", "quarter": "Quarter", "eighth": "Eighth"}


def size_basket(prices: pd.DataFrame, tickers: list[str], rf: float,
                views: pd.DataFrame | None = None,
                prices_now: dict[str, float] | None = None,
                meta: pd.DataFrame | None = None,
                market_return: float = MARKET_RETURN,
                horizon: float = DEFAULT_HORIZON,
                max_weight: float = PORTFOLIO_MAX_WEIGHT,
                max_leverage: float = PORTFOLIO_MAX_LEVERAGE,
                shrink: float = COV_SHRINK,
                n_scenarios: int = N_SCENARIOS,
                neutral: bool = False,
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per-name stats, then joint Kelly weights. Returns (results, weights, summary)."""
    if neutral:
        log("Neutral mode: each view keeps its shape but is scaled to its CAPM return, "
            "so there's no edge. A test of your inputs, not a sizing.")
    results = compute_kelly(prices, rf, market_return, views=views,
                            horizon=horizon, prices_now=prices_now, neutral=neutral)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        results = pd.concat([results, pd.DataFrame({"ticker": missing,
                                                    "status": "no price data"})])
    if meta is not None and len(meta):
        results = meta.merge(results, on="ticker", how="right")
    weights, summary = portfolio_kelly(prices, results, tickers, rf,
                                       max_weight=max_weight, max_leverage=max_leverage,
                                       shrink=shrink, horizon=horizon,
                                       n_scenarios=n_scenarios)
    return results, weights, summary


def basket_message(summary: pd.DataFrame) -> str:
    s = summary.iloc[0]
    n, total, h = int(s["views_used"]), int(s["basket_size"]), s["horizon_years"]
    if n:
        return (f"Sized on your research for {n} of {total} names over a {h:g}-year "
                f"hold. The rest use CAPM.")
    return ("No views yet, so every name uses CAPM, which assumes no edge and sizes by "
            "risk alone. Add your research to size on conviction.")


def headline(summary: pd.DataFrame, fraction: str) -> list[tuple[str, str, str]]:
    """(label, value, explanation), in the book's order: permanent-loss risk first."""
    s = summary.set_index("kelly_fraction").loc[fraction]
    hz = f"{s['horizon_years']:g}y"
    return [
        (f"Chance of loss ({hz})", f"{s['prob_loss']:.0%}",
         "Chance the basket ends the horizon below where it started"),
        (f"Worst case, 1-in-20 ({hz})", f"{s['worst_5pct']:.0%}",
         "5th-percentile outcome over the horizon"),
        ("Exp. growth", f"{s['expected_growth_rate']:.1%}/yr",
         "Expected compound growth, what Kelly maximizes"),
        ("Invested", f"{s['invested']:.0%}", ""),
        ("Cash", f"{s['cash']:.0%}", "Earns the risk-free rate"),
        ("Price swings", f"{s['volatility']:.0%}/yr",
         "How bumpy the ride is. Not risk in the book's sense, but what you sit through"),
    ]


def allocation_table(weights: pd.DataFrame, fraction: str, amount: float) -> pd.DataFrame:
    """One row per name plus CASH: weight, dollars, edge, margin of safety,
    solo Kelly, price swings. Fractions are decimals."""
    w = weights.copy()
    out = pd.DataFrame({
        "ticker": w["ticker"],
        "weight": w[f"weight_{fraction}_kelly"],
        "dollars": w[f"weight_{fraction}_kelly"] * amount,
        "edge": w.get("edge_source"),
        "margin_of_safety": w.get("margin_of_safety"),
        "solo_kelly": w.get("standalone_full_kelly"),
        "swing": w.get("annual_volatility"),
    })
    names = out[out.ticker != "CASH"].sort_values(["weight", "ticker"], ascending=[False, True])
    out = pd.concat([names, out[out.ticker == "CASH"]], ignore_index=True)
    if out["margin_of_safety"].isna().all():
        out = out.drop(columns="margin_of_safety")
    return out


def fractions_table(weights: pd.DataFrame, summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Weights at every fraction, and the headline stats at every fraction."""
    wide = weights[["ticker"] + [f"weight_{f}_kelly" for f in FRACTIONS]].copy()
    wide.columns = ["Ticker"] + [FRACTION_NAMES[f] for f in FRACTIONS]
    hz = f"{summary['horizon_years'].iloc[0]:g}y"
    stats = (summary.set_index("kelly_fraction")
             [["prob_loss", "worst_5pct", "expected_growth_rate", "invested",
               "expected_return", "volatility"]]
             .rename(index=FRACTION_NAMES, columns={
                 "prob_loss": f"Chance of loss ({hz})",
                 "worst_5pct": f"Worst 1-in-20 ({hz})",
                 "expected_growth_rate": "Exp. growth/yr", "invested": "Invested",
                 "expected_return": "Exp. return/yr", "volatility": "Price swings/yr"})
             .rename_axis("Kelly"))
    return wide.set_index("Ticker"), stats


def view_notes(logs: list[str]) -> list[str]:
    """Engine notes about your views (re-anchored, ignored, wrong horizon)."""
    return [m.split(": ", 1)[1] for m in logs if m.startswith("Views:")]


def run_note(summary: pd.DataFrame, rf: float, rf_source: str, logs: list[str],
             include_views: bool = True) -> str:
    keep = ("Portfolio:", "Views:") if include_views else ("Portfolio:",)
    notes = [m.split(": ", 1)[1] for m in logs if m.startswith(keep)]
    notes += [m for m in logs if m.startswith("Neutral mode")]
    return (f"Risk-free {rf:.2%} ({rf_source}) · {int(summary['months_used'].iloc[0])} "
            f"months of price history for correlations"
            + (f" · {'; '.join(notes)}" if notes else ""))


# ---------------------------------------------------------- screen (opt-in) ---
def screen_sp500(rf: float, market_return: float = MARKET_RETURN,
                 horizon: float = DEFAULT_HORIZON, views: pd.DataFrame | None = None,
                 extra: list[str] | None = None, limit: int | None = None) -> pd.DataFrame:
    """
    The no-edge baseline: solo Kelly for every S&P 500 name under CAPM (and
    your views where you have them). CAPM sizes by risk alone, so this ranks
    risk structure, not opportunities. The book's ideas come from research.
    """
    meta = get_sp500_constituents()
    meta["universe"] = "S&P 500"
    tickers = meta["ticker"].tolist()[:limit] if limit else meta["ticker"].tolist()
    views = views if views is not None else normalize_views(None)
    add = [t for t in dict.fromkeys([_yahoo_symbol(t) for t in (extra or [])]
                                    + list(views.index)) if t not in tickers]
    tickers += add
    non_sp = [t for t in tickers if t not in set(meta["ticker"])]
    if non_sp:
        extra_meta = lookup_extra_meta(non_sp)
        extra_meta["universe"] = "Extra"
        meta = pd.concat([meta, extra_meta], ignore_index=True)
    prices = download_monthly_closes(tickers + [BENCHMARK], VARIANCE_YEARS)
    if prices.empty or BENCHMARK not in prices.columns:
        raise RuntimeError("no price data returned")
    prices_now = download_latest_prices(list(views.index)) if len(views) else {}
    res = compute_kelly(prices, rf, market_return, views=views, horizon=horizon,
                        prices_now=prices_now)
    missing = sorted(set(tickers) - set(prices.columns))
    if missing:
        res = pd.concat([res, pd.DataFrame({"ticker": missing, "status": "no price data"})])
    res = meta.merge(res, on="ticker", how="right")
    res.insert(0, "run_date", date.today().isoformat())
    res["risk_free_rate"] = rf
    res["market_return"] = market_return
    return res.sort_values("full_kelly", ascending=False, na_position="last")


# ------------------------------------------------------------------ files ---
def default_out_dir() -> Path:
    desktop = Path.home() / "Desktop"
    return desktop if desktop.exists() else Path.cwd()


def safe_to_csv(df: pd.DataFrame, path: Path) -> Path:
    """Write CSV; if the file is locked (open in Excel), use a timestamped name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return path
    except PermissionError:
        alt = path.with_name(f"{path.stem}_{datetime.now():%H%M%S}{path.suffix}")
        log(f"{path.name} is open/locked (Excel?). Writing {alt.name} instead.")
        df.to_csv(alt, index=False)
        return alt


# Yahoo's sector names -> the GICS names the S&P 500 list uses, so a basket
# mixing S&P and non-S&P names reads one way. Sectors already named the same
# (Energy, Industrials, Utilities, Real Estate, Communication Services) pass through.
YAHOO_TO_GICS = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Basic Materials": "Materials",
}


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
        rows.append({"ticker": t, "name": name, "sector": YAHOO_TO_GICS.get(sector, sector)})
    return pd.DataFrame(rows, columns=["ticker", "name", "sector"])


def basket_meta(tickers: list[str]) -> pd.DataFrame:
    """Name/sector for a basket: the S&P 500 list, plus a Yahoo lookup for the rest."""
    try:
        with captured_log():                       # keep the source chatter off the CLI
            meta = get_sp500_constituents()[["ticker", "name", "sector"]]
    except Exception:
        meta = pd.DataFrame(columns=["ticker", "name", "sector"])
    meta = meta[meta["ticker"].isin(tickers)]
    missing = [t for t in tickers if t not in set(meta["ticker"])]
    if missing:
        meta = pd.concat([meta, lookup_extra_meta(missing)], ignore_index=True)
    return meta.reset_index(drop=True)


# -------------------------------------------------------------------- CLI ---
def _fmt_table(df: pd.DataFrame, pct: list[str], money: list[str] = ()) -> str:
    f = {c: (lambda x: "" if pd.isna(x) else f"{x:.2f}") for c in ["beta"] if c in df.columns}
    f.update({c: (lambda x: "" if pd.isna(x) else f"{x:.1%}") for c in pct if c in df.columns})
    f.update({c: (lambda x: "" if pd.isna(x) else f"${x:,.0f}") for c in money if c in df.columns})
    return df.to_string(index=False, formatters=f, na_rep="")


def print_journal(journal: pd.DataFrame) -> None:
    now = download_latest_prices(sorted(set(journal["ticker"])))
    entries, s, calib = journal_report(journal, now)
    print(f"\nConviction journal: {s['entries']} calls · {s['resolved']} resolved · "
          f"{s['due']} due for an outcome · {s['open']} open")
    if s["resolved"]:
        print(f"  {s['verdict']}")
        print(f"  Brier score {s['brier']:.2f} (0 = perfect) · on average you gave "
              f"{s['avg_prob_of_actual']:.0%} odds to what actually happened")
        if s.get("market_verdict"):
            print(f"  You vs. the market: {s['market_verdict']}")
        print("  " + _fmt_table(calib.rename(columns={
            "bucket": "Bull odds you gave", "calls": "Calls",
            "predicted_bull": "Predicted", "actual_bull": "Came true"}),
            pct=["Predicted", "Came true"]).replace("\n", "\n  "))
    due = entries[entries["status"] == "due"]
    if len(due):
        print("  Due: fill bear/base/bull in the outcome column for "
              + ", ".join(f"{t} (logged {d})" for t, d in zip(due["ticker"], due["date"])))


EXAMPLES = """
examples:
  python kelly.py NVDA ASML KO MSFT VOO                 size this basket (half Kelly, $10,000)
  python kelly.py NVDA ASML KO --fraction quarter --amount 25000
  python kelly.py                                       size the names in views.csv
  python kelly.py NVDA ASML KO --log                    ...and log today's views to journal.csv
  python kelly.py --report                              conviction journal only
  python kelly.py --screen --extra TSM ASML             S&P 500 no-edge baseline to CSV

views.csv and journal.csv in the current folder are used automatically,
the same files the phone app saves and loads.
"""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="kelly.py",
        description="Kelly Sizer: Kelly position sizing after Concentrated Investing "
                    "(Benello, van Biema & Carlisle).",
        epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tickers", nargs="*", help="Your basket, e.g. NVDA ASML KO MSFT VOO")
    p.add_argument("--portfolio", nargs="+", help=argparse.SUPPRESS)  # old spelling

    s = p.add_argument_group("sizing (same settings as the app)")
    s.add_argument("--fraction", choices=list(FRACTIONS), default="half",
                   help="Kelly fraction to show (default half; all four are saved)")
    s.add_argument("--amount", type=float, default=10_000,
                   help="Dollars to allocate (default 10,000)")
    s.add_argument("--horizon", type=float, default=DEFAULT_HORIZON,
                   help=f"Buy-and-hold years (default {DEFAULT_HORIZON:g})")
    s.add_argument("--max-weight", type=float, default=PORTFOLIO_MAX_WEIGHT,
                   help=f"Max per name (default {PORTFOLIO_MAX_WEIGHT:g}; "
                        f"book ceiling {BOOK_MAX_WEIGHT:g})")
    s.add_argument("--max-invested", "--max-leverage", dest="max_invested", type=float,
                   default=PORTFOLIO_MAX_LEVERAGE,
                   help="Max total invested; 1.0 = no margin (default)")
    s.add_argument("--shrink", type=float, default=COV_SHRINK,
                   help=f"Correlation shrinkage toward independence (default {COV_SHRINK:g})")
    s.add_argument("--market-return", type=float, default=MARKET_RETURN,
                   help=f"Long-run market return for CAPM names (default {MARKET_RETURN})")
    s.add_argument("--book-mode", action="store_true",
                   help="Cash earns 0%%, as in the book's formula f = (bp - q)/b "
                        "(default: the Treasury yield matched to the horizon)")
    s.add_argument("--street", action="store_true",
                   help="Names without a view start from analysts' low/mean/high price "
                        "targets (a labeled consensus starter with placeholder odds; saved "
                        "to street_views_DATE.csv to edit into your views)")
    s.add_argument("--market", action="store_true",
                   help="After sizing, show the option-implied odds beside your own odds "
                        "for each view (a reference; never used in the sizing)")
    s.add_argument("--neutral", action="store_true",
                   help="Test mode: scale each view to its CAPM return, keeping its shape "
                        "(no edge). Sizes should land near the no-views baseline")
    s.add_argument("--scenarios", type=int, default=N_SCENARIOS, help=argparse.SUPPRESS)

    r = p.add_argument_group("your research")
    r.add_argument("--views", type=Path, default=Path("views.csv"),
                   help="Views CSV (default views.csv in this folder, if present)")
    r.add_argument("--journal", type=Path, default=Path("journal.csv"),
                   help="Journal CSV (default journal.csv in this folder)")
    r.add_argument("--log", action="store_true", help="Log today's views to the journal")
    r.add_argument("--report", action="store_true", help="Show the journal report only")

    o = p.add_argument_group("output")
    o.add_argument("--out-dir", type=Path, default=None,
                   help="Folder for result CSVs (default Desktop, else this folder)")
    o.add_argument("--no-save", action="store_true", help="Print only; don't write CSVs")

    b = p.add_argument_group("no-edge baseline (opt-in)")
    b.add_argument("--screen", action="store_true",
                   help="Solo Kelly for every S&P 500 name under CAPM, saved to CSV")
    b.add_argument("--extra", nargs="+", default=[], help="Add non-S&P names to the screen")
    b.add_argument("--limit", type=int, help="Only the first N constituents (quick test)")
    args = p.parse_args(argv)

    tickers = [_yahoo_symbol(t).upper() for t in (args.tickers or args.portfolio or [])]
    views = load_views(args.views) if args.views.exists() else normalize_views(None)
    journal = load_journal(args.journal)
    out_dir = args.out_dir or default_out_dir()
    stamp = f"{date.today():%Y-%m-%d}"

    if args.report:
        if not len(journal):
            print(f"No journal yet at {args.journal}. Log views with --log.")
            return 0
        print_journal(journal)
        return 0
    if not tickers and not args.screen:
        tickers = list(views.index)
        if not tickers:
            p.print_help()
            print("\nGive a basket, e.g.:  python kelly.py NVDA ASML KO MSFT VOO")
            return 2
    if args.max_weight > BOOK_MAX_WEIGHT:
        print(f"Note: --max-weight {args.max_weight:.0%} is above the book's 40% ceiling "
              f"(Buffett's largest position, American Express). Treat this as a what-if.")
    if len(views):
        print(f"Using your views from {args.views}: {', '.join(views.index)}")

    rf, rf_source = risk_free_for(args.horizon, args.book_mode)

    # ---- the no-edge baseline (opt-in)
    if args.screen:
        print("Screening the S&P 500 under CAPM, the no-edge baseline. This ranks risk, "
              "not opportunities.")
        res = screen_sp500(rf, args.market_return, args.horizon, views,
                           args.extra, args.limit)
        ok = res[res["status"] == "ok"]
        cols = [c for c in ["ticker", "edge_source", "sector", "beta",
                            "annual_volatility", "full_kelly", "half_kelly"] if c in ok]
        print(f"\nTop 20 of {len(ok)} by solo Kelly ({args.horizon:g}-year hold):\n"
              + _fmt_table(ok[cols].head(20), pct=["annual_volatility", "full_kelly",
                                                   "half_kelly"]))
        if not args.no_save:
            print(f"Saved: {safe_to_csv(res, out_dir / f'kelly_screen_{stamp}.csv')}")
        if not tickers:
            return 0

    # ---- size the basket
    if len(tickers) < 2:
        print("Give at least 2 tickers.")
        return 2
    prices = download_monthly_closes(sorted(set(tickers)) + [BENCHMARK], VARIANCE_YEARS)
    if prices.empty or BENCHMARK not in prices.columns:
        print("ERROR: no price data came back from Yahoo. Try again in a minute.")
        return 1
    if args.street:
        missing = [t for t in tickers if t not in views.index and t != BENCHMARK]
        starter, notes = street_views(street_targets(missing)) if missing else (
            normalize_views(None), [])
        print(f"\nStreet starter for {len(starter)} of {len(missing)} names without a view "
              f"(consensus 12-month targets, placeholder odds; not your edge):")
        for n in notes:
            print("  " + n)
        if len(starter):
            views = pd.concat([views, starter])
            if not args.no_save:
                print(f"  Saved: {safe_to_csv(starter.reset_index(), out_dir / f'street_views_{stamp}.csv')}"
                      " (edit it, then merge it into your views)")
    in_basket = views.loc[[t for t in tickers if t in views.index]]
    prices_now = download_latest_prices(list(in_basket.index)) if len(in_basket) else {}
    meta = basket_meta(tickers)
    try:
        with captured_log() as logs:
            results, weights, summary = size_basket(
                prices, tickers, rf, views=in_basket if len(in_basket) else None,
                prices_now=prices_now, meta=meta, market_return=args.market_return,
                horizon=args.horizon, max_weight=args.max_weight,
                max_leverage=args.max_invested, shrink=args.shrink,
                n_scenarios=args.scenarios, neutral=args.neutral)
    except ValueError as e:
        print(f"Can't size this basket: {e}")
        return 2

    frac = args.fraction
    print(f"\nKelly Sizer · {FRACTION_NAMES[frac]} Kelly · ${args.amount:,.0f} · "
          f"{args.horizon:g}-year hold")
    print(basket_message(summary))
    stats = headline(summary, frac)
    for row in (stats[:3], stats[3:]):
        print("  " + "   ".join(f"{label}: {value}" for label, value, _ in row))
    alloc = allocation_table(weights, frac, args.amount).rename(columns={
        "ticker": "Ticker", "weight": "Weight", "dollars": "$", "edge": "Edge",
        "margin_of_safety": "MoS", "solo_kelly": "Solo Kelly", "swing": "Swing"})
    print("\n" + _fmt_table(alloc, pct=["Weight", "MoS", "Solo Kelly", "Swing"], money=["$"]))
    wide, fstats = fractions_table(weights, summary)
    wiw = what_if_wrong(prices, tickers, rf, in_basket if len(in_basket) else None,
                        weights, frac, prices_now=prices_now,
                        market_return=args.market_return, horizon=args.horizon,
                        max_weight=args.max_weight, max_leverage=args.max_invested,
                        shrink=args.shrink, n_scenarios=args.scenarios,
                        neutral=args.neutral)
    if wiw is not None:
        by_name, held = wiw
        print("\nWhat if I'm wrong?")
        print(what_if_message(by_name, held))
        t = by_name.rename(columns={"ticker": "Ticker", "edge": "Edge",
                                    "mine": f"{FRACTION_NAMES[frac]} Kelly",
                                    "verdict": "Verdict"})
        print(_fmt_table(t, pct=list(t.columns[2:-1])))
        h = held[["scenario", "prob_loss", "worst_5pct", "expected_growth_rate"]].rename(
            columns={"scenario": "Your basket if the truth is", "prob_loss": "Chance of loss",
                     "worst_5pct": "Worst 1-in-20", "expected_growth_rate": "Exp. growth/yr"})
        print(_fmt_table(h, pct=list(h.columns[1:])))
    if args.market and len(in_basket):
        check = market_check(in_basket, prices_now, args.horizon, rf)
        print("\nMarket check: your odds vs. option-implied odds (risk-neutral; a reference, "
              "never used in the sizing)")
        m = check.rename(columns={"ticker": "Ticker", "your_bear": "Your bear",
                                  "market_bear": "Market", "bear_price": "Bear $",
                                  "your_bull": "Your bull", "market_bull": "Market ",
                                  "bull_price": "Bull $", "expiry": "Expiry", "note": "Note"})
        print(_fmt_table(m.drop(columns=["years"]),
                         pct=["Your bear", "Market", "Your bull", "Market "],
                         money=["Bear $", "Bull $"]))
        for gap in market_gaps(check):
            print("  " + gap)
    print("\nAll fractions:\n" + _fmt_table(fstats.reset_index(),
                                            pct=[c for c in fstats.columns]))
    print(run_note(summary, rf, rf_source, logs))
    if not summary["solver_ok"].all():
        print("WARNING: optimizer did not fully converge for some fractions.")

    if not args.no_save:
        w_path = safe_to_csv(weights, out_dir / f"kelly_portfolio_{stamp}.csv")
        s_path = safe_to_csv(summary, out_dir / f"kelly_portfolio_summary_{stamp}.csv")
        print(f"Saved: {w_path}\nSaved: {s_path}")

    # ---- conviction journal
    if args.log:
        if not len(in_basket):
            print("Nothing to log: no names in this basket have views.")
        else:
            last = prices.ffill().iloc[-1].to_dict()
            resolved = resolve_views(in_basket, {**last, **prices_now})
            market = market_check(in_basket, {**last, **prices_now}, args.horizon, rf)
            journal = append_journal(journal, journal_entries(resolved, args.horizon,
                                                              market=market))
            safe_to_csv(journal, args.journal)
            print(f"Logged {len(resolved)} views to {args.journal}")
    if len(journal):
        print_journal(journal)
    return 0


if __name__ == "__main__":
    sys.exit(main())
