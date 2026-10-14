"""
Kelly Portfolio Sizer — phone-friendly Streamlit front end for kelly_sp500.py.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud -> point at this repo, main file app.py
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import kelly_sp500 as ks

st.set_page_config(page_title="Kelly Sizer", page_icon="📐", layout="centered")

FRACTIONS = {"Full": "full", "Half": "half", "Quarter": "quarter", "Eighth": "eighth"}
DEFAULT_TICKERS = ["NVDA", "ASML", "AVGO", "MSFT", "VOO"]

# Backtest chart
TR_BENCH = "^SP500TR"     # S&P 500 with dividends reinvested (from 1988)
CASH_PROXY = "^IRX"       # 13-week T-bill yield, annual %
PORT, BENCH = "Your portfolio", "S&P 500"
TEAL = "#379A8B"          # also primaryColor in .streamlit/config.toml
ORANGE = "#eb6834"        # checked against TEAL for colorblind separation


# ------------------------------------------------------------- cached data ---
@st.cache_data(ttl=6 * 3600, show_spinner=False)
def risk_free() -> tuple[float, str]:
    return ks.get_risk_free_rate()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def sp500_list() -> pd.DataFrame:
    try:
        return ks.get_sp500_constituents()
    except Exception:
        return pd.DataFrame(columns=["ticker", "name", "sector"])


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def prices_for(tickers: tuple[str, ...]) -> pd.DataFrame:
    return ks.download_monthly_closes(list(tickers) + [ks.BENCHMARK], ks.VARIANCE_YEARS)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def full_history(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Monthly closes over each ticker's full history, for the backtest.
    Raises on failure so Streamlit doesn't cache an empty result."""
    import yfinance as yf

    raw = yf.download(list(tickers) + [TR_BENCH, ks.BENCHMARK, CASH_PROXY],
                      period="max", interval="1mo", auto_adjust=True,
                      group_by="column", threads=True, progress=False)
    close = raw["Close"].dropna(how="all")
    if close.empty:
        raise RuntimeError("no price history returned")
    this_month = pd.Timestamp.today().to_period("M").to_timestamp()
    return close[close.index < this_month]  # drop the partial current month


def backtest(hist: pd.DataFrame, held: pd.Series, rf: float) -> pd.DataFrame:
    """Growth of $1 in today's weights (rebalanced monthly) vs the S&P 500,
    from the first month every held name and the benchmark have data."""
    has = lambda c: c in hist.columns and hist[c].notna().any()
    missing = [t for t in held.index if not has(t)]
    if missing:
        raise RuntimeError(f"no history for {', '.join(missing)}")
    bench = TR_BENCH if has(TR_BENCH) else ks.BENCHMARK

    rets = hist[list(held.index) + [bench]].pct_change(fill_method=None)
    if has(CASH_PROXY):  # last month's T-bill yield, earned this month
        cash = (hist[CASH_PROXY].ffill() / 100 / 12).shift(1)
    else:
        cash = pd.Series(rf / 12, index=hist.index)
    df = pd.concat([rets, cash.rename("_cash")], axis=1).dropna()

    port = df[held.index] @ held.values + df["_cash"] * (1 - held.sum())
    growth = (1 + pd.DataFrame({PORT: port, BENCH: df[bench]})).cumprod()
    start = hist.index[hist.index.get_loc(df.index[0]) - 1]
    return pd.concat([pd.DataFrame({PORT: [1.0], BENCH: [1.0]}, index=[start]), growth])


def cagr(s: pd.Series) -> float:
    return (s.iloc[-1] / s.iloc[0]) ** (12 / (len(s) - 1)) - 1


def max_drawdown(s: pd.Series) -> float:
    return (s / s.cummax() - 1).min()


def doubling_ticks(start: float, lo: float, hi: float) -> list[float]:
    """Gridlines at each doubling of the starting value, evenly spaced on a log axis."""
    ks_ = range(int(np.floor(np.log2(lo / start))), int(np.ceil(np.log2(hi / start))) + 1)
    ticks = [start * 2 ** k for k in ks_ if lo <= start * 2 ** k <= hi]
    return ticks if len(ticks) >= 3 else alt.Undefined  # narrow range: let Vega pick


def growth_chart(values: pd.DataFrame, log: bool = False) -> alt.Chart:
    """Two lines (linear or log y), hover crosshair with both values, end labels."""
    ink = "#FAFAFA" if st.context.theme.type == "dark" else "#31333F"
    long = values.rename_axis("date").reset_index().melt(
        "date", var_name="series", value_name="value")
    color = alt.Color("series:N", title=None, legend=alt.Legend(orient="top"),
                      scale=alt.Scale(domain=[PORT, BENCH], range=[TEAL, ORANGE]))
    x = alt.X("date:T", title=None)
    if log:
        lo, hi = values.min().min(), values.max().max()
        scale = alt.Scale(type="log", nice=False, domain=[lo / 1.05, hi * 1.05])
        axis = alt.Axis(format="$.2~s", values=doubling_ticks(values.iloc[0, 0], lo, hi))
    else:
        scale = alt.Scale(zero=True)
        axis = alt.Axis(format="$.2~s", tickCount=5)
    y = alt.Y("value:Q", title=None, scale=scale, axis=axis)

    lines = alt.Chart(long).mark_line(strokeWidth=2).encode(x, y, color)

    hover = alt.selection_point(nearest=True, on="pointerover", fields=["date"],
                                empty=False, clear="pointerout")
    rule = (
        alt.Chart(long)
        .transform_pivot("series", value="value", groupby=["date"])
        .mark_rule(color="#8a8986", strokeWidth=12)  # wide, invisible hit target
        .encode(
            x,
            opacity=alt.condition(hover, alt.value(0.25), alt.value(0)),
            tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"),
                     alt.Tooltip(field=PORT, type="quantitative", format="$,.0f"),
                     alt.Tooltip(field=BENCH, type="quantitative", format="$,.0f")],
        )
        .add_params(hover)
    )
    dots = (alt.Chart(long).mark_point(filled=True, size=70)
            .encode(x, y, color, opacity=alt.condition(hover, alt.value(1), alt.value(0))))

    last = long[long["date"] == long["date"].max()].copy()
    last["dy"] = [-12 if v == last["value"].max() else 14 for v in last["value"]]
    labels = alt.layer(*[
        alt.Chart(last[last["series"] == s]).mark_text(
            align="right", dy=row_dy, fontSize=12, color=ink).encode(x, y, text="series:N")
        for s, row_dy in zip(last["series"], last["dy"])
    ])
    return alt.layer(lines, rule, dots, labels).properties(height=320)


def capture_logs():
    """Collect the engine's console messages so we can show them in the UI."""
    msgs: list[str] = []
    ks.log = msgs.append
    return msgs


# --------------------------------------------------------------------- UI ---
st.title("📐 Kelly Sizer")
st.caption("Joint Kelly weights for a basket, with correlations and position caps.")

meta = sp500_list()
options = sorted(set(meta["ticker"]) | set(DEFAULT_TICKERS))

with st.form("inputs"):
    picks = st.multiselect(
        "Tickers (type any symbol, e.g. TSM, ASML, VOO)",
        options=options,
        default=DEFAULT_TICKERS,
        max_selections=15,
        accept_new_options=True,
    )
    c1, c2 = st.columns(2)
    fraction_label = c1.radio("Kelly fraction", list(FRACTIONS), index=1, horizontal=True)
    amount = c2.number_input("Amount to allocate ($)", min_value=0, value=10_000,
                             step=1_000)

    with st.expander("Settings"):
        max_weight = st.slider("Max per name", 0.05, 1.0, ks.PORTFOLIO_MAX_WEIGHT, 0.05,
                               format="%.2f")
        max_lev = st.slider("Max invested (1.0 = no margin)", 0.5, 2.0,
                            ks.PORTFOLIO_MAX_LEVERAGE, 0.1)
        shrink = st.slider("Covariance shrinkage", 0.0, 1.0, ks.COV_SHRINK, 0.05)
        mkt_ret = st.number_input("Long-run market return", 0.0, 0.30,
                                  ks.MARKET_RETURN, 0.005, format="%.4f")
        log_scale = st.checkbox("Log scale on backtest chart",
                                help="Each gridline is a doubling. Better for long histories.")

    run = st.form_submit_button("Size it", type="primary", width="stretch")

if not run:
    st.info("Pick 2–15 tickers and tap **Size it**.")
    st.stop()

tickers = list(dict.fromkeys(ks._yahoo_symbol(t).upper() for t in picks))
if len(tickers) < 2:
    st.warning("Pick at least 2 tickers.")
    st.stop()

logs = capture_logs()
with st.spinner("Pulling prices and solving…"):
    rf, rf_src = risk_free()
    prices = prices_for(tuple(sorted(tickers)))
    if prices.empty or ks.BENCHMARK not in prices.columns:
        st.error("No price data came back from Yahoo. Try again in a minute.")
        st.stop()

    results = ks.compute_kelly(prices, rf, mkt_ret)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        results = pd.concat([results, pd.DataFrame({"ticker": missing,
                                                    "status": "no price data"})])
    results = meta.merge(results, on="ticker", how="right")

    try:
        weights, summary = ks.portfolio_kelly(prices, results, tickers, rf,
                                              max_weight=max_weight,
                                              max_leverage=max_lev,
                                              shrink=shrink)
    except ValueError as e:
        st.error(str(e))
        st.stop()

frac = FRACTIONS[fraction_label]
wcol = f"weight_{frac}_kelly"
s = summary.set_index("kelly_fraction").loc[frac]

# ---- headline numbers
m1, m2, m3 = st.columns(3)
m1.metric("Invested", f"{s['invested']:.0%}")
m2.metric("Cash", f"{s['cash']:.0%}")
m3.metric("Exp. growth", f"{s['expected_growth_rate']:.1%}")
m4, m5, m6 = st.columns(3)
m4.metric("Exp. return", f"{s['expected_return']:.1%}")
m5.metric("Volatility", f"{s['volatility']:.1%}")
m6.metric("Sharpe", f"{s['sharpe']:.2f}")

# ---- allocation table
tbl = weights[["ticker", wcol, "standalone_full_kelly", "annual_volatility", "beta"]].copy()
tbl = tbl.rename(columns={wcol: "weight"})
tbl["dollars"] = tbl["weight"] * amount
tbl = pd.concat([tbl[tbl.ticker != "CASH"].sort_values("weight", ascending=False),
                 tbl[tbl.ticker == "CASH"]])
for c in ["weight", "standalone_full_kelly", "annual_volatility"]:
    tbl[c] = tbl[c] * 100

st.subheader(f"{fraction_label} Kelly allocation")
st.dataframe(
    tbl[["ticker", "weight", "dollars", "standalone_full_kelly",
         "annual_volatility", "beta"]],
    hide_index=True,
    width="stretch",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", pinned=True),
        "weight": st.column_config.ProgressColumn("Weight", format="%.1f%%",
                                                  min_value=0, max_value=100),
        "dollars": st.column_config.NumberColumn("$", format="$%,.0f"),
        "standalone_full_kelly": st.column_config.NumberColumn("Solo Kelly",
                                                               format="%.0f%%"),
        "annual_volatility": st.column_config.NumberColumn("Vol", format="%.0f%%"),
        "beta": st.column_config.NumberColumn("Beta", format="%.2f"),
    },
)

# ---- all fractions side by side
with st.expander("Compare all Kelly fractions"):
    wide = weights[["ticker"] + [f"weight_{f}_kelly" for f in FRACTIONS.values()]].copy()
    wide.columns = ["Ticker"] + list(FRACTIONS)
    st.dataframe(wide.set_index("Ticker").map(lambda x: f"{x:.1%}"),
                 width="stretch")
    st.dataframe(
        summary.set_index("kelly_fraction")[
            ["invested", "expected_return", "volatility", "expected_growth_rate"]
        ].map(lambda x: f"{x:.1%}"),
        width="stretch",
    )

# ---- notes
notes = [m for m in logs if m.startswith("Portfolio:")]
st.caption(
    f"Risk-free {rf:.2%} ({rf_src}) · {int(s['months_used'])} months of shared history"
    + (f" · {'; '.join(n.removeprefix('Portfolio: ') for n in notes)}" if notes else "")
)
if not bool(s["solver_ok"]):
    st.warning("Optimizer didn't fully converge for this fraction; treat as approximate.")

# ---- backtest chart
st.subheader("Backtest vs S&P 500")
held = weights.set_index("ticker")[wcol].drop("CASH")
held = held[held > 0]
growth = None
if not held.empty:
    try:
        with st.spinner("Pulling full price history…"):
            growth = backtest(full_history(tuple(sorted(held.index))), held, rf)
    except Exception:
        pass
if growth is None or len(growth) < 13:
    st.caption("Price history for the backtest isn't available right now.")
else:
    start_value = amount or 10_000
    values = growth * start_value
    st.altair_chart(growth_chart(values, log=log_scale), width="stretch")
    p, b = values[PORT], values[BENCH]
    st.caption(
        f"Growth of ${start_value:,.0f} from {values.index[0]:%b %Y}, rebalanced monthly "
        f"to the {fraction_label.lower()} Kelly weights above"
        f"{' (log scale: each gridline is a doubling)' if log_scale else ''}. "
        f"Portfolio {cagr(p):.1%}/yr, worst drop {max_drawdown(p):.0%} · "
        f"S&P 500 {cagr(b):.1%}/yr, worst drop {max_drawdown(b):.0%}."
    )
    with st.expander("Backtest data"):
        yearly = values.resample("YE").last()
        yearly.index = yearly.index.year
        st.dataframe(yearly.rename_axis("Year").map(lambda x: f"${x:,.0f}"),
                     width="stretch")

# ---- methodology
st.subheader("Methodology")
st.markdown(
    f"""
The **Kelly criterion** sizes each bet to maximize long-run compound growth. Bet too
little and you leave growth on the table; bet too much and volatility drags growth
down, eventually below zero. For a single stock the optimal fraction is
*(expected return − risk-free rate) ÷ variance*, shown as **Solo Kelly** above. Holding
several stocks at once, correlations matter: two names that move together are closer
to one big bet, so the joint weights end up smaller than the solo figures.

Full Kelly is the growth-maximizing size, but it's a rough ride and very sensitive to
errors in the return estimates. **Half Kelly** keeps about 75% of the growth with half
the volatility, which is why fractions are the usual choice.

- **Expected returns** come from CAPM: the 10-year Treasury yield plus each stock's
  beta (5 years of monthly returns vs. the S&P 500) times the market risk premium
  ({mkt_ret:.1%} long-run market return minus the risk-free rate).
- **Risk** is the covariance of monthly returns over up to 10 years of shared history,
  shrunk {shrink:.0%} toward its diagonal to damp estimation noise.
- **Weights** maximize the basket's expected log growth (multi-asset Kelly), capped at
  {max_weight:.0%} per name and {max_lev:.0%} invested in total. Fractional Kelly is solved
  as higher risk aversion, not by scaling the full-Kelly weights. Anything not invested
  sits in cash.
- **The backtest** applies today's weights to each month's actual returns, with cash
  earning the 3-month T-bill rate and the S&P 500 including dividends. The weights were
  chosen with hindsight on the same data, and it ignores costs and taxes, so treat it
  as optimistic.

Sizes reflect risk and correlation, not conviction. Not financial advice.
"""
)
