"""
Kelly Portfolio Sizer — phone-friendly Streamlit front end for kelly_sp500.py.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud -> point at this repo, main file app.py
"""

from __future__ import annotations

import importlib
import os

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import kelly_sp500 as ks

st.set_page_config(page_title="Kelly Sizer", page_icon="📐", layout="centered")


@st.cache_resource
def _engine_loaded() -> dict:
    """Process-wide record of which kelly_sp500.py version is in memory."""
    return {"mtime": os.path.getmtime(ks.__file__)}


# Streamlit re-runs app.py when it changes but keeps already-imported modules.
# After a merge, Cloud would pair the new app with the old engine in memory
# (e.g. AttributeError: no attribute 'DEFAULT_HORIZON'), so reload it when the
# file on disk is newer than what's loaded.
_loaded, _mtime = _engine_loaded(), os.path.getmtime(ks.__file__)
if _mtime != _loaded["mtime"] or not hasattr(ks, "DEFAULT_HORIZON"):
    ks = importlib.reload(ks)
    _loaded["mtime"] = _mtime

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
st.caption("Joint Kelly weights for a basket, with correlations and position caps. "
           "Add your own research to size on conviction.")

meta = sp500_list()
options = sorted(set(meta["ticker"]) | set(DEFAULT_TICKERS))
VIEW_EDIT_COLS = {"p_win": "Win %", "upside": "Upside %", "downside": "Downside %"}


def _init_views() -> dict:
    """Start from views.csv in the repo if present (git add -f views.csv)."""
    try:
        v = ks.load_views("views.csv")
        return {t: dict(r) for t, r in v[ks.VIEW_COLS].iterrows()}
    except Exception:
        return {}


if "views" not in st.session_state:
    st.session_state.views = _init_views()
if "ran" not in st.session_state:
    st.session_state.ran = False
if "ticker_options" not in st.session_state:
    # Frozen per session: if the option list changes, Streamlit treats the
    # picker as a new widget and resets the basket to the defaults.
    st.session_state.ticker_options = sorted(set(options) | set(st.session_state.views))

picks = st.multiselect(
    "Tickers (type any symbol, e.g. TSM, ASML, VOO)",
    options=st.session_state.ticker_options,
    default=DEFAULT_TICKERS,
    max_selections=15,
    accept_new_options=True,
    key="picks",
)
tickers = list(dict.fromkeys(ks._yahoo_symbol(t).upper() for t in picks))

with st.expander("Your research (views)", expanded=bool(st.session_state.views)):
    st.caption(
        "The book's Kelly inputs, from your own analysis. **Win %**: chance the thesis "
        "plays out over the horizon. **Upside %**: gain if it does. **Downside %**: "
        "*permanent* loss if it doesn't (not a price dip). Leave a row blank to use CAPM."
    )
    store = st.session_state.views
    grid = pd.DataFrame(
        [{"ticker": t,
          **{c: (store[t][c] * 100 if t in store else None) for c in VIEW_EDIT_COLS}}
         for t in tickers]
    )
    edited = st.data_editor(
        grid, hide_index=True, width="stretch", key=f"views_{'_'.join(tickers)}",
        disabled=["ticker"],
        column_config={
            "ticker": st.column_config.TextColumn("Ticker"),
            "p_win": st.column_config.NumberColumn("Win %", min_value=1, max_value=99,
                                                   step=5, format="%d%%"),
            "upside": st.column_config.NumberColumn("Upside %", min_value=1, step=5,
                                                    format="%d%%"),
            "downside": st.column_config.NumberColumn("Downside %", min_value=1,
                                                      max_value=100, step=5,
                                                      format="%d%%"),
        },
    )
    for _, r in edited.iterrows():
        vals = [r[c] for c in VIEW_EDIT_COLS]
        if all(pd.notna(v) for v in vals):
            store[r["ticker"]] = {c: float(r[c]) / 100 for c in VIEW_EDIT_COLS}
        else:
            store.pop(r["ticker"], None)

    up, down = st.columns(2)
    file = up.file_uploader("Load views CSV", type="csv", label_visibility="collapsed")
    if file is not None and st.session_state.get("_loaded") != file.file_id:
        try:
            loaded = ks.normalize_views(pd.read_csv(file))
            store.update({t: dict(r) for t, r in loaded[ks.VIEW_COLS].iterrows()})
            st.session_state._loaded = file.file_id
            st.rerun()
        except ValueError as e:
            st.error(str(e))
    if store:
        csv = (pd.DataFrame.from_dict(store, orient="index").rename_axis("ticker")
               .reset_index().to_csv(index=False))
        down.download_button("Save views CSV", csv, "views.csv", "text/csv",
                             width="stretch")

c1, c2 = st.columns(2)
fraction_label = c1.radio("Kelly fraction", list(FRACTIONS), index=1, horizontal=True)
amount = c2.number_input("Amount to allocate ($)", min_value=0, value=10_000, step=1_000)

with st.expander("Settings"):
    horizon = st.slider("Thesis horizon (years)", 1.0, 10.0, ks.DEFAULT_HORIZON, 0.5,
                        help="How long your views take to play out. Used only when "
                             "at least one name has a view.")
    max_weight = st.slider("Max per name", 0.05, 1.0, ks.PORTFOLIO_MAX_WEIGHT, 0.05,
                           format="%.2f",
                           help="Buffett: '25 percent for the largest'; "
                                "40% for an American Express-level idea.")
    max_lev = st.slider("Max invested (1.0 = no margin)", 0.5, 2.0,
                        ks.PORTFOLIO_MAX_LEVERAGE, 0.1)
    shrink = st.slider("Covariance shrinkage", 0.0, 1.0, ks.COV_SHRINK, 0.05)
    mkt_ret = st.number_input("Long-run market return", 0.0, 0.30,
                              ks.MARKET_RETURN, 0.005, format="%.4f")

if st.button("Size it", type="primary", width="stretch"):
    st.session_state.ran = True
if not st.session_state.ran:
    st.info("Pick 2–15 tickers, optionally add your research, and tap **Size it**. "
            "After that, results update as you change things.")
    st.stop()
if len(tickers) < 2:
    st.warning("Pick at least 2 tickers.")
    st.stop()

views = ks.normalize_views(
    pd.DataFrame([{"ticker": t, **v} for t, v in st.session_state.views.items()
                  if t in tickers])
) if any(t in st.session_state.views for t in tickers) else None

logs = capture_logs()
with st.spinner("Pulling prices and solving…"):
    rf, rf_src = risk_free()
    prices = prices_for(tuple(sorted(tickers)))
    if prices.empty or ks.BENCHMARK not in prices.columns:
        st.error("No price data came back from Yahoo. Try again in a minute.")
        st.stop()

    results = ks.compute_kelly(prices, rf, mkt_ret, views=views, horizon=horizon)
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        results = pd.concat([results, pd.DataFrame({"ticker": missing,
                                                    "status": "no price data"})])
    results = meta.merge(results, on="ticker", how="right")

    try:
        weights, summary = ks.portfolio_kelly(prices, results, tickers, rf,
                                              max_weight=max_weight,
                                              max_leverage=max_lev,
                                              shrink=shrink, horizon=horizon)
    except ValueError as e:
        st.error(str(e))
        st.stop()

frac = FRACTIONS[fraction_label]
wcol = f"weight_{frac}_kelly"
s = summary.set_index("kelly_fraction").loc[frac]
views_mode = s["engine"] == "views"
hz = f"{s['horizon_years']:g}y"

if views_mode:
    st.success(f"Views mode: sized on your research over a {s['horizon_years']:g}-year "
               f"horizon. Names without a view use CAPM.")

# ---- headline numbers
# The book defines risk as permanent loss of capital, so loss odds lead.
loss_help = ("From your permanent-loss inputs" if views_mode
             else "CAPM mode: estimated from price volatility (add views to use "
                  "permanent-loss inputs)")
m1, m2, m3 = st.columns(3)
m1.metric(f"Chance of loss ({hz})", f"{s['prob_loss']:.0%}",
          help=f"Chance the basket ends below where it started. {loss_help}.")
m2.metric(f"Worst case, 1-in-20 ({hz})", f"{s['worst_5pct']:.0%}",
          help="5th-percentile outcome over the horizon.")
m3.metric("Exp. growth", f"{s['expected_growth_rate']:.1%}/yr",
          help="Expected compound growth, what Kelly maximizes.")
m4, m5, m6 = st.columns(3)
m4.metric("Invested", f"{s['invested']:.0%}")
m5.metric("Cash", f"{s['cash']:.0%}")
m6.metric("Price swings", f"{s['volatility']:.0%}/yr",
          help="How bumpy the ride is (annualized volatility). Not risk in the book's "
               "sense, but it's what you have to sit through. It's why fractional "
               "Kelly exists.")

# ---- allocation table
cols = ["ticker", wcol, "edge_source", "standalone_full_kelly", "annual_volatility", "beta"]
tbl = weights[[c for c in cols if c in weights.columns]].copy()
tbl = tbl.rename(columns={wcol: "weight"})
tbl["dollars"] = tbl["weight"] * amount
tbl = pd.concat([tbl[tbl.ticker != "CASH"].sort_values("weight", ascending=False),
                 tbl[tbl.ticker == "CASH"]])
for c in ["weight", "standalone_full_kelly", "annual_volatility"]:
    tbl[c] = tbl[c] * 100

st.subheader(f"{fraction_label} Kelly allocation")
st.dataframe(
    tbl[[c for c in ["ticker", "weight", "dollars", "edge_source",
                     "standalone_full_kelly", "annual_volatility", "beta"]
         if c in tbl.columns]],
    hide_index=True,
    width="stretch",
    column_config={
        "ticker": st.column_config.TextColumn("Ticker", pinned=True),
        "weight": st.column_config.ProgressColumn("Weight", format="%.1f%%",
                                                  min_value=0, max_value=100),
        "dollars": st.column_config.NumberColumn("$", format="$%,.0f"),
        "edge_source": st.column_config.TextColumn("Edge"),
        "standalone_full_kelly": st.column_config.NumberColumn("Solo Kelly",
                                                               format="%.0f%%"),
        "annual_volatility": st.column_config.NumberColumn(
            "Swing", format="%.0f%%",
            help="Annual price volatility: how bumpy, not how risky"),
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
            ["prob_loss", "worst_5pct", "expected_growth_rate", "invested",
             "expected_return", "volatility"]
        ].rename(columns={
            "prob_loss": f"Chance of loss ({hz})", "worst_5pct": f"Worst 1-in-20 ({hz})",
            "expected_growth_rate": "Exp. growth/yr", "invested": "Invested",
            "expected_return": "Exp. return/yr", "volatility": "Price swings/yr",
        }).rename_axis("Kelly").map(lambda x: f"{x:.1%}"),
        width="stretch",
    )

# ---- notes
notes = [m for m in logs if m.startswith("Portfolio:")]
st.caption(
    f"Risk-free {rf:.2%} ({rf_src}) · {int(s['months_used'])} months of price history"
    + (f" · {'; '.join(n.removeprefix('Portfolio: ') for n in notes)}" if notes else "")
)
if not bool(s["solver_ok"]):
    st.warning("Optimizer didn't fully converge for this fraction; treat as approximate.")

# ---- optional price-history check (outside the book's method)
st.divider()
show_bt = st.toggle(
    "Price-history check (backtest)", value=False,
    help="Optional. Replays today's weights on past prices. It isn't part of the "
         "book's method and can't test your views.")
if not show_bt:
    st.caption("Off by default. The book sizes bets on forward-looking research, not "
               "on how a basket happened to trade. Turn on to see how bumpy this mix "
               "has been historically.")
held = weights.set_index("ticker")[wcol].drop("CASH")
held = held[held > 0]
growth = None
log_scale = False
if show_bt:
    log_scale = st.checkbox("Log scale",
                            help="Each gridline is a doubling. Better for long histories.")
if show_bt and not held.empty:
    try:
        with st.spinner("Pulling full price history…"):
            growth = backtest(full_history(tuple(sorted(held.index))), held, rf)
    except Exception:
        pass
if not show_bt:
    pass
elif growth is None or len(growth) < 13:
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
        + " This replays past prices only: it shows the ride, not whether your "
          "theses are right. Weights were chosen with hindsight; no costs or taxes."
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
Inspired by *Concentrated Investing* (Benello, van Biema & Carlisle), chapters 3–4.

The **Kelly criterion** sizes each bet to maximize the expected *logarithm* of wealth,
i.e. long-run compound growth. In the book's terms it's **edge ÷ odds**: no edge, no
bet; a bigger edge or better odds, a bigger bet. Kelly is the *maximum* sensible size.
Betting more adds risk without adding growth, which is why **half Kelly** (about 75% of
the growth with half the volatility) is the common choice.

**Where the edge comes from.** The book's investors find it through fundamental
research, a "high-probability insight" with a margin of safety. That's what the
**research table** captures: your probability that the thesis works, the upside if it
does, and the *permanent* loss if it doesn't. The book defines risk as permanent loss,
not volatility. Names without a view fall back to CAPM: the 10-year Treasury yield plus
beta × the market premium ({mkt_ret:.1%} long-run market return minus the risk-free
rate). CAPM assumes no edge, so it sizes by risk alone.

**Sizing bets together.** Thorp's warning: computing each Kelly bet on its own
"generally overestimates" it. You need the portfolio's joint properties. So weights are
solved for the whole basket at once, using how the names have historically moved
together (correlations shrunk {shrink:.0%} toward independence), capped at
{max_weight:.0%} per name and {max_lev:.0%} invested in total. Anything not invested sits
in cash.

- **Views mode** (any name has a view): simulates {ks.N_SCENARIOS:,} correlated outcomes
  over a {horizon:g}-year horizon, each viewed name winning or suffering its permanent
  loss and the others following CAPM, then picks the weights with the highest
  expected log wealth. A bet that could wipe out the portfolio scores −∞, so the
  result never risks ruin. Fractional Kelly scales those bets (half Kelly = bet half).
- **CAPM mode** (no views): the continuous form, *(expected return − risk-free) ÷
  variance*, solved jointly from up to 10 years of monthly returns.
- **Solo Kelly** is each name's Kelly on its own, a ceiling, not a target.
- **Risk vs. swings.** Chance of loss and the 1-in-20 worst case lead because the book
  defines risk as permanent loss of capital. **Price swings** (volatility) are shown
  separately. They aren't risk, but they're the drawdowns you have to sit through, and
  they're why Thorp recommends fractional Kelly.
- **The price-history check** (optional, off by default) replays today's weights on past
  monthly returns versus the S&P 500 with dividends. It's outside the book's method:
  it can't test your views, the weights were chosen with hindsight, and it ignores
  costs and taxes.

Your inputs are the whole game here: Kelly sizes on the edge you give it, so optimistic
views produce oversized bets. Not financial advice.
"""
)
