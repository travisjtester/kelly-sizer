"""
Kelly Portfolio Sizer — phone-friendly Streamlit front end for kelly_sp500.py.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud -> point at this repo, main file app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

import kelly_sp500 as ks

st.set_page_config(page_title="Kelly Sizer", page_icon="📐", layout="centered")

FRACTIONS = {"Full": "full", "Half": "half", "Quarter": "quarter", "Eighth": "eighth"}
DEFAULT_TICKERS = ["NVDA", "ASML", "AVGO", "MSFT", "VOO"]


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
st.caption("CAPM expected returns; sizes reflect risk and correlation, not conviction. "
           "Not financial advice.")
