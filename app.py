"""
Kelly Sizer — phone-friendly Streamlit front end for kelly_sp500.py, after
Concentrated Investing (Benello, van Biema & Carlisle), chapters 3-4.

Run locally:   streamlit run app.py
Deploy:        Streamlit Community Cloud -> point at this repo, main file app.py
"""

from __future__ import annotations

import importlib
import os
from datetime import date

import pandas as pd
import streamlit as st

import kelly_sp500 as ks

st.set_page_config(page_title="Kelly Sizer", page_icon="📐", layout="centered")


@st.cache_resource
def _engine_loaded() -> dict:
    """Process-wide record of which kelly_sp500.py version is in memory."""
    return {"mtime": os.path.getmtime(ks.__file__)}


# Streamlit re-runs app.py when it changes but keeps already-imported modules.
# After a merge, Cloud would pair the new app with the old engine in memory,
# so reload it when the file on disk is newer than what's loaded.
_loaded, _mtime = _engine_loaded(), os.path.getmtime(ks.__file__)
if _mtime != _loaded["mtime"] or not hasattr(ks, "journal_report"):
    ks = importlib.reload(ks)
    _loaded["mtime"] = _mtime

FRACTIONS = {"Full": "full", "Half": "half", "Quarter": "quarter", "Eighth": "eighth"}
DEFAULT_TICKERS = ["NVDA", "ASML", "AVGO", "MSFT", "VOO"]
VIEW_FIELDS = ["p_bear", "bear", "p_base", "base", "bull", "value"]


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


@st.cache_data(ttl=3600, show_spinner=False)
def latest_for(tickers: tuple[str, ...]) -> dict[str, float]:
    return ks.download_latest_prices(list(tickers))


def capture_logs():
    """Collect the engine's console messages so we can show them in the UI."""
    msgs: list[str] = []
    ks.log = msgs.append
    return msgs


def _pct(x):
    return None if x is None or pd.isna(x) else round(float(x) * 100, 4)


# ------------------------------------------------------------------ state ---
def _init_views() -> dict:
    """Start from views.csv in the repo if present (git add -f views.csv)."""
    try:
        v = ks.load_views("views.csv")
        return {t: {k: r[k] for k in VIEW_FIELDS} for t, r in v.iterrows()}
    except Exception:
        return {}


def _init_journal() -> pd.DataFrame:
    try:
        return ks.load_journal("journal.csv")
    except Exception:
        return ks.normalize_journal(None)


meta = sp500_list()
options = sorted(set(meta["ticker"]) | set(DEFAULT_TICKERS))
logs = capture_logs()

if "views" not in st.session_state:
    st.session_state.views = _init_views()
if "journal" not in st.session_state:
    st.session_state.journal = _init_journal()
if "ran" not in st.session_state:
    st.session_state.ran = False
if "ticker_options" not in st.session_state:
    # Frozen per session: if the option list changes, Streamlit treats the
    # picker as a new widget and resets the basket to the defaults.
    st.session_state.ticker_options = sorted(set(options) | set(st.session_state.views))

# --------------------------------------------------------------------- UI ---
st.title("📐 Kelly Sizer")
st.caption("Kelly position sizing after *Concentrated Investing*: bet big when you hold "
           "an edge, never risk ruin, and size the whole basket at once.")

picks = st.multiselect(
    "Tickers (type any symbol, e.g. TSM, ASML, VOO)",
    options=st.session_state.ticker_options,
    default=DEFAULT_TICKERS,
    max_selections=15,
    accept_new_options=True,
    key="picks",
)
tickers = list(dict.fromkeys(ks._yahoo_symbol(t).upper() for t in picks))
store = st.session_state.views

# ---- your research
with st.expander("Your research (views)", expanded=bool(store)):
    st.caption(
        "Your edge, from your own analysis, over the thesis horizon. **Bear**: the "
        "*permanent* loss if you're wrong (a broken thesis, not a price dip) and its odds. "
        "**Base** (optional): a middle case. **Bull**: the gain if you're right, or enter "
        "your **Value $** per share and the bull case becomes your margin of safety. "
        "Bull odds are whatever's left. Leave a row blank to use CAPM."
    )
    # Same cache key as the sizing step below, so price and margin of safety agree.
    viewed = tuple(sorted(t for t in tickers if t in store))
    px = latest_for(viewed) if viewed else {}
    rows = []
    for t in tickers:
        v = store.get(t, {})
        p_bull = (1 - (v.get("p_bear") or 0) - (v.get("p_base") or 0)) if v else None
        price = px.get(t)
        mos = (1 - price / v["value"]) if price and pd.notna(v.get("value")) else None
        rows.append({"ticker": t,
                     "p_bear": _pct(v.get("p_bear")), "bear": _pct(v.get("bear")),
                     "p_base": _pct(v.get("p_base")), "base": _pct(v.get("base")),
                     "bull": _pct(v.get("bull")), "value": v.get("value"),
                     "p_bull": _pct(p_bull), "price": price, "mos": _pct(mos)})
    grid = pd.DataFrame(rows, columns=["ticker", "p_bear", "bear", "p_base", "base",
                                       "bull", "value", "p_bull", "price", "mos"])
    pct = lambda label, **kw: st.column_config.NumberColumn(label, format="%.0f%%", **kw)
    edited = st.data_editor(
        grid, hide_index=True, width="stretch", key=f"views_{'_'.join(tickers)}",
        disabled=["ticker", "p_bull", "price", "mos"],
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "p_bear": pct("Bear odds", min_value=1, max_value=99, step=5),
            "bear": pct("Bear loss", min_value=1, max_value=100, step=5,
                        help="Permanent loss of capital if the thesis breaks"),
            "p_base": pct("Base odds", min_value=0, max_value=98, step=5),
            "base": pct("Base", step=5, help="Return in the middle case (can be negative)"),
            "bull": pct("Bull", min_value=1, step=5,
                        help="Gain if you're right. Leave blank to use Value $"),
            "value": st.column_config.NumberColumn("Value $", min_value=0.01, format="$%.2f",
                                                   help="Your intrinsic value per share"),
            "p_bull": pct("Bull odds", help="100% minus bear and base odds"),
            "price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "mos": pct("Margin of safety", help="1 − price ÷ value"),
        },
    )
    problems = []
    for _, r in edited.iterrows():
        entry = {k: (r[k] / 100 if k != "value" else r[k]) if pd.notna(r[k]) else None
                 for k in VIEW_FIELDS}
        if entry["p_bear"] is None or entry["bear"] is None or (
                entry["bull"] is None and entry["value"] is None):
            store.pop(r["ticker"], None)
            continue
        try:
            ks.normalize_views(pd.DataFrame([{"ticker": r["ticker"], **entry}]))
            store[r["ticker"]] = entry
        except ValueError as e:
            store.pop(r["ticker"], None)
            problems.append(f"**{r['ticker']}**: {e}")
    for msg in problems:
        st.error(msg)

    up, down = st.columns(2)
    file = up.file_uploader("Load views CSV", type="csv", key="views_upload",
                            label_visibility="collapsed")
    if file is not None and st.session_state.get("_views_loaded") != file.file_id:
        try:
            loaded = ks.normalize_views(pd.read_csv(file))
            store.update({t: {k: r[k] for k in VIEW_FIELDS} for t, r in loaded.iterrows()})
            st.session_state._views_loaded = file.file_id
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
    horizon = st.slider("Holding horizon (years)", 1.0, 10.0, ks.DEFAULT_HORIZON, 0.5,
                        help="Buy and hold for this long: how long your theses take to "
                             "play out. Applies to every name.")
    max_weight = st.slider("Max per name", 0.05, 1.0, ks.PORTFOLIO_MAX_WEIGHT, 0.05,
                           format="%.2f",
                           help="Buffett: '25 percent for the largest'; "
                                "40% for an American Express-level idea.")
    max_lev = st.slider("Max invested (1.0 = no margin)", 0.5, 2.0,
                        ks.PORTFOLIO_MAX_LEVERAGE, 0.1)
    shrink = st.slider("Correlation shrinkage", 0.0, 1.0, ks.COV_SHRINK, 0.05,
                       help="Pull correlations toward independence. Raise it if your "
                            "theses are less linked than the stocks' prices.")
    mkt_ret = st.number_input("Long-run market return (CAPM names)", 0.0, 0.30,
                              ks.MARKET_RETURN, 0.005, format="%.4f")

if st.button("Size it", type="primary", width="stretch"):
    st.session_state.ran = True

results_ready = False
if not st.session_state.ran:
    st.info("Pick 2–15 tickers, optionally add your research, and tap **Size it**. "
            "After that, results update as you change things.")
elif len(tickers) < 2:
    st.warning("Pick at least 2 tickers.")
else:
    in_basket = {t: v for t, v in store.items() if t in tickers}
    views = (ks.normalize_views(pd.DataFrame([{"ticker": t, **v}
                                              for t, v in in_basket.items()]))
             if in_basket else None)
    with st.spinner("Pulling prices and solving…"):
        rf, rf_src = risk_free()
        prices = prices_for(tuple(sorted(tickers)))
        prices_now = latest_for(tuple(sorted(in_basket))) if in_basket else {}
        if prices.empty or ks.BENCHMARK not in prices.columns:
            st.error("No price data came back from Yahoo. Try again in a minute.")
        else:
            results = ks.compute_kelly(prices, rf, mkt_ret, views=views,
                                       horizon=horizon, prices_now=prices_now)
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
                results_ready = True
            except ValueError as e:
                st.error(str(e))

if results_ready:
    frac = FRACTIONS[fraction_label]
    wcol = f"weight_{frac}_kelly"
    s = summary.set_index("kelly_fraction").loc[frac]
    n_views, hz = int(s["views_used"]), f"{s['horizon_years']:g}y"

    if n_views:
        st.success(f"Sized on your research for {n_views} of {int(s['basket_size'])} "
                   f"names over a {s['horizon_years']:g}-year hold. The rest use CAPM.")
    else:
        st.info("No views yet, so every name uses CAPM, which assumes no edge and sizes "
                "by risk alone. Add your research to size on conviction.")

    # ---- headline: the book defines risk as permanent loss of capital
    loss_help = ("Uses your permanent-loss inputs; CAPM names use price history"
                 if n_views else "CAPM names: estimated from price history")
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Chance of loss ({hz})", f"{s['prob_loss']:.0%}",
              help=f"Chance the basket ends the horizon below where it started. {loss_help}.")
    m2.metric(f"Worst case, 1-in-20 ({hz})", f"{s['worst_5pct']:.0%}",
              help="5th-percentile outcome over the horizon.")
    m3.metric("Exp. growth", f"{s['expected_growth_rate']:.1%}/yr",
              help="Expected compound growth, what Kelly maximizes.")
    m4, m5, m6 = st.columns(3)
    m4.metric("Invested", f"{s['invested']:.0%}")
    m5.metric("Cash", f"{s['cash']:.0%}")
    m6.metric("Price swings", f"{s['volatility']:.0%}/yr",
              help="How bumpy the ride is (annualized). Not risk in the book's sense, "
                   "but it's what you have to sit through. It's why fractional Kelly exists.")

    # ---- allocation table
    cols = ["ticker", wcol, "edge_source", "margin_of_safety", "standalone_full_kelly",
            "annual_volatility"]
    tbl = weights[[c for c in cols if c in weights.columns]].copy()
    tbl = tbl.rename(columns={wcol: "weight"})
    tbl["dollars"] = tbl["weight"] * amount
    tbl = pd.concat([tbl[tbl.ticker != "CASH"].sort_values("weight", ascending=False),
                     tbl[tbl.ticker == "CASH"]])
    for c in ["weight", "standalone_full_kelly", "annual_volatility", "margin_of_safety"]:
        if c in tbl.columns:
            tbl[c] = tbl[c] * 100
    show = ["ticker", "weight", "dollars", "edge_source", "standalone_full_kelly",
            "annual_volatility"]
    if "margin_of_safety" in tbl.columns and tbl["margin_of_safety"].notna().any():
        show.insert(4, "margin_of_safety")

    st.subheader(f"{fraction_label} Kelly allocation")
    st.dataframe(
        tbl[show], hide_index=True, width="stretch",
        column_config={
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "weight": st.column_config.ProgressColumn("Weight", format="%.1f%%",
                                                      min_value=0, max_value=100),
            "dollars": st.column_config.NumberColumn("$", format="$%,.0f"),
            "edge_source": st.column_config.TextColumn("Edge"),
            "margin_of_safety": st.column_config.NumberColumn("MoS", format="%.0f%%",
                                                              help="Margin of safety"),
            "standalone_full_kelly": st.column_config.NumberColumn(
                "Solo Kelly", format="%.0f%%",
                help="This name's Kelly bet on its own: a ceiling, not a target"),
            "annual_volatility": st.column_config.NumberColumn(
                "Swing", format="%.0f%%",
                help="Annual price volatility: how bumpy, not how risky"),
        },
    )

    with st.expander("Compare all Kelly fractions"):
        wide = weights[["ticker"] + [f"weight_{f}_kelly" for f in FRACTIONS.values()]].copy()
        wide.columns = ["Ticker"] + list(FRACTIONS)
        st.dataframe(wide.set_index("Ticker").map(lambda x: f"{x:.1%}"), width="stretch")
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

    notes = [m.removeprefix("Portfolio: ") for m in logs if m.startswith("Portfolio:")]
    notes += [m.removeprefix("Views: ") for m in logs if m.startswith("Views:")]
    st.caption(f"Risk-free {rf:.2%} ({rf_src}) · {int(s['months_used'])} months of "
               f"price history for correlations" + (f" · {'; '.join(notes)}" if notes else ""))
    if not bool(s["solver_ok"]):
        st.warning("Optimizer didn't fully converge for this fraction; treat as approximate.")

# ---- conviction journal
st.divider()
st.subheader("Conviction journal")
st.caption("Kelly is only as good as your probabilities. Log your views, record what "
           "actually happened when the horizon ends, and see whether your 60% calls come "
           "true about 60% of the time. Judge the thesis, not the price: a drop isn't a "
           "permanent loss.")
journal = st.session_state.journal
n_log = len([t for t in tickers if t in store])
if st.button(f"Log today's views ({n_log})", disabled=not (results_ready and n_log),
             width="stretch",
             help="Records each view with today's date and price. Size it first."):
    resolved = ks.resolve_views(views, {**prices.ffill().iloc[-1].to_dict(), **prices_now})
    st.session_state.journal = journal = ks.append_journal(
        journal, ks.journal_entries(resolved, horizon))
    st.toast(f"Logged {len(resolved)} views")

if len(journal):
    j_px = latest_for(tuple(sorted(set(journal["ticker"]))))
    entries, jsum, calib = ks.journal_report(journal, j_px)
    a, b, c = st.columns(3)
    a.metric("Calls", jsum["entries"])
    b.metric("Resolved", jsum["resolved"])
    c.metric("Due for an outcome", jsum["due"],
             help="Horizon has passed: record bear, base, or bull")
    if jsum["resolved"]:
        st.info(f"{jsum['verdict']} Brier score {jsum['brier']:.2f} (0 = perfect). "
                f"On average you gave {jsum['avg_prob_of_actual']:.0%} odds to what "
                f"actually happened.")
        st.dataframe(
            calib.rename(columns={"bucket": "Bull odds you gave", "calls": "Calls",
                                  "predicted_bull": "Predicted", "actual_bull": "Came true"}),
            hide_index=True, width="stretch",
            column_config={"Predicted": st.column_config.NumberColumn(format="%.0f%%"),
                           "Came true": st.column_config.NumberColumn(format="%.0f%%")},
        )
    view = entries[["date", "ticker", "status", "p_bull", "bull", "entry_price",
                    "return_since_entry", "horizon_ends", "outcome"]].copy()
    for col in ["p_bull", "bull", "return_since_entry"]:
        view[col] = view[col] * 100
    edited_j = st.data_editor(
        view, hide_index=True, width="stretch", key=f"journal_{len(journal)}",
        disabled=[c for c in view.columns if c != "outcome"],
        column_config={
            "date": st.column_config.DateColumn("Logged"),
            "ticker": st.column_config.TextColumn("Ticker", pinned=True),
            "status": st.column_config.TextColumn("Status"),
            "p_bull": st.column_config.NumberColumn("Bull odds", format="%.0f%%"),
            "bull": st.column_config.NumberColumn("Bull", format="%.0f%%"),
            "entry_price": st.column_config.NumberColumn("Entry", format="$%.2f"),
            "return_since_entry": st.column_config.NumberColumn("Since entry", format="%.0f%%",
                                                                help="Price return, for reference"),
            "horizon_ends": st.column_config.DateColumn("Horizon ends"),
            "outcome": st.column_config.SelectboxColumn(
                "Outcome", options=["", *ks.OUTCOMES],
                help="Which scenario played out for the thesis"),
        },
    )
    changed = edited_j["outcome"].fillna("") != journal["outcome"].fillna("")
    if changed.any():
        j = journal.copy()
        j.loc[changed, "outcome"] = edited_j.loc[changed, "outcome"].fillna("")
        j.loc[changed, "outcome_date"] = [date.today() if o else None
                                          for o in j.loc[changed, "outcome"]]
        st.session_state.journal = ks.normalize_journal(j)
        st.rerun()

up_j, down_j = st.columns(2)
jfile = up_j.file_uploader("Load journal CSV", type="csv", key="journal_upload",
                           label_visibility="collapsed")
if jfile is not None and st.session_state.get("_journal_loaded") != jfile.file_id:
    st.session_state.journal = ks.append_journal(journal, pd.read_csv(jfile))
    st.session_state._journal_loaded = jfile.file_id
    st.rerun()
if len(journal):
    down_j.download_button("Save journal CSV", journal.to_csv(index=False), "journal.csv",
                           "text/csv", width="stretch")
st.caption("The journal lives in this browser session. Save the CSV (on iPhone it goes "
           "to Files) and load it next time.")

# ---- methodology
st.subheader("Methodology")
st.markdown(
    f"""
After *Concentrated Investing* (Benello, van Biema & Carlisle), chapters 3–4.

**Kelly** sizes each bet to maximize the expected *logarithm* of wealth: long-run
compound growth. In the book's words it's **edge ÷ odds**: no edge, no bet; a bigger
edge or better odds, a bigger bet. Kelly is the *maximum* sensible size. Betting more
adds risk without adding growth, which is why **half Kelly** (about 75% of the growth
with half the volatility) is the usual choice. A fraction here means "bet that fraction
of what Kelly says," then apply your position limits.

**Your edge.** The book's investors find edge through research: a "high-probability
insight" bought with a margin of safety. Each view is up to three outcomes over the
holding horizon: a **bear** case of *permanent* loss (the book's definition of risk,
not volatility), an optional **base** case, and a **bull** case, entered directly or as
your intrinsic **value** per share against today's price. Names without a view use
CAPM, the 10-year Treasury plus beta × the market premium ({mkt_ret:.1%} long-run
return minus the risk-free rate). CAPM assumes no edge, so it sizes by risk alone.

**Sizing bets together.** Thorp's warning: Kelly computed for each bet alone
"generally overestimates" it, so you need the portfolio's joint properties. The app
simulates {ks.N_SCENARIOS:,} joint outcomes over a {horizon:g}-year buy-and-hold
(correlations from price history, shrunk {shrink:.0%} toward independence) and picks
the weights with the highest expected log wealth, capped at {max_weight:.0%} per name
and {max_lev:.0%} invested. A mix that could wipe out the portfolio scores −∞, so the
result never risks ruin. Whatever isn't invested earns the risk-free rate.

- **Chance of loss** and the **1-in-20 worst case** lead the results because risk is
  permanent loss. **Price swings** are shown separately: not risk, but the ride you
  sit through.
- **Solo Kelly** is each name's Kelly on its own, a ceiling, not a target.
- **The journal** checks the one thing Kelly can't: whether your probabilities are any
  good.

Kelly sizes on the edge you give it, so overconfident odds or an understated bear case
produce oversized bets. Not financial advice.
"""
)
