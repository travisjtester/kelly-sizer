# Kelly Sizer

Kelly Criterion position sizing, built to follow *Concentrated Investing* (Benello, van Biema & Carlisle, Wiley 2016), chapters 3–4: bet big when you hold an edge, don't bet when you don't, never risk ruin, and size the whole portfolio at once.

- **Your views:** bear / base / bull scenarios per name from your own research. The bear case is the *permanent* loss if you're wrong, and the bull case can come from your intrinsic-value estimate (margin of safety).
- **Portfolio Kelly:** weights that maximize expected log wealth for the whole basket over a buy-and-hold horizon (Thorp's "joint properties" rule).
- **Conviction journal:** log dated views, record what happened, and check whether your probabilities are calibrated.
- **Screener:** Kelly for every S&P 500 name under CAPM, the "no edge" baseline, to a dated CSV.
- **Phone app** (`app.py`) with all of the above.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
Windows: use PowerShell or a VS Code terminal. Quote paths that contain spaces (`cd "C:\...\Python Scripts\kelly-sp500"`); use `py` if `python` isn't recognized.

## Your views
```csv
ticker,p_bear,bear,p_base,base,bull,value,notes
NVDA,0.25,0.50,0.35,0.15,1.00,,"Bear: capex bust, permanent derating"
ASML,20,35,40,20,,1100,"Bull from intrinsic value $1,100/share"
KO,0.10,0.15,,,0.30,,"Two outcomes only: base left blank"
```

| Column | Meaning |
|---|---|
| `p_bear`, `bear` | Odds and size of the **permanent** loss if the thesis breaks (`0.5` = −50%). A broken thesis, not a price dip: the book defines risk as permanent loss of capital |
| `p_base`, `base` | Optional middle case: odds and return (can be negative) |
| `bull` | Gain if you're right (`1.0` = +100%) |
| `value` | Optional intrinsic value per share. If `bull` is blank, bull = value ÷ current price − 1, and the output shows **margin of safety** = 1 − price ÷ value |
| `notes` | Optional |

Bull odds are whatever is left: `1 − p_bear − p_base`. Use decimals or percents consistently within a column; a `%` sign always means percent. Files in the older `p_win, upside, downside` format still load (as bear/bull). Start from `views.example.csv`.

```bash
python kelly_sp500.py --views views.csv --portfolio NVDA ASML KO MSFT VOO
python kelly_sp500.py --views views.csv --portfolio NVDA ASML KO --horizon 5 --max-weight 0.40
```
Tickers in the views file are added to the run automatically. A name with a view is kept even with short price history (e.g. a recent IPO); with under 12 months of prices it's treated as uncorrelated.

## Portfolio Kelly
```bash
python kelly_sp500.py --portfolio AAPL MSFT KO TSM ASML NVDA
python kelly_sp500.py --portfolio-top 15
```

| Flag | Default | Purpose |
|---|---|---|
| `--portfolio` / `--portfolio-top N` | — | The basket: explicit, or the top N by solo Kelly |
| `--horizon` | 3 | Buy-and-hold years (how long theses take to play out) |
| `--max-weight` | 0.25 | Per-name cap (Buffett: "25 percent for the largest"; 0.40 for an AmEx-level idea) |
| `--max-leverage` | 1.0 | Max total invested; >1 allows margin |
| `--shrink` | 0.20 | Pull correlations toward independence (0–1) |
| `--scenarios` | 20000 | Simulated joint outcomes |

**How it sizes**
1. Each name gets a distribution of horizon returns: your scenarios where you have a view; otherwise **CAPM** (expected return = 10Y Treasury + beta × market premium, lognormal with historical volatility). CAPM assumes no edge, so it sizes by risk alone.
2. 20,000 joint outcomes are drawn with a Gaussian copula on historical correlations (viewed names land on bear / base / bull with your odds).
3. The weights maximize **expected log wealth** at the horizon, long-only, within your caps; the rest earns the risk-free rate. A mix that could wipe out the portfolio scores −∞, so the result never risks ruin.
4. **Fractional Kelly** c = "bet c × what Kelly says," then apply your caps: the Kelly bet is solved with limits scaled by 1/c and then multiplied by c. Uncapped, that's exactly c × full Kelly. When a cap binds, half Kelly is half the Kelly bet trimmed to the cap, not half of an already-capped bet.
5. **Solo Kelly** per name is the same calculation for that bet alone versus cash: a ceiling, not a target.

Seeds are fixed, so reruns give identical weights. Writes `kelly_portfolio_YYYY-MM-DD.csv` (weights + a `CASH` row) and `kelly_portfolio_summary_YYYY-MM-DD.csv`.

## Conviction journal
Kelly is only as good as your probabilities. The journal records each view with the date and entry price; when the horizon ends, you record which scenario played out (judge the thesis, not the price).

```bash
python kelly_sp500.py --views views.csv --portfolio NVDA ASML KO --journal journal.csv --log   # log today's views
python kelly_sp500.py --tickers KO --journal journal.csv                                       # report only
```
Fill the `outcome` column (`bear` / `base` / `bull`) in `journal.csv` as calls resolve. The report shows:
- calls that are **open**, **due** (horizon passed, outcome needed) and **resolved**
- **calibration**: for the bull odds you gave (<40%, 40–60%, 60–80%, 80%+), how often bull actually happened
- **Brier score** (0 = perfect) and the average probability you gave to what actually happened
- a verdict once there are 10+ resolved calls: well calibrated, running hot (lower your odds or bet a smaller fraction), or underconfident

## Phone app (Streamlit)
```bash
python -m streamlit run app.py          # local: opens http://localhost:8501
```
**Deploy (free, Streamlit Community Cloud):** push to GitHub → share.streamlit.io → **Create app** → this repo, `app.py`. On iPhone: open the URL in Safari → Share → **Add to Home Screen**.

- **Your research:** Bear odds / Bear loss / Base odds / Base / Bull or Value $ per ticker; bull odds, current price and margin of safety fill in automatically. Blank row = CAPM.
- **Results lead with the book's risk:** chance of loss and the 1-in-20 worst case over the horizon, then expected growth, invested, cash. **Price swings** (volatility) are shown separately: not risk, but the ride you sit through, which is why fractional Kelly exists.
- **Conviction journal:** log today's views, pick each outcome from a dropdown when it resolves, and see calibration.
- **Save / Load** views and journal as CSVs (iPhone saves to Files). Cloud sessions don't keep files, so save before you leave. To preload, commit `views.csv` / `journal.csv` to the repo root (`git add -f`, since `*.csv` is gitignored), but only if you're fine with them being as public as the repo.

Results update live after the first **Size it**. Prices are cached (monthly 12h, latest 1h, risk-free 6h, constituents 24h).

## How the book maps to the tool

| Book (page) | Tool |
|---|---|
| Kelly maximizes expected log of wealth; edge/odds; no edge, no bet (pp. 75–76) | The objective everywhere; reproduces the book's worked example (51% at even money → 2%) |
| Kelly never risks ruin (p. 86) | Ruinous mixes score −∞; solo Kelly is bounded below the ruin point |
| Edge from fundamental research on probability and risk/reward (p. 2) | Your bear / base / bull scenarios |
| Risk = permanent loss of capital, not volatility (p. 3) | Bear case = permanent loss; results lead with chance of loss; volatility labeled "price swings" |
| Margin of safety (pp. 3, 208) | `value` input → bull case and margin-of-safety column |
| Size bets jointly; isolated Kelly "generally overestimates" (p. 105) | One solve for the whole basket; solo Kelly shown as a ceiling |
| Kelly is the maximum bet; half Kelly ≈ ¾ of the growth, ½ the volatility (pp. 85–86) | Fractions = c × the Kelly bet, then caps; leverage 1.0 by default |
| Buffett: "25 percent for the largest," up to 40% (p. 96) | `--max-weight` default 0.25 |
| Concentrate in 5–20 names (p. 3) | App takes up to 15 |
| Buy and hold; Shannon didn't trim (p. 82) | Buy-and-hold horizon, no rebalancing assumption |
| Kelly needs accurate probabilities; misjudged downside overstates the bet (p. 105) | Conviction journal + calibration |
| "Know-nothing" investors should index (p. 3) | Names without a view are sized as market exposure under CAPM |

## Method details
| Input | Source | Fallback |
|---|---|---|
| Constituents | Wikipedia | GitHub `datasets/s-and-p-500-companies` |
| Risk-free rate | FRED DGS10 (10Y Treasury) | Yahoo `^TNX`, then 4.5% |
| Monthly prices | Yahoo via yfinance, dividend/split-adjusted | 3 retries with backoff |
| Latest prices (value / journal) | Yahoo daily closes | last month-end close |

- **Beta**: regressed vs `^GSPC`, 5y monthly returns. **Volatility**: 10y monthly, annualized; CAPM names need 60+ months.
- **CAPM horizon returns**: lognormal with mean `ln(1+E[R]) − σ²/2` per year; solo Kelly via 41-point Gauss–Hermite quadrature.
- **Correlations**: pairwise over shared months (12+), shrunk toward independence, repaired to be positive semi-definite.
- Update `MARKET_RETURN` once a year (link in the script).

## Outputs
**`kelly_sp500_YYYY-MM-DD.csv`**, sorted by `full_kelly`. Rates and weights are decimals (0.12 = 12%).

| Column | Definition |
|---|---|
| `run_date`, `ticker`, `name`, `sector`, `universe` | Identity (`universe` = `S&P 500` or `Extra`) |
| `edge_source` | `views` if Kelly comes from your scenarios, else `CAPM` |
| `beta`, `expected_return` | CAPM inputs (annual) |
| `annual_variance`, `annual_volatility` | 10y monthly, annualized |
| `hist_cagr` | Realized compound annual return over the data window (context only) |
| `months_of_data`, `horizon_years` | History used; holding horizon |
| `full_kelly` … `eighth_kelly` | Solo Kelly over the horizon, and fractions |
| `capm_full_kelly` | CAPM solo Kelly, for comparison with your view |
| `p_bear`, `bear`, `p_base`, `base`, `p_bull`, `bull`, `value`, `price`, `margin_of_safety` | Your view (views only) |
| `view_expected_return`, `view_annualized_return` | Probability-weighted return over the horizon, and annualized |
| `status` | `ok`, `insufficient history (N mo)`, or `no price data` |
| `risk_free_rate`, `market_return` | Run assumptions |

**`kelly_portfolio_YYYY-MM-DD.csv`**: one row per name plus `CASH`; `weight_full_kelly` … `weight_eighth_kelly` each sum to 1, with `standalone_full_kelly`, `edge_source` and the view columns.

**`kelly_portfolio_summary_YYYY-MM-DD.csv`**: one row per fraction: `views_used`, `basket_size`, `months_used`, `horizon_years`, `invested`, `cash`, `holdings`, `prob_loss`, `worst_5pct` (both over the horizon), `expected_growth_rate` (annualized compound), `expected_return` (annualized), `volatility` (price swings, annualized), `solver_ok`.

## Operational notes
- **FRED timeouts** are common from scripts; the run falls back to Yahoo `^TNX` (same 10Y yield).
- **Locked output files**: if a CSV is open in Excel, the script writes a timestamped copy instead of failing.
- **Yahoo on cloud hosts**: shared cloud IPs get rate-limited more than home IPs. If the hosted app returns no data, self-host on a home machine/NAS.
- **After a deploy** the app reloads the engine automatically when `kelly_sp500.py` changes.

## Caveats
- **Your inputs are the whole game.** Kelly sizes on the edge you give it: overconfident odds or an understated bear case produce oversized bets (Thorp, p. 105). Use half or quarter Kelly, and let the journal tell you whether your odds deserve more.
- Scenarios are a simplification of a real distribution, and all views share one horizon.
- Correlations between theses are borrowed from how the stocks' prices moved together. For idiosyncratic theses that likely overstates the link; raise `--shrink`.
- Names without a view use CAPM, which by design has no edge.
- Extras (e.g. TSM, ASML ADRs) are priced in USD, so FX is in their returns.
- Not financial advice.
