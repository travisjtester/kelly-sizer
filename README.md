# Kelly Criterion S&P 500 Screener

Kelly Criterion position sizing, inspired by *Concentrated Investing* (Benello, van Biema & Carlisle, Wiley 2016), chapters 3–4.

- **Screener:** CAPM-based Kelly for every S&P 500 constituent (plus extras), written to a dated CSV.
- **Portfolio Kelly:** joint weights for a basket, accounting for correlation (Thorp's "joint properties" rule).
- **Views mode:** size on *your* edge, meaning probability of winning, upside, and permanent downside from your own research, which is how the book's investors use Kelly. Names without a view fall back to CAPM.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run
Windows: use PowerShell or a VS Code terminal. Quote paths that contain spaces (`cd "C:\...\Python Scripts\kelly-sp500"`); use `py` if `python` isn't recognized.
```bash
python kelly_sp500.py                         # full index -> Desktop/kelly_sp500_YYYY-MM-DD.csv
python kelly_sp500.py --limit 25              # quick smoke test
python kelly_sp500.py --tickers AAPL MSFT KO  # only these names
python kelly_sp500.py --extra TSM ASML        # full index + non-S&P names
python kelly_sp500.py --out data/run.csv --market-return 0.10
```

### Portfolio Kelly
```bash
python kelly_sp500.py --portfolio AAPL MSFT KO TSM ASML NVDA
python kelly_sp500.py --portfolio-top 15
python kelly_sp500.py --portfolio-top 15 --max-weight 0.15 --max-leverage 1.0
```
Writes two extra files next to the main CSV: `kelly_portfolio_YYYY-MM-DD.csv` (weights) and `kelly_portfolio_summary_YYYY-MM-DD.csv` (portfolio stats per Kelly fraction). Tickers passed to `--portfolio` are downloaded automatically even if they're not in the index.

**Basket rules**
- Names with under 60 months of history are dropped and logged (e.g. recent IPOs like ARM).
- Covariance uses only the months **all** remaining names share; the console logs the window and which ticker limits it. One young name shortens the window for everyone, so drop it for a full 10-year estimate.
- Use one share class per company (GOOG *or* GOOGL).
- Add a broad-market ETF (e.g. VOO) to size a concentrated basket against the market instead of against cash.

| Flag | Default | Purpose |
|---|---|---|
| `--portfolio` | — | Explicit basket to optimize |
| `--portfolio-top N` | — | Basket = top N by standalone full Kelly |
| `--max-weight` | 0.25 | Per-name cap |
| `--max-leverage` | 1.0 | Max total invested; >1 allows margin |
| `--shrink` | 0.20 | Covariance shrinkage toward diagonal (0 = raw sample) |

## Views mode (your own edge)
CAPM assumes no stock has an edge, so on its own it sizes by risk. The book's investors find edge through fundamental research and size on it. Give the script your three Kelly inputs per name:

```csv
ticker,p_win,upside,downside,notes
NVDA,0.60,0.80,0.40,"AI capex continues vs. capex bust"
ASML,70,60,30,"EUV monopoly"
```

| Column | Meaning |
|---|---|
| `p_win` | Probability the thesis plays out over the horizon |
| `upside` | Gain if it does (`0.8` = +80%; `1.5` = +150%) |
| `downside` | **Permanent** loss if it doesn't (`0.4` = −40%). Not a price dip; the book defines risk as permanent loss of capital |
| `notes` | Optional |

Use decimals or percents consistently within a column; a `%` sign always means percent. Start from `views.example.csv`.

```bash
python kelly_sp500.py --views views.csv --portfolio NVDA ASML KO MSFT VOO
python kelly_sp500.py --views views.csv --portfolio NVDA ASML KO --horizon 5 --max-weight 0.40
```

Tickers in the views file are added to the run automatically. A name with a view is kept even with short price history (e.g. a recent IPO); it's treated as uncorrelated if it has under 12 months of prices.

**How it sizes**
- **Solo Kelly** for a viewed name is the exact binary bet against cash at the horizon: `f* = (1+Rf)·(p/(a+Rf) − q/(b−Rf))` (b = upside, a = downside, Rf = risk-free over the horizon). With Rf = 0 and a total-loss downside it is the book's `(bp − q)/b`; the book's worked example (51%, even money) gives 2%.
- **Portfolio (views engine):** simulates 20,000 joint outcomes over the horizon with a Gaussian copula on historical correlations. Viewed names win or take their permanent loss; other names are lognormal with CAPM mean and historical volatility. It then maximizes **expected log wealth** for the whole basket at once. A position set that could wipe out the portfolio scores −∞, so the result never risks ruin. Fractional Kelly scales the full-Kelly bets (half Kelly = bet half). The seed is fixed, so reruns are identical.
- If no basket name has a view, the CAPM engine below runs unchanged.

| Flag | Default | Purpose |
|---|---|---|
| `--views` | — | Views CSV |
| `--horizon` | 3 | Years for theses to play out |
| `--scenarios` | 20000 | Simulated joint outcomes |

**How the book's principles map to the tool**

| Book (page) | Tool |
|---|---|
| Kelly maximizes expected log of wealth; edge/odds (pp. 75–76) | Views engine objective; binary closed form for solo Kelly |
| Edge comes from fundamental analysis of probability and risk/reward (p. 2) | `p_win`, `upside`, `downside` |
| Risk = permanent loss of capital, not volatility (p. 3) | `downside` is permanent loss; app headline leads with chance of loss; volatility shown as "price swings"; backtest opt-in |
| Size bets jointly; isolated Kelly "generally overestimates" (p. 105) | Portfolio solved as one basket; solo Kelly shown as a ceiling |
| Kelly is the maximum bet; half Kelly ≈ ¾ of the growth, ½ the volatility (pp. 85–86) | Fractions; leverage capped at 1.0 by default |
| Buffett: "25 percent for the largest," up to 40% (p. 96) | `--max-weight` default 0.25 |
| Concentrate in 5–20 names (p. 3) | App takes up to 15 |

## Phone app (Streamlit)
`app.py` is a mobile-friendly front end: pick up to 15 tickers, choose a Kelly fraction and a dollar amount, and get the allocation table plus portfolio stats.

```bash
streamlit run app.py          # local: opens http://localhost:8501
```

**Deploy (free, Streamlit Community Cloud)**
1. Push this repo to GitHub.
2. Go to share.streamlit.io → **Create app** → pick the repo, branch, and `app.py`.
3. In app settings, restrict viewers to your email if you want it private.
4. On iPhone: open the URL in Safari → Share → **Add to Home Screen**.

**Your research in the app:** open **Your research (views)**, fill Win % / Upside % / Downside % for any picked ticker (blank = CAPM), and the app switches to views mode. Use **Save views CSV** to keep them (iPhone saves to Files) and the uploader to load them back. To have the app start with your views, commit a `views.csv` at the repo root (`git add -f views.csv`, since `*.csv` is gitignored). Only do that if you're fine with it being as public as the repo.

After the first **Size it**, results update live as you change tickers, research, or settings.

**How the app frames risk (following the book):** the headline leads with **Chance of loss** and the **1-in-20 worst case** over the horizon. In views mode both come from your permanent-loss inputs, matching the book's definition of risk. **Price swings** (volatility) are shown separately: not risk, but the ride you have to sit through, which is why fractional Kelly exists. The **price-history check** (backtest vs. the S&P 500) is optional and off by default, since it's outside the book's method and can't test your views. Turning it off also skips the full-history download.

Prices are cached 12h per basket, the risk-free rate 6h, and the constituent list 24h, so repeat taps are instant and Yahoo isn't hammered.

## Method
| Input | Source | Fallback |
|---|---|---|
| Constituents | Wikipedia | GitHub `datasets/s-and-p-500-companies` |
| Risk-free rate | FRED DGS10 (10Y Treasury) | Yahoo `^TNX`, then 4.5% |
| Prices | Yahoo via yfinance, monthly, dividend/split-adjusted | 3 retries with backoff |
| Extra-ticker name/sector | Yahoo `.info` | blank |

- **Beta**: regressed vs `^GSPC`, 5y monthly returns
- **Expected return**: CAPM, `Rf + beta * (MARKET_RETURN - Rf)`
- **Variance**: 10y monthly returns, annualized x12; requires 60+ months
- **Standalone Kelly**: `(E[R] - Rf) / variance`, plus 1/2, 1/4, 1/8 fractions
- **Portfolio Kelly**: maximizes `w'(mu - Rf) - (1/2c) * w'Σw` subject to `0 <= w_i <= max_weight`, `sum(w) <= max_leverage`, solved separately for each fraction `c`. Σ is the shrunk, annualized covariance over the months all basket names share. Uninvested weight is cash, assumed to earn Rf.
- **Cash** = 1 − Σw. The optimizer stops adding to a name once its marginal excess return no longer covers the marginal variance it adds, given current holdings. With no cap binding, `w = c · Σ⁻¹(mu − Rf)`, so weights scale linearly with `c` and cash grows as the fraction shrinks. Full Kelly often hits `--max-leverage` and shows 0% cash.

Update `MARKET_RETURN` once a year (link in the script).

## Output: `kelly_sp500_YYYY-MM-DD.csv`
Sorted by `full_kelly` descending; rows with no result sort last. Rates and weights are decimals (0.12 = 12%).

| Column | Definition |
|---|---|
| `run_date` | Run date |
| `ticker` | Yahoo symbol (`.` → `-`, e.g. BRK-B) |
| `name` | Company name |
| `sector` | GICS sector (Yahoo's sector label for extras) |
| `universe` | `S&P 500` or `Extra` |
| `beta` | Covariance with `^GSPC` ÷ market variance, 5y monthly |
| `expected_return` | CAPM annual arithmetic return |
| `annual_variance` | Monthly return variance × 12, 10y window; Kelly denominator |
| `annual_volatility` | √`annual_variance` |
| `sharpe_capm` | (`expected_return` − Rf) ÷ `annual_volatility` |
| `hist_cagr` | Realized compound annual return over the data window |
| `expected_cagr` | `expected_return` − `annual_variance`/2 (CAPM return after volatility drag) |
| `cagr_gap` | `hist_cagr` − `expected_cagr` |
| `performance_category` | Bucketed `cagr_gap`: < −6% Strong underperformer, −6 to −2% Underperformer, ±2% In line, 2 to 6% Outperformer, > 6% Strong outperformer |
| `months_of_data` | Monthly returns used (60 minimum) |
| `full_kelly` | Standalone growth-optimal fraction; > 1 implies leverage |
| `half_kelly` / `quarter_kelly` / `eighth_kelly` | `full_kelly` ÷ 2 / 4 / 8 |
| `status` | `ok`, `insufficient history (N mo)`, or `no price data` |
| `risk_free_rate` | 10Y Treasury yield used for the run |
| `market_return` | Long-run market return assumption used for the run |
| `edge_source` | `views` if the Kelly columns come from your view, else `CAPM` |
| `p_win`, `upside`, `downside`, `horizon_years` | Your view, as fractions (views only) |
| `view_expected_return` | `p·upside − (1−p)·downside` over the horizon |
| `view_annualized_return` | The same, annualized |
| `capm_full_kelly` | CAPM Kelly for comparison (`full_kelly` holds the view-based figure when `edge_source = views`) |

## Output: `kelly_portfolio_YYYY-MM-DD.csv`
One row per basket name plus a `CASH` row. Each weight column sums to 1.

| Column | Definition |
|---|---|
| `weight_full_kelly` … `weight_eighth_kelly` | Optimal portfolio weight at that Kelly fraction |
| `standalone_full_kelly` | The same name's single-asset Kelly, for comparison |
| `edge_source`, `p_win`, `upside`, `downside` | Where each name's edge came from |
| `name`, `sector`, `universe`, `beta`, `annual_volatility`, `expected_return` | Carried from the main output |

## Output: `kelly_portfolio_summary_YYYY-MM-DD.csv`
One row per Kelly fraction: `engine` (`CAPM` or `views`), `horizon_years`, `invested`, `cash`, `holdings`, `expected_return` (annualized), `volatility`, `sharpe`, `expected_growth_rate` (annualized compound), `prob_loss` (chance of ending below the starting value over the horizon), `worst_5pct` (5th-percentile return over the horizon), `months_used`, `solver_ok`. CAPM mode reports `prob_loss`/`worst_5pct` over 1 year (lognormal approximation).

## Operational notes
- **FRED timeouts** are common from scripts (likely bot protection); the run falls back to Yahoo `^TNX`, which is the same 10Y yield. Not an error.
- **Locked output files**: if a CSV is open in Excel, the script writes a timestamped copy (`..._HHMMSS.csv`) instead of failing.
- **Yahoo on cloud hosts**: shared cloud IPs get rate-limited more than home IPs. If the hosted app starts returning no data, self-host it on a home machine/NAS instead.
- **Same-day reruns** overwrite that day's files unless they're locked.

## Caveats
- Standalone Kelly sizes each stock **in isolation**; that column won't sum to 100%. Use portfolio Kelly to size a basket.
- Under CAPM, Kelly reduces to `beta * MRP / variance`. Rankings reflect risk structure, not stock-specific alpha. Add views to size on conviction.
- **Views mode is only as good as your inputs.** Kelly sizes on the edge you give it, so overconfident `p_win` or understated `downside` produce oversized bets. Thorp's warning (p. 105): misjudging the downside or a Black Swan means the computed Kelly is too big. This is another reason to use half or quarter Kelly.
- Views share one horizon. Correlations between theses are borrowed from how the stocks have moved together, which is a reasonable default, not a certainty.
- `cagr_gap` is backward-looking. A large positive gap can mean a great business or a one-time re-rating.
- Extras (e.g. TSM, ASML ADRs) are priced in USD, so FX moves are baked into their returns. Beta is still measured against the S&P 500.
- Full Kelly is aggressive; half or quarter is the common practical choice.
- The covariance window reflects recent history only (≤10 years, often less for young names); a basket that thrived in one regime will look better than it would over a full cycle.
- Not financial advice.
