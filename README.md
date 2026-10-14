# Kelly Criterion S&P 500 Screener

Computes CAPM-based Kelly allocations for every S&P 500 constituent (plus any extra tickers) and writes a dated CSV. Optionally solves joint Kelly weights for a basket, accounting for correlation between holdings.

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

## Output: `kelly_portfolio_YYYY-MM-DD.csv`
One row per basket name plus a `CASH` row. Each weight column sums to 1.

| Column | Definition |
|---|---|
| `weight_full_kelly` … `weight_eighth_kelly` | Optimal portfolio weight at that Kelly fraction |
| `standalone_full_kelly` | The same name's single-asset Kelly, for comparison |
| `name`, `sector`, `universe`, `beta`, `annual_volatility`, `expected_return` | Carried from the main output |

## Output: `kelly_portfolio_summary_YYYY-MM-DD.csv`
One row per Kelly fraction: `invested`, `cash`, `holdings`, `expected_return`, `volatility`, `sharpe`, `expected_growth_rate` (≈ return − variance/2), `months_used`, `solver_ok`.

## Operational notes
- **FRED timeouts** are common from scripts (likely bot protection); the run falls back to Yahoo `^TNX`, which is the same 10Y yield. Not an error.
- **Locked output files**: if a CSV is open in Excel, the script writes a timestamped copy (`..._HHMMSS.csv`) instead of failing.
- **Same-day reruns** overwrite that day's files unless they're locked.

## Caveats
- Standalone Kelly sizes each stock **in isolation**; that column won't sum to 100%. Use portfolio Kelly to size a basket.
- Under CAPM, Kelly reduces to `beta * MRP / variance`. Rankings reflect risk structure, not stock-specific alpha.
- `cagr_gap` is backward-looking. A large positive gap can mean a great business or a one-time re-rating.
- Extras (e.g. TSM, ASML ADRs) are priced in USD, so FX moves are baked into their returns. Beta is still measured against the S&P 500.
- Full Kelly is aggressive; half or quarter is the common practical choice.
- The covariance window reflects recent history only (≤10 years, often less for young names); a basket that thrived in one regime will look better than it would over a full cycle.
- Not financial advice.
