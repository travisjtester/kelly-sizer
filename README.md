# Kelly Criterion S&P 500 Screener

Computes CAPM-based Kelly allocations for every S&P 500 constituent and writes a dated CSV.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

## Run
```bash
python kelly_sp500.py                         # full index -> Desktop/kelly_sp500_YYYY-MM-DD.csv
python kelly_sp500.py --limit 25              # quick smoke test
python kelly_sp500.py --tickers AAPL MSFT KO  # specific names
python kelly_sp500.py --out data/run.csv --market-return 0.10
```

## Method
| Input | Source | Fallback |
|---|---|---|
| Constituents | Wikipedia | GitHub `datasets/s-and-p-500-companies` |
| Risk-free rate | FRED DGS10 (10Y Treasury) | Yahoo `^TNX`, then 4.5% |
| Prices | Yahoo via yfinance, monthly, dividend/split-adjusted | 3 retries with backoff |

- **Beta**: regressed vs `^GSPC`, 5y monthly returns
- **Expected return**: CAPM, `Rf + beta * (MARKET_RETURN - Rf)`
- **Variance**: 10y monthly returns, annualized x12; requires 60+ months
- **Kelly**: `(E[R] - Rf) / variance`, plus 1/2, 1/4, 1/8 fractions

Update `MARKET_RETURN` once a year (link in the script).

## Caveats
- Each stock is sized **in isolation**. The column won't sum to 100%; it ignores correlation between holdings.
- Under CAPM, Kelly collapses to `beta * MRP / variance`, so ranking mostly reflects beta-to-variance, not stock-specific alpha.
- Full Kelly is aggressive; half or quarter is the common practical choice.
- Not financial advice.
