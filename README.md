# Kelly Sizer

Kelly Criterion position sizing, built to follow *Concentrated Investing* (Benello, van Biema & Carlisle, Wiley 2016), chapters 3–4: bet big when you hold an edge, don't bet when you don't, never risk ruin, and size the whole portfolio at once.

One engine, two ways in: a **phone app** for sizing on the couch and a **command line** for your desk. They take the same inputs, show the same results in the same words, and share the same two files: `views.csv` (your research) and `journal.csv` (your track record).

## The workflow
1. **Pick a basket**: 2–15 names you're considering.
2. **Add your research** (optional): for each name, a **bear** case (the *permanent* loss if you're wrong, and its odds), an optional **base** case, and a **bull** case, either entered directly or as your **intrinsic value** per share (margin of safety). Names without research use CAPM, which assumes no edge.
3. **Size it**: joint Kelly weights for the whole basket over a buy-and-hold horizon, at the Kelly fraction you choose. Results lead with the book's risk (chance of loss, 1-in-20 worst case), then growth, invested, cash, and price swings. A **What if I'm wrong?** check shows which bets depend on your bear case being right.
4. **Log it** to the conviction journal. When each horizon ends, record which scenario played out. The journal tells you whether your probabilities are calibrated, which decides whether you've earned bigger fractions.

## Phone app
```bash
python -m streamlit run app.py          # local: http://localhost:8501
```
Deployed free on Streamlit Community Cloud (share.streamlit.io → this repo → `app.py`). On iPhone: open the URL in Safari → Share → **Add to Home Screen**.

- Research table: Bear odds / Bear loss / Base odds / Base / Bull or Value $; bull odds, price and margin of safety fill in.
- Brand palette in `.streamlit/config.toml` (light and dark): teal `#379A8B` for primary and success, coffee `#D1B07C` for caution, red `#DB444B` for "What if I'm wrong?" and fragile bets, grey `#D3D3D3` for borders. Text shades meet 4.5:1 contrast in both modes.
- **Save / Load** views and journal CSVs (iPhone saves to Files). Cloud sessions don't keep files, so save before you leave. The same files work on the command line.
- To preload, commit `views.csv` / `journal.csv` to the repo root (`git add -f`; `*.csv` is gitignored), but only if you're fine with them being as public as the repo.

## Command line
```bash
python -m venv .venv
.venv\Scripts\activate                  # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```
```bash
python kelly.py NVDA ASML KO MSFT VOO                  # size this basket (half Kelly, $10,000)
python kelly.py NVDA ASML KO --fraction quarter --amount 25000 --horizon 5
python kelly.py                                        # size the names in views.csv
python kelly.py NVDA ASML KO --log                     # ...and log today's views to journal.csv
python kelly.py --report                               # conviction journal only
python kelly.py --screen --extra TSM ASML              # opt-in: S&P 500 no-edge baseline
```
`views.csv` and `journal.csv` in the current folder are picked up automatically (`--views` / `--journal` to point elsewhere). Results print in the app's layout and save to your Desktop (`--out-dir`, or `--no-save`). Windows: quote paths with spaces; use `py` if `python` isn't recognized.

| Setting | App | CLI | Default |
|---|---|---|---|
| Kelly fraction | Kelly fraction | `--fraction` | half |
| Amount | Amount to allocate | `--amount` | $10,000 |
| Holding horizon | Settings | `--horizon` | 3 years |
| Max per name | Settings (5–40%) | `--max-weight` (warns above 40%) | 25% |
| Max invested | Settings | `--max-invested` | 1.0 (no margin) |
| Correlation shrinkage | Settings | `--shrink` | 0.20 |
| Long-run market return (CAPM) | Settings | `--market-return` | 10.68% |

## Your views
```csv
ticker,p_bear,bear,p_base,base,bull,value,notes
NVDA,0.25,0.50,0.35,0.15,1.00,,"Bear: capex bust, permanent derating"
ASML,0.20,0.35,0.40,0.20,,1100,"Bull from intrinsic value $1,100/share"
KO,0.10,0.15,,,0.30,,"Two outcomes only: base left blank"
```

| Column | Meaning |
|---|---|
| `p_bear`, `bear` | Odds and size of the **permanent** loss if the thesis breaks (`0.5` = −50%). A broken thesis, not a price dip |
| `p_base`, `base` | Optional middle case: odds and return (can be negative) |
| `bull` | Gain if you're right (`1.0` = +100%) |
| `value` | Optional intrinsic value per share. If `bull` is blank, bull = value ÷ current price − 1; output shows **margin of safety** = 1 − price ÷ value |
| `notes` | Optional |

Bull odds = `1 − p_bear − p_base`. **Units:** use one style per file. The odds columns decide it: odds written as percents (`40`) mean the whole file is percents; decimals (`0.4`) mean decimals. A `%` sign always means percent. Older `p_win, upside, downside` files still load. Start from `views.example.csv`.

## How it sizes
1. Each name gets a distribution of horizon returns: your scenarios where you have a view; otherwise **CAPM** (10Y Treasury + beta × market premium, lognormal with historical volatility).
2. 20,000 joint outcomes are drawn with a Gaussian copula on historical correlations, so correlated names tend to land on the same side.
3. Weights maximize **expected log wealth** at the horizon, long-only, within your caps; the rest earns the risk-free rate. A mix that could wipe out the portfolio scores −∞, so the result **never risks ruin**.
4. **Fractional Kelly** c = bet c × what Kelly says, *then* apply your caps.
5. **Solo Kelly** per name is the same calculation for that bet alone: a ceiling, not a target.

Buy and hold, no rebalancing assumption. Seeds are fixed, so reruns are identical.

## What if I'm wrong?
Kelly is only as good as your bear case. A small miss moves the *edge* a lot, because edge is the small gap between your expected return and cash. Example: a view at 25% / −50% bear, 35% / +15% base, 40% / +100% bull over 3 years has an 18.6-point edge over cash. Make the bear case 35% likely and the edge falls to 3.6 points, and half Kelly drops from 30% to 6%. That's Thorp's warning (p. 105): misjudging the downside means the computed bet is too big.

Every result with views includes this check, in the app and on the command line:
- **Re-sized if you're wrong:** the basket sized again with every bear case **10 points more likely** (taken from the bull case), and again with every bear loss **1.5× deeper**.
- **A verdict per name:** **Robust** (keeps 75%+ of its size under both), **Sensitive** (40–75%), or **Fragile** (under 40%).
- **What being wrong costs:** your basket *as sized*, scored as if each stress were the truth (chance of loss, 1-in-20 worst case, growth).

A Fragile name is one where your conviction is carrying the bet. Recheck the bear case, or size it at quarter Kelly.

## Conviction journal
Each logged view records the date, entry price and your scenarios. When the horizon ends it shows as **due**; record `bear`, `base` or `bull` (app: dropdown; CLI: the `outcome` column). Judge the thesis, not the price. You get:
- **calibration**: for the bull odds you gave (<40%, 40–60%, 60–80%, 80%+), how often bull happened
- **Brier score** (0 = perfect) and the average probability you gave to what actually happened
- a verdict after 10+ resolved calls: well calibrated, running hot (lower your odds or bet a smaller fraction), or underconfident

## The no-edge baseline (`--screen`)
Solo Kelly for every S&P 500 name under CAPM (plus `--extra` names and your views), saved to `kelly_screen_YYYY-MM-DD.csv`. CAPM assumes no edge, so this ranks **risk structure, not opportunities**. It's a reference point for comparing your views, not a source of ideas. The book's ideas come from research.

## How the book maps to the tool

| Book (page) | Tool |
|---|---|
| Kelly maximizes expected log of wealth; edge/odds; no edge, no bet (pp. 75–76) | The objective everywhere; reproduces the book's example (51% at even money → 2%) |
| Kelly never risks ruin (p. 86) | Ruinous mixes score −∞; solo Kelly is bounded below the ruin point |
| Edge from research on probability and risk/reward (p. 2) | Your bear / base / bull scenarios |
| Risk = permanent loss of capital, not volatility (p. 3) | Bear = permanent loss; results lead with chance of loss; volatility labeled "price swings" |
| Margin of safety (pp. 3, 208) | `value` input → bull case and margin-of-safety column |
| Size bets jointly; isolated Kelly "generally overestimates" (p. 105) | One solve for the whole basket; solo Kelly shown as a ceiling |
| Kelly is the maximum bet; half Kelly ≈ ¾ of the growth, ½ the volatility (pp. 85–86) | Fractions = c × the Kelly bet, then caps; default half |
| Buffett: "25 percent for the largest," up to 40% (p. 96) | Default 25%; app ceiling 40%; CLI warns above it |
| Concentrate in 5–20 names (p. 3) | Up to 15 in the app |
| Buy and hold; Shannon didn't trim (p. 82) | Buy-and-hold horizon |
| Kelly needs accurate probabilities; misjudged downside overstates the bet (p. 105) | "What if I'm wrong?" stress test on every result, plus the conviction journal's calibration over time |
| "Know-nothing" investors should index (p. 3) | No view → sized as market exposure under CAPM; `--screen` labeled as the no-edge baseline |

## Files
| File | What it is |
|---|---|
| `kelly.py` | The engine and the command line |
| `app.py` | The phone app (same engine) |
| `views.csv`, `journal.csv` | Your research and track record (not committed; shared by app and CLI) |
| `views.example.csv` | A starting views file |
| `tests/` | Offline test suite (fake Yahoo data): book math, views, what-if-wrong, journal, CLI, app |

**Outputs** (decimals: 0.12 = 12%):
- `kelly_portfolio_YYYY-MM-DD.csv`: one row per name plus `CASH`, with `weight_full_kelly` … `weight_eighth_kelly` (each sums to 1), `edge_source`, `standalone_full_kelly`, your view columns, `margin_of_safety`. The app's **Save these results** button writes the same file.
- `kelly_portfolio_summary_YYYY-MM-DD.csv`: one row per fraction: `views_used`, `basket_size`, `months_used`, `horizon_years`, `invested`, `cash`, `holdings`, `prob_loss`, `worst_5pct` (over the horizon), `expected_growth_rate`, `expected_return`, `volatility` (annualized), `solver_ok`.
- `kelly_screen_YYYY-MM-DD.csv` (`--screen` only): per-name `beta`, `expected_return`, `annual_volatility`, `hist_cagr`, `full_kelly` … `eighth_kelly`, `capm_full_kelly`, view columns, `status`.

## Development
```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest            # ~4 minutes, offline
```
Run the tests before merging a branch. Data sources: constituents from Wikipedia (GitHub dataset fallback); 10Y Treasury from FRED (Yahoo `^TNX`, then 4.5% fallback); prices from Yahoo via yfinance (monthly for correlations, daily for current prices). Update `MARKET_RETURN` in `kelly.py` once a year.

## Operational notes
- **FRED timeouts** are common from scripts; the fallback is the same 10Y yield.
- **Locked files**: if a CSV is open in Excel, a timestamped copy is written instead.
- **Yahoo on cloud hosts** gets rate-limited more than home IPs. If the hosted app returns no data, self-host on a home machine/NAS.
- **After a deploy** the app reloads the engine automatically when `kelly.py` changes.

## Caveats
- **Your inputs are the whole game.** Kelly sizes on the edge you give it: overconfident odds or an understated bear case produce oversized bets (Thorp, p. 105). Use half or quarter Kelly until the journal says your odds deserve more.
- Scenarios simplify a real distribution, and all views share one horizon.
- Correlations between theses come from how the stocks' prices moved together. For idiosyncratic theses that likely overstates the link; raise the shrinkage.
- Extras (e.g. TSM, ASML ADRs) are priced in USD, so FX is in their returns.
- Not financial advice.
