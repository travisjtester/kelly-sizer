# Kelly Sizer

[![Tests](https://github.com/travisjtester/kelly-sizer/actions/workflows/tests.yml/badge.svg)](https://github.com/travisjtester/kelly-sizer/actions/workflows/tests.yml) [![Live data check](https://github.com/travisjtester/kelly-sizer/actions/workflows/live-data.yml/badge.svg)](https://github.com/travisjtester/kelly-sizer/actions/workflows/live-data.yml)

Kelly Criterion position sizing, built to follow *Concentrated Investing* (Benello, van Biema & Carlisle, Wiley 2016), chapters 3–4: bet big when you hold an edge, don't bet when you don't, never risk ruin, and size the whole portfolio at once.

One engine, two ways in: a **phone app** for sizing on the couch and a **command line** for your desk. They take the same inputs, show the same results in the same words, and share the same two files: `views.csv` (your research) and `journal.csv` (your track record).

## The workflow
1. **Pick a basket**: 2–15 names you're considering.
2. **Add your research** (optional): for each name, a **bear** case (the *permanent* loss if you're wrong, and its odds), an optional **base** case, and a **bull** case, either entered directly or as your **intrinsic value** per share (margin of safety). Names without research use CAPM, which assumes no edge.
3. **Size it**: joint Kelly weights for the whole basket over a buy-and-hold horizon, at the Kelly fraction you choose. Results lead with the book's risk (chance of loss, 1-in-20 worst case), then growth, invested, cash, and price swings. A **What if I'm wrong?** check shows which bets depend on your bear case being right.
4. **Log it** to the conviction journal. When each horizon ends, record which scenario played out. The journal tells you whether your probabilities are calibrated, which decides whether you've earned bigger fractions.

## Five-minute walkthrough
On the [live app](https://kelly-sizer.streamlit.app), with the example file. Each step answers one question.

1. **What does no edge look like?** Keep the default basket and tap **Size it** with no research. Every name uses CAPM: small, spread-out weights and mostly cash. Note the invested % and expected growth. That's your baseline.
2. **What does your research change?** Download [`views.example.csv`](views.example.csv), then **Your research → Load views CSV**. KO joins the basket and NVDA, ASML and KO now carry views. Size it again. Everything that moved away from the baseline is the edge your views claim.
3. **Which bets depend on the bear case?** Read **What if I'm wrong?**. A **Fragile** name loses most of its size if the bear case is a little worse than you said.
4. **Does the market agree?** Open **Market check** and turn on the toggle. Your odds sit beside the option-implied odds. Every gap of 10+ points is a claim you should be able to explain.
5. **Is it the edge or the shape?** In **Settings**, turn on **Neutral inputs**. Your views keep their shape but lose their edge, and the sizes fall back toward the baseline. Turn it off again.
6. **Will you remember this call?** Under **Conviction journal**, tap **Log today's views**, then **Save journal CSV**. When each horizon ends, record what happened. After 10 resolved calls the journal tells you whether your odds are calibrated.

Same thing at a desk: `python kelly.py --views views.example.csv NVDA ASML KO MSFT VOO --market`.

## Phone app
```bash
python -m streamlit run app.py          # local: http://localhost:8501
```
Deployed free on Streamlit Community Cloud (share.streamlit.io → this repo → `app.py`). On iPhone: open the URL in Safari → Share → **Add to Home Screen**.

- Research table: Bear odds / Bear loss / Base odds / Base / Bull or Value $; bull odds, price and margin of safety fill in.
- Brand palette in `.streamlit/config.toml` (light and dark): teal `#379A8B` for primary and success, coffee `#D1B07C` for caution, red `#DB444B` for "What if I'm wrong?" and fragile bets, grey `#D3D3D3` for borders. Text shades meet 4.5:1 contrast in both modes.
- **Saved data**: with storage set up (below), your journal and views load when you unlock and save themselves as you go. Without it, **Save / Load** the CSVs (iPhone saves to Files) before you leave, since cloud sessions don't keep files. Either way the same files work on the command line.
- To preload, commit `views.csv` / `journal.csv` to the repo root (`git add -f`; `*.csv` is gitignored), but only if you're fine with them being as public as the repo.

### Saving your data (private GitHub repo)
The app's URL is public, so your data sits in a **private** repo and the app only touches it after you enter a passphrase. One-time setup, about five minutes:

1. **Create a private repo** on GitHub, e.g. `kelly-sizer-data` (empty is fine).
2. **Create a fine-grained token**: GitHub → Settings → Developer settings → Fine-grained tokens → Generate. Repository access: **only** `kelly-sizer-data`. Permissions → Repository → **Contents: Read and write**. Note the expiry date; you'll renew it.
3. **Add secrets**: on Streamlit Cloud, your app → Settings → Secrets. Paste (template in `.streamlit/secrets.toml.example`):
   ```toml
   [storage]
   repo = "travisjtester/kelly-sizer-data"
   token = "github_pat_..."
   passphrase = "a long phrase only you know"
   ```
4. **Open the app → Unlock** with your passphrase. Your views join the basket and your journal loads. From then on every change saves itself, one commit per save, so the repo's history records every edit.

- **Two devices at once:** journal entries from both are combined; for a view, the device that saves last wins.
- **On a computer:** clone `kelly-sizer-data` and run the CLI inside it (`python ../kelly-sizer/kelly.py`). It reads and writes the same `views.csv` and `journal.csv`; `git pull` before and `git push` after.
- **Security:** the token can only touch that one repo and never appears in this one (`.streamlit/secrets.toml` is gitignored). Wrong passphrases are slowed down and limited to 5 per session, so use a long one.

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
| Neutral inputs (test mode) | Settings | `--neutral` | off |
| Book mode (cash earns 0%) | Settings | `--book-mode` | off |
| Market check | Market check (after your odds) | `--market` | off |

**Neutral inputs** scales each view so its expected return equals its CAPM return while keeping its shape (the spread between bear, base and bull). With the edge removed, sizes should land near the no-views baseline; whatever gap remains is the scenario shape alone. Use it to test a set of inputs, such as analyst targets, not to size.

## Your views
```csv
ticker,p_bear,bear,p_base,base,bull,value,notes,as_of_price,horizon
NVDA,0.25,0.50,0.35,0.15,1.00,,"Bear: capex bust, permanent derating",185.00,3
ASML,0.20,0.35,0.40,0.20,,1100,"Bull from intrinsic value $1,100/share",,3
KO,0.10,0.15,,,0.30,,"Two outcomes only: base left blank",,
```

| Column | Meaning |
|---|---|
| `p_bear`, `bear` | Odds and size of the **permanent** loss if the thesis breaks (`0.5` = −50%). A broken thesis, not a price dip |
| `p_base`, `base` | Optional middle case: odds and return (can be negative) |
| `bull` | Gain if you're right (`1.0` = +100%) |
| `value` | Optional intrinsic value per share. If `bull` is blank, bull = value ÷ current price − 1; output shows **margin of safety** = 1 − price ÷ value |
| `notes` | Optional |
| `as_of_price` | Optional: the share price your returns were measured from. Scenario **prices** are held fixed and returns re-measured from today's price, so a stale view re-anchors itself (set at $100 with a 30% bear, the bear price is $70; at $90 that reads as a 22% bear). If the price has already run through your bear or bull price, the view falls back to CAPM with a note. Views typed in the app are stamped with the day's price |
| `horizon` | Optional: years the view assumes (`1` for 12-month targets). If it differs from the run's horizon you get a warning; returns aren't rescaled |

Bull odds = `1 − p_bear − p_base`. **Units:** use one style per file. The odds columns decide it: odds written as percents (`40`) mean the whole file is percents; decimals (`0.4`) mean decimals. A `%` sign always means percent. Older `p_win, upside, downside` files still load. Start from `views.example.csv`.


### Market check: your odds vs. the market's
Munger's winning bettors "look and sift the world for a mispriced bet" (pp. 209–210). Option prices are the market's odds; your edge is where your odds differ. **Market check** shows, for each name you have a view on, the option-implied chance of finishing below your bear price and above your bull price, beside your own odds, with a sentence wherever the gap is 10 points or more.

- **Hidden until you commit.** The app shows market odds only for views you've already entered, behind a toggle, so they can't anchor your estimate.
- **Never used in the sizing.** Copying the market's odds into your views would make your edge zero by construction (that's what neutral mode does on purpose).
- **How it's worked out:** the longest expiry at or under your horizon; volatility solved from each out-of-the-money option's bid/ask mid (not Yahoo's own figure), skipping quotes with no bid or a spread over 50% of the mid; a straight line between the two nearest strikes at your price; then N(−d2) below the bear price and N(d2) above the bull price. Prices outside the traded strikes say "no reliable quotes" instead of guessing.
- **Read it as a reference.** The odds are risk-neutral (a risk premium makes downside read a little high and upside a little low), they price a share price on a date rather than a broken thesis, and listed options rarely run past about 2 years. European pricing without dividends, so high-yield names read slightly off.

## How it sizes
1. Each name gets a distribution of horizon returns: your scenarios where you have a view; otherwise **CAPM** (horizon-matched Treasury + beta × market premium, lognormal with historical volatility).
2. 20,000 joint outcomes are drawn with a Student-t copula (5 degrees of freedom) on historical correlations, so correlated names tend to land on the same side, and bad outcomes land together more often than a Gaussian copula allows, as in a crash. Each name keeps its own odds. A pair with under 12 months of shared prices (a recent listing) gets the basket's average correlation rather than zero.
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
Page numbers are *Concentrated Investing*'s printed pages, checked against the text (November 2026).

### From the book
| The book (page) | In the tool |
|---|---|
| Kelly maximizes the expected log of wealth; edge/odds; no edge, no bet (pp. 75–76) | The objective everywhere; reproduces the book's example (51% at even money → 2%) |
| The formula: f* = (bp − q)/b, with idle cash earning nothing (pp. 75–76) | **Book mode** (`--book-mode`): cash at 0%, so a single bet's Kelly is exactly this formula |
| Kelly sizes from probability (chance of winning vs. losing) and risk vs. reward (p. 1) | Your odds and payoffs for each scenario |
| Concentration is only for those prepared to do intensive research (Preface, p. xi) | Views come from your research; names without one get no edge |
| "Know-nothing" investors should treat markets as efficient and index (p. 3) | No view → sized as plain market exposure under CAPM; `--screen` labeled as the no-edge baseline |
| Risk is the permanent loss of capital, not volatility (p. 3) | Bear = permanent loss; results lead with chance of loss; volatility labeled "price swings" |
| Margin of safety (pp. 3, 208) | `value` input → bull case and margin-of-safety column |
| The book's investors hold anywhere from 5 to 20 securities (p. 3) | Up to 15 names in the app |
| Kelly never risks ruin (p. 86) | Ruinous mixes score −∞; solo Kelly is bounded below the ruin point |
| Kelly is the maximum bet; half Kelly gives ¾ of the growth with ½ the volatility (pp. 85–86) | Fractions = c × the Kelly bet, then caps; default half |
| Buffett: 80% in five positions, "25 percent for the largest," once up to 40% (p. 96) | Default 25% per name; app ceiling 40%; CLI warns above it |
| Shannon didn't trim positions as they grew (pp. 81–82) | Buy-and-hold horizon; no rebalancing |
| Size bets jointly: Kelly in isolation "generally overestimates" (Thorp, p. 105) | One solve for the whole basket; solo Kelly shown as a ceiling |
| Misjudging the lower bound, or its odds, overstates the bet (Thorp, p. 105) | "What if I'm wrong?" on every result |
| Winning bettors "look and sift the world for a mispriced bet" (Munger, pp. 209–210) | Market check: your odds beside the market's, revealed after you enter yours; never used in sizing |

### The tool's extensions (not in the book)
The book gives the principles; it doesn't give a method for a multi-name, multi-outcome portfolio. These fill that gap.

| Extension | Why |
|---|---|
| Bear / base / bull scenarios instead of one win/lose bet | Lets bear mean a broken thesis while keeping a middle case. With two outcomes and cash at 0% it reduces to the book's formula |
| CAPM for names without a view | Lets an index or a no-view name sit in a basket as plain market exposure, with no edge |
| Joint sizing over 20,000 simulated outcomes (Student-t copula on historical correlations) | The book asks for joint sizing (p. 105) but gives no method. The t-copula makes bad outcomes cluster, as in a crash |
| Cash earns the Treasury yield matched to the horizon | Idle cash really does earn something; book mode turns this off |
| Fractional Kelly solved with your caps scaled by 1/c, then scaled down | Half Kelly is half the Kelly bet, trimmed to your caps, not half of an already-capped bet |
| The two stresses in "What if I'm wrong?" (+10 points of bear odds, 1.5× deeper bear) | The warning is Thorp's (p. 105); the stress sizes are the tool's |
| Conviction journal with calibration and a Brier score | The book says Kelly needs accurate odds; the journal checks yours |
| Market check's option math | The question is Munger's; reading option chains is the tool's |
| Stale-view re-anchoring, horizon flag, neutral inputs | Guardrails on your inputs |

## Files
| File | What it is |
|---|---|
| `kelly.py` | The engine and the command line |
| `app.py` | The phone app (same engine) |
| `views.csv`, `journal.csv` | Your research and track record (not committed; shared by app and CLI) |
| `github_store.py` | Saves your journal and views to a private GitHub repo (the app's saved data) |
| `views.example.csv` | A starting views file |
| `CHANGELOG.md` | What changed, by pull request |
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
**Testing.** `python -m pytest` runs the offline suite (fake Yahoo, about 6 minutes). GitHub Actions runs it on every pull request on Python 3.11 and 3.12, and `main` only accepts a PR once it passes, because `main` deploys straight to the live app. `python -m pytest smoke` runs the live data checks against the real sources; Actions runs them every Monday and emails the owner if one breaks.

Data sources: constituents from Wikipedia (GitHub dataset fallback); the Treasury yield whose maturity is nearest the horizon (1, 2, 3, 5, 7 or 10 years) from FRED (Yahoo `^FVX`/`^TNX`, then 4.5% fallback); prices from Yahoo via yfinance (monthly for correlations, daily for current prices). Update `MARKET_RETURN` in `kelly.py` once a year.

## Operational notes
- **FRED timeouts** are common from scripts; the fallback is the same 10Y yield.
- **Locked files**: if a CSV is open in Excel, a timestamped copy is written instead.
- **Yahoo on cloud hosts** gets rate-limited more than home IPs. If the hosted app returns no data, self-host on a home machine/NAS.
- **After a deploy** the app reloads the engine automatically when `kelly.py` changes.

## Caveats
- **Your inputs are the whole game.** Kelly sizes on the edge you give it: overconfident odds or an understated bear case produce oversized bets (Thorp, p. 105). Use half or quarter Kelly until the journal says your odds deserve more.
- Scenarios simplify a real distribution, and all views share one horizon (a view's `horizon` column flags a mismatch but can't give it its own).
- Cash earns a horizon-matched Treasury. The book's Kelly formula takes cash at zero, so this is the tool's extension.
- Correlations between theses come from how the stocks' prices moved together. For idiosyncratic theses that likely overstates the link; raise the shrinkage.
- Extras (e.g. TSM, ASML ADRs) are priced in USD, so FX is in their returns.
- Not financial advice.
