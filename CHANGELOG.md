# Changelog

What changed, newest first, by pull request. Everything merged to `main` deploys to the [live app](https://kelly-sizer.streamlit.app).

## 2026-11-11
- **Street starter, screenshots, table fix** (#27): blank rows can start from analysts' low / mean / high 12-month targets, with placeholder odds, `horizon = 1` and `as_of_price`, labeled as consensus. App button in Your research; CLI `--street` (saved to `street_views_DATE.csv`). Never overwrites a view you wrote. README walkthrough now has screenshots of the live app. Fix: the research table's computed columns (bull odds, price, margin of safety) lagged one edit behind.
- **License** (#26): MIT. README notes the book's text isn't reproduced here, and that the tool isn't financial advice.
- **Dependabot** (#25): pip minimums are raised only when they block a newer version, keeping Python 3.11 supported.
- **Saved data** (#24): with storage set up in Streamlit secrets, the app loads your journal and views from a private GitHub repo once you enter your passphrase, and saves every change as you go (one commit each). Journal entries from two devices merge. The CLI uses the same files from a clone of that repo. Without secrets nothing changes. Also corrects the in-app method note, which still said CAPM used the 10-year Treasury.
- **You vs. the market** (#23): each journal entry records the market's odds beside yours (new columns `market_bear`, `market_bull`, `market_expiry`; older journals still load). After 10+ resolved calls with market odds, the journal scores both on the same calls and says whether your edge looks real. Fixes the calibration table showing 60% as "1%".
- **Book and docs** (#19): two page citations corrected against the book's text (probability and risk vs. reward is p. 1; Shannon's no-trimming is pp. 81–82). The README's book table is split into what comes from the book and the tool's extensions. **Book mode** (`--book-mode`, app Settings): cash earns 0%, so a single bet's Kelly is exactly the book's f* = (bp − q)/b. Five-minute walkthrough, this changelog, and `views.example.csv` with the `as_of_price` and `horizon` columns.

## 2026-11-10
- **Market check** (#17): option-implied odds of finishing below your bear price and above your bull price, beside your own odds, with a question wherever the gap is 10+ points. Hidden until you've entered your odds; never used in the sizing. CLI `--market`.
- **Engine correctness** (#16): a Student-t copula, so bad outcomes cluster as in a crash. Recent listings get the basket's average correlation instead of zero. Tests show expected growth is stable across random seeds and scenario counts.
- **CI and live data checks** (#15): the offline suite runs on every PR (Python 3.11 and 3.12), and `main` only accepts a PR once it passes. A weekly live check hits the real data sources. Its first run found FRED timing out from cloud hosts (the Treasury's own yield curve is now the primary source) and yfinance returning tickers as all-blank without raising (blank tickers are now fetched again).
- **Stale views, horizon, risk-free, neutral mode** (#14): optional `as_of_price` re-anchors a view to today's price with its scenario prices held fixed; optional `horizon` flags a mismatch; cash earns the Treasury yield matched to the horizon; **Neutral inputs** scales each view to its CAPM return to test inputs; view warnings show as alerts.
- **Input checks and metadata** (#13): a warning when a file's bear column mixes signs; the CLI carries name and sector; sectors use GICS names; loading a views CSV adds its tickers to the basket.
- **Export fixes** (#12): every CSV export uses fractions (0.119, not 11.93); names outside the S&P 500 get a name and sector.

## 2026-10-19
- **Brand and solver** (#11): teal, coffee and red palette; a smooth ruin penalty fixes solver stalls at fractional Kelly; phone table polish.

## 2026-10-18
- **"What if I'm wrong?"** (#10): two stresses show which bets depend on the bear case being right (Thorp, p. 105).
- **One product** (#9): Kelly Sizer, one engine shared by the app, the CLI and the tests.
- **Buffett's ceiling** (#8): per-name weight capped at 40%.
- **Closer to the book** (#6, #7): bear/base/bull scenarios, intrinsic value and margin of safety, the conviction journal; risk framed as chance of loss with volatility as "price swings"; backtests, CAGR comparisons, Sharpe ratios and the rebalancing assumption removed as unfaithful to the book.
- **Views** (#4, #5): size on your own edge; the app reloads the engine after a deploy.

## 2026-10-14
- **First version** (#1–#3): portfolio Kelly sizing, the Streamlit phone app, and the theme.
