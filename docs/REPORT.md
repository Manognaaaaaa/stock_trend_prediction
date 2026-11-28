# Ranking Saudi Stocks with Machine Learning: Draft Report

**Oil-Aware Model Switching and Failure Warning on the Full Tadawul Market**
Design project, BITS Pilani Dubai Campus. Author: Manogna Raju. Advisor: Dr. Suyash Shukla.

> Draft generated from the files in `results/`. Every number below comes from a CSV in
> `results/tables/` written by the pipeline (`python -m src.evaluate` and the steps before it);
> none comes from a notebook. Rerun the pipeline and update the numbers before submitting.

---

## 1. Dataset (Step 1)

| Item | Value | Source |
| --- | --- | --- |
| Stock universe | 273 listed and 21 delisted Main Market companies (saudiexchange.sa, 27 Nov 2026); 19 REITs excluded | `data/reference/universe.csv` |
| Stocks with Yahoo prices | 251 of 254 listed non-REIT (3 listed after 2025); **0 of 21 delisted** | `dataset_summary.csv` |
| Stock price history | 2010-02-22 to 2025-12-31 (Yahoo has no earlier Saudi stock data) | `dataset_summary.csv` |
| TASI | 2008-01-01 to 2025-12-31, 4,496 days, from Investing.com (Yahoo has no TASI history) | `dataset_summary.csv` |
| Brent (BZ=F), VIX | 2008-01-02 to 2025-12-31 | `dataset_summary.csv` |
| Clean stock-days | 616,956 of 660,532 | `dataset_summary.csv` |
| Weekly panel | 135,066 stock-weeks, 127,915 eligible; 106 eligible stocks per week in 2010 rising to 239 in 2025 | `dataset_summary.csv` |

**Cleaning rules and rows removed.** Missing price 78; not a TASI trading day 18,782; zero volume
24,716; corporate-action flags 840 (return set to missing, not dropped); fewer than 3 trading days
in the week 4,009 stock-weeks; fewer than 60 days of history 3,142 stock-weeks.

**Calendar.** Weeks are W-FRI periods ending on the last TASI trading day, which handles the
June 2013 change from a Saturday-Wednesday to a Sunday-Thursday week. Yahoo's Saudi dates before
2013-06-29 do not line up with TASI (no Saturday sessions, shifting day offsets), so those rows are
kept and flagged `date_suspect` (81,244 stock-days, 18,916 eligible weeks). Half of the 840
corporate-action flags fall in that period and are likely date artefacts.

**Market value.** Dated share counts cover 79.8% of stock-days; the rest use today's share count,
which is a look-ahead and is reported as a limitation.

## 2. Features, target and folds (Step 2)

- **13 stock features** on each stock's own trading rows: momentum (1 week to 12 months, skipping
  the latest month for 12 months), 20- and 60-day volatility, 120-day betas to Brent and TASI,
  turnover, Amihud illiquidity, volume change and log market value. Ranked to [-1, 1] each week.
- **Target:** next-week return (week-end close to next week-end close) minus the stock's sector
  mean; sectors with fewer than 5 stocks are pooled by broad sector or into "Other"
  (`sector_mapping_summary.csv`).
- **Folds:** expanding window, 12 test years 2014-2025, validation on the previous year, a
  trading-day embargo between blocks (`data/processed/folds.json`).
- **Leakage tests:** all five pass on synthetic and real data (shift test, target timing, fold
  overlap and embargo, oil/VIX dating, random labels). Single-feature mean weekly IC is at most
  |0.032| (60-day volatility), well under the 0.10 leakage alarm.

## 3. Backtester (Step 3)

Top 10% by score, equal weight, long only, stocks at the +10% limit or not traded on the
rebalance day excluded, 25 bps one-way cost on turnover. Sanity checks (`backtest_sanity.csv`):
random scores lose about the cost (-0.40% a week against a 0.45% cost), perfect foresight gives a
Sharpe of 11.8, 12-month momentum a Sharpe of 0.16.

## 4. Results, Table 1 (test weeks 2014-2025, net of 25 bps)

From `results/tables/table1.csv`:

| Method | Mean Rank IC | IC t-stat | Precision top 10% | Annual return | Sharpe | Max drawdown | Turnover |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ridge | 0.0285 | 5.9 | 0.069 | -11.6% | -0.62 | -81.6% | 0.92 |
| LightGBM | 0.0294 | 7.2 | 0.088 | -10.4% | -0.48 | -76.9% | 1.16 |
| MLP | 0.0355 | 7.7 | 0.074 | -9.5% | -0.45 | -76.1% | 0.93 |
| Simple average | 0.0345 | 7.4 | 0.070 | -9.6% | -0.48 | -76.6% | 0.94 |
| **Oil-aware selector** | 0.0337 | 7.4 | 0.072 | -9.8% | -0.47 | -77.1% | 0.98 |
| Selector (logistic) | 0.0338 | 7.4 | 0.072 | -10.1% | -0.50 | -77.9% | 0.96 |
| Selector without oil | 0.0329 | 7.2 | 0.070 | -10.0% | -0.50 | -77.9% | 0.97 |
| Hard switch | 0.0282 | 6.3 | 0.079 | -12.0% | -0.61 | -82.4% | 1.18 |
| Recent-IC weights | 0.0338 | 7.3 | 0.073 | -10.1% | -0.51 | -78.1% | 0.97 |
| Frequent retraining | 0.0373 | 8.0 | 0.074 | -10.9% | -0.55 | -80.0% | 0.97 |
| Selector + warning | 0.0337 | 7.4 | 0.072 | -11.6% | -0.69 | -81.9% | 0.87 |
| Equal-weight universe | | | | +1.5% | 0.17 | -52.5% | |
| TASI buy and hold | | | | +1.8% | 0.19 | -50.6% | |

**RQ1 (do ML models beat simple benchmarks after costs?).** No. Every model ranks stocks with
clear, significant skill (weekly Rank IC 0.028-0.037, t-statistics 6-8), but the top decile beats
its sector by only about 3-7 bps a week, while weekly turnover of 0.9-1.2 costs about 23-29 bps a
week at 25 bps. Gross of costs the portfolios roughly match the equal-weight universe; net of costs
they lose about 10% a year. At 10 bps the best methods are close to zero (LightGBM Sharpe -0.00,
selector -0.05; `robustness.csv`).

**RQ2 (does oil-aware switching beat single models and the simple average?).** Not meaningfully.
Selector vs simple average: mean IC difference -0.0008 (Diebold-Mariano p = 0.07, in favour of the
simple average), Sharpe difference +0.001 with a 95% block-bootstrap interval of -0.07 to +0.07
(`significance.csv`). The selector beats Ridge on IC (p = 0.003) but not LightGBM (p = 0.06) or the
MLP. Removing the oil inputs lowers IC slightly (difference 0.0007, p = 0.03) with no clear Sharpe
effect (interval -0.02 to +0.09). The selector leans towards LightGBM, most of all in the panic
regime (weight 0.44 vs 0.38 in calm weeks; `regime_selector_weights.csv`).

**RQ3 (can confidence and disagreement signal when to invest less?).** Not in this setup. The
bad-week classifier has an AUROC of 0.53 (`warning_metrics.csv`). At the main 20% threshold the
rule invests 84% on average and lowers volatility from 18.1% to 15.9%, but annual return falls
from -9.8% to -11.6%, the drawdown is slightly deeper, and the Sharpe difference vs the selector is
-0.22 (interval -0.42 to -0.04). Changing exposure itself adds turnover cost. All three thresholds
(10%, 20%, 30%) give the same picture (`warning_sensitivity.csv`).

## 5. Crisis windows and regimes

Cumulative net return inside each window (`crisis_windows.csv`):

| Window | Selector | Simple average | Selector + warning | Equal-weight | TASI |
| --- | --- | --- | --- | --- | --- |
| Oil crash 2014-16 | -35.2% | -34.8% | -35.8% | -36.7% | -37.0% |
| MSCI inclusion 2019 | -2.0% | -1.3% | -1.2% | +0.8% | -11.4% |
| COVID 2020 | -11.6% | -8.6% | -13.8% | -11.3% | -10.5% |
| Rate hikes 2022 | -29.8% | -27.9% | -30.0% | -25.9% | -17.4% |

GMM regimes over the test weeks (`regime_summary.csv`): calm 388 weeks, oil stress 202 weeks,
panic 23 weeks (mean VIX 34, Brent down 23% over the month). All model portfolios have negative
Sharpe in calm (about -0.8 to -1.0) and oil-stress weeks (about -0.4 to -0.7), while the benchmarks
are near zero or positive there (equal-weight -0.02 and 0.18, TASI 0.15 and 0.06). In the few
panic weeks everything has a positive Sharpe; the MLP (2.07) and simple average (1.76) are highest
(`regime_results.csv`).

## 6. Discussion

- **Ranking skill is real but too small to trade weekly.** The models rank the cross-section
  consistently (Rank IC t > 5 in every version), yet the gain is concentrated in ordering the middle
  of the distribution, not in the top decile's mean return. Weekly rebalancing replaces most of the
  portfolio each week, and costs dominate. Natural next steps: a longer holding period, a turnover
  penalty or a no-trade band, and scoring on the top-decile return rather than on Rank IC.
- **Deep vs simple models.** The MLP has the highest single-model IC (0.036) and the best net
  Sharpe of the three; Ridge is weakest. Frequent retraining gives the highest IC of all (0.037)
  but no better portfolio.
- **Model switching.** With three models whose rankings are highly correlated, a learned selector
  adds little over a simple average; the hard switch is clearly worse.
- **Warning rule.** The bad-week classifier is close to random, and the warning reduces volatility
  at the cost of return.
- **Liquid-only and without-largest results** are similar or slightly worse (`robustness.csv`), so
  the losses do not come from illiquid names alone.

## 7. Limitations

- **Survivorship:** all 21 delisted companies are missing from Yahoo, so the universe contains only
  survivors. LSEG or Bloomberg access would fix this.
- **Pre-2013 dates:** Yahoo's Saudi dates before 2013-06-29 do not match TASI; those rows are
  flagged but kept in training (they fall only in training blocks, not in any test year).
- **Share counts:** 20% of stock-days use today's share count (look-ahead in turnover and size).
- **MASTER not trained:** the data loader and Colab notebook are built (`src/models/master_adapter.py`,
  `notebooks/master_colab.ipynb`), but the official code and a GPU were not available here. The MLP
  is the deep learning model in all results.
- **Few selector rows:** about 50 weeks a year, 594 in total; the selector and warning classifiers
  use equal weights / no signal until 52 past weeks exist (2014-2015).
- **Price-only features:** no fundamentals, news or analyst data.
- **Costs:** a flat 25 bps one-way; real Tadawul costs vary by stock size and liquidity.

## 8. Changes from the midsem report (state openly)

- Data start: stocks from February 2010 (Yahoo's earliest), first test year 2014.
- Equal-weight universe added as a benchmark next to TASI.
- A separate bad-week classifier for the warning rule.
- A logistic-regression selector as a check on the LightGBM selector.
- MASTER inputs reduced to 13 stock features and 30 market features (if MASTER is trained).

## 9. Open questions for Dr. Shukla

- Is long-only evaluation acceptable, given that the ranking skill sits mostly below the top decile?
- Is price-only data enough without fundamentals?
- Should the main result use a longer holding period or a turnover penalty, given that costs
  decide the outcome?
