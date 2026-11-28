# Ranking Saudi Stocks with Machine Learning

**Oil-Aware Model Switching and Failure Warning on the Full Tadawul Market**

Design project, BITS Pilani Dubai Campus
Author: Manogna Raju
Faculty advisor: Dr. Suyash Shukla

---

## What this project does

Every week, this system ranks all stocks on the Saudi Exchange (Tadawul) main market by how likely they are to beat other stocks in their own sector over the next week. It then buys the top 10% (long only) and holds them for one week.

Three standard models produce the rankings:

| Model | Why it is included |
| --- | --- |
| Ridge regression | Simple, stable linear patterns. Tends to work in calm markets. |
| LightGBM | Non-linear patterns, for example momentum that matters only when volatility is low. |
| Small neural network (MLP) | Combinations of many features at once. The deep learning part of the main method. |

A fourth model, **MASTER** (AAAI 2024), is trained as a deep learning benchmark.

The project's own contribution sits on top of these models:

1. **Oil-aware selector.** A small LightGBM classifier looks at oil and market conditions at the start of each week (Brent return and volatility, TASI volatility, VIX, recent model accuracy). It decides how much to trust each model that week. The three rankings are blended using those trust levels.
2. **Warning rule.** If the system expects every model to struggle, or the models disagree strongly, it invests less and holds the rest in cash.

## Research questions

- **RQ1:** Do ML and deep learning models beat simple benchmarks when ranking stocks on the full Saudi market after trading costs?
- **RQ2:** Does switching between models based on oil and market conditions beat any single model and a simple average of the models?
- **RQ3:** Can selector confidence and model disagreement signal when to invest less, and does acting on that signal reduce losses during market shocks?

## Why the Saudi market

- Deep learning stock rankers (MASTER, StockMixer, PDU) have only been tested on large US and Chinese markets with thousands of stocks. Tadawul has a few hundred.
- Saudi studies so far cover only the TASI index or about 11 large stocks, never the full main market.
- Oil drives the Saudi market, yet no model switching method uses oil.
- The market went through clear shocks in the test period: the 2014 to 2016 oil crash, the 2019 MSCI inclusion, the 2020 COVID and oil price crash, and the 2022 rate hikes.

---

## Pipeline

```
 Raw data (Tadawul stocks, TASI, Brent, VIX)
        |
        v
 Clean + weekly panel  -->  Features (13 per stock)  +  Sector-relative target
        |
        v
 Walk-forward folds (train up to T-2, tune on T-1, test on T, 5-day embargo)
        |
        v
 Base models: Ridge | LightGBM | MLP (5 seeds)      MASTER (benchmark)
        |                                              |
        v                                              |
 Saved out-of-sample predictions  <--------------------+
        |
        v
 Oil-aware selector (LightGBM, small)  -->  Weekly model weights  -->  Blended ranking
        |
        v
 Warning rule (bad-week classifier + disagreement)  -->  Exposure 100% / 50% / 0%
        |
        v
 Backtest (top 10%, equal weight, long only, after costs)
        |
        v
 Evaluation (metrics, crisis windows, regimes, significance)  -->  Report + Streamlit demo
```

---

## Data

| Data | Source | yfinance ticker |
| --- | --- | --- |
| Daily price and volume, all main market stocks | Yahoo Finance, Saudi Exchange | `XXXX.SR` (for example `2222.SR`, `1120.SR`) |
| TASI index | Yahoo Finance | `^TASI.SR` |
| Brent crude oil | Yahoo Finance | `BZ=F` |
| VIX | Yahoo Finance | `^VIX` |
| Sector (industry group) | Saudi Exchange | n/a |
| Delisted and suspended companies | Saudi Exchange, Issuers Trading Information page | n/a |
| Delisted stock prices (if Yahoo lacks them) | LSEG or Bloomberg via university library | n/a |

Period: as early as Yahoo allows (target 2008) to 2025. If data starts only in 2012, the selector's test period starts around 2018. See `docs/STEPS.md`, Step 1.

Raw data is **not** committed to the repo. Run the download script to recreate it.

### Tadawul facts the code depends on

| Fact | Effect on the code |
| --- | --- |
| Trading days are Sunday to Thursday (Saturday to Wednesday before mid 2013) | Weeks are defined by their last trading day, not by a fixed weekday |
| Saudi close (15:00 Riyadh) happens before the US session opens | Brent and VIX for a Saudi week use the previous US close, never the same day |
| Daily price limit is plus or minus 10% on the main market | A stock closing at +10% on rebalance day is not buyable that week; larger moves are flagged as corporate actions |
| Total trading commission is 15.5 bps per trade | Base cost is 25 bps one way (commission plus spread); 10 and 50 bps are tested too |
| Short selling is restricted | Portfolios are long only |

---

## Features and target

**Stock features (13), ranked across stocks each week and scaled to [-1, 1]:**

| Group | Features |
| --- | --- |
| Momentum | returns over 1 week, 1 month, 3 months, 6 months, 12 months (skip latest month) |
| Volatility | std of daily returns over 20 and 60 days |
| Oil and market links | 120-day beta to Brent, 120-day beta to TASI |
| Liquidity | turnover, Amihud illiquidity, change in volume |
| Size | log market value |

**Selector inputs (weekly):** Brent 1-month return, Brent 20-day volatility, TASI 20-day volatility, VIX level, VIX 1-week change, each model's mean Rank IC over the past 4 and 12 weeks.

**Target:** next 5-day return of the stock minus the mean next 5-day return of its sector. Sectors with fewer than 5 stocks in a week are merged into a broader group.

---

## Evaluation

| What is tested | Metrics |
| --- | --- |
| Ranking quality | Rank IC, ICIR, precision at top 10%, NDCG |
| Portfolio results (after costs) | annualised return, Sharpe ratio, Sortino ratio, maximum drawdown, turnover |
| Warning quality | AUROC, precision, recall, F1, Brier score |
| Significance | Diebold-Mariano test (Newey-West), block bootstrap on Sharpe differences |

**Benchmarks:** TASI buy and hold, equal-weight universe, each model alone, MASTER, simple average of the three models, selector without oil inputs, hard switching, recent-IC weights, frequent retraining, full method without the warning.

**Robustness checks:** liquid stocks only (bottom 30% by traded value removed), without the 10 largest companies, costs of 10, 25 and 50 bps, warning thresholds of 10%, 20% and 30%.

**Crisis windows:** 2014 to 2016 oil crash, 2019 MSCI inclusion, 2020 COVID (February to May), 2022 rate hikes. Weeks are also grouped into calm, oil stress and panic regimes with a Gaussian Mixture Model, for reporting only.

---

## Repository structure

```
stock_trend_prediction/
├── README.md
├── config.yaml              # every setting: dates, costs, top %, seeds, thresholds
├── requirements.txt
├── .gitignore
├── docs/
│   ├── STEPS.md             # the build plan, step by step
│   └── SPEC_STEPS_0_2.md    # detailed spec for Steps 0 to 2 (wins over STEPS.md)
├── pytest.ini
├── data/
│   ├── reference/           # hand-made universe.csv and sector_map.csv, committed
│   ├── raw/                 # downloads, git-ignored
│   └── processed/           # weekly panel, features, targets, git-ignored
├── src/
│   ├── data_download.py     # Step 1
│   ├── clean.py             # Step 1
│   ├── features.py          # Step 2
│   ├── market_features.py   # Step 2
│   ├── target.py            # Step 2
│   ├── splits.py            # Step 2
│   ├── backtest.py          # Step 3
│   ├── models/
│   │   ├── ridge.py         # Step 4
│   │   ├── lgbm.py          # Step 4
│   │   ├── mlp.py           # Step 4
│   │   └── master_adapter.py  # Step 5
│   ├── selector.py          # Step 6
│   ├── warning.py           # Step 7
│   ├── regimes.py           # Step 8
│   └── evaluate.py          # Step 8
├── tests/
│   ├── test_leakage.py      # Step 2
│   └── test_backtest.py     # Step 3
├── notebooks/               # exploration only, never the source of a reported number
├── results/
│   ├── preds/               # out-of-sample predictions per model and fold
│   ├── tables/
│   └── figures/
└── app/
    ├── streamlit_app.py     # Step 9
    ├── data/                # small precomputed files for the demo
    ├── Dockerfile           # for Hugging Face Spaces
    └── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/Manognaaaaaa/stock_trend_prediction.git
cd stock_trend_prediction
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                           # runs the leakage and backtest tests
```

GPU is only needed for MASTER (Step 5). Use Google Colab or Kaggle for that step.

## How to run (once each step is built)

```bash
python -m src.data_download      # Step 1: download raw data
python -m src.clean              # Step 1: clean and build weekly panel
python -m src.features           # Step 2: features, target, folds
python -m src.models.ridge       # Step 4
python -m src.models.lgbm        # Step 4
python -m src.models.mlp         # Step 4
python -m src.selector           # Step 6
python -m src.warning            # Step 7
python -m src.evaluate           # Step 8: all tables and figures
streamlit run app/streamlit_app.py   # Step 9
```

---

## Build plan

The full plan, with tasks, outputs and checks for each step, is in [`docs/STEPS.md`](docs/STEPS.md).

| Step | What | Weeks | Gate before moving on |
| --- | --- | --- | --- |
| 0 | Repo, environment, config | 1 | `pytest` runs |
| 1 | Data download and cleaning | 1 to 2 | dataset summary table saved |
| 2 | Features, target, walk-forward folds | 2 to 3 | all 5 leakage tests pass |
| 3 | Backtester | 3 | random, perfect-foresight and momentum checks behave |
| 4 | Base models (Ridge, LightGBM, MLP) | 3 to 5 | first results table |
| 5 | MASTER benchmark | 5 to 6 | results row, or stop after 2 weeks |
| 6 | Oil-aware selector | 6 to 7 | blended and comparison scores saved |
| 7 | Warning rule | 7 to 8 | exposure series and threshold table saved |
| 8 | Evaluation | 8 to 9 | every report table and figure made by script |
| 9 | Hosted demo | 9 to 10 | public link works |
| 10 | Final report | 3 to 10 | report submitted |

**If time runs short, cut in this order:** MASTER, frequent retraining comparison, GMM regimes, robustness checks other than costs, demo pages 3 to 5.
**Never cut:** leakage tests, trading costs, the simple average benchmark, the selector without oil.

### Progress

- [x] Step 0: Repo, environment, config
- [x] Step 1: Data
- [x] Step 2: Features, target, folds
- [x] Step 3: Backtester
- [x] Step 4: Base models
- [~] Step 5: MASTER
- [x] Step 6: Selector
- [x] Step 7: Warning rule
- [x] Step 8: Evaluation
- [~] Step 9: Demo
- [~] Step 10: Report

---

## Results

To be filled in by `src/evaluate.py`.

| Method | Rank IC | ICIR | Annual return | Sharpe | Max drawdown | Turnover |
| --- | --- | --- | --- | --- | --- | --- |
| TASI buy and hold | | | | | | |
| Equal-weight universe | | | | | | |
| Ridge | | | | | | |
| LightGBM | | | | | | |
| MLP | | | | | | |
| MASTER | | | | | | |
| Simple average | | | | | | |
| Selector without oil | | | | | | |
| Oil-aware selector | | | | | | |
| Oil-aware selector with warning | | | | | | |

## Demo

Link: to be added after Step 9 (Streamlit, hosted on Hugging Face Spaces with the Docker SDK).

---

## Limitations (known in advance)

- Yahoo Finance may not have every delisted stock, so some survivorship bias may remain. The count is reported.
- The selector trains on one row per week, so it has only a few hundred rows. It is kept deliberately small, and a logistic regression selector is run as a check.
- MASTER is adapted to Saudi data with fewer input features than the original (13 stock features and about 30 market features instead of 158 and 63).
- Features are price and volume only. No fundamentals or news.

## Key references

1. S. Gu, B. Kelly, D. Xiu, "Empirical Asset Pricing via Machine Learning," *Review of Financial Studies*, 2020.
2. T. Li et al., "MASTER: Market-Guided Stock Transformer for Stock Price Forecasting," AAAI 2024. Code: https://github.com/SJTU-DMTai/MASTER
3. G. Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," NeurIPS 2017.
4. F. X. Diebold, R. S. Mariano, "Comparing Predictive Accuracy," *Journal of Business and Economic Statistics*, 1995.

The full list of 20 reviewed papers is in the project report.

## License

MIT
