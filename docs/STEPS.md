# Build Plan, Step by Step

Each step lists its goal, the files it creates, the tasks, the outputs, and a "done when" check. Do not start a step until the previous step's check passes.

Time budget: about 10 weeks at 10 to 15 hours a week.

---

## Step 0: Repo, environment, config (week 1, 2 to 3 days)

**Goal:** a working repo where every setting lives in one config file.

**Files:** `config.yaml`, `requirements.txt`, `.gitignore`, folder layout from the README.

**Tasks**
- [ ] Create the folder layout shown in the README
- [ ] Write `requirements.txt`: pandas, numpy, pyarrow, pyyaml, yfinance, scikit-learn, lightgbm, torch, optuna, statsmodels, scipy, arch, quantstats, streamlit, plotly, pytest
- [ ] Write `config.yaml` with: start date, end date, top_pct (0.10), cost_bps ([10, 25, 50], base 25), seeds ([0, 1, 2, 3, 4]), min_sector_size (5), liquidity_drop_pct (0.30), embargo_days (5), warning thresholds (0.20, sensitivity [0.10, 0.20, 0.30]), exposure levels ([1.0, 0.5, 0.0]), crisis window dates
- [ ] `.gitignore` excludes `data/raw/`, `data/processed/`, `.venv/`, `__pycache__/`, `mlruns/`, large files in `results/preds/`
- [ ] Set up a Colab or Kaggle notebook that clones the repo and installs requirements

**Done when:** `pytest` runs and the repo is pushed to GitHub.

---

## Step 1: Data download and cleaning (weeks 1 to 2)

**Goal:** one clean daily table and one weekly panel for all main market stocks, plus TASI, Brent and VIX.

**Files:** `src/data_download.py`, `src/clean.py`, `data/raw/universe.csv`

### 1.1 Stock list
- [ ] Download the current main market list with sector (industry group) from the Saudi Exchange website
- [ ] Add delisted and suspended names from the Saudi Exchange "Suspended/Delisted/Transformed Companies" page
- [ ] Save `data/raw/universe.csv`: symbol, name, sector, listed_date, delisted_date

### 1.2 Prices
- [ ] Download daily OHLC, adjusted close and volume for every symbol with the `.SR` suffix, from the earliest date available
- [ ] Download `^TASI.SR`, `BZ=F`, `^VIX`
- [ ] Record the earliest date for each series. If Saudi stocks go back to 2007 or 2008, use that as the start date
- [ ] Save raw files untouched in `data/raw/`. Never overwrite them

### 1.3 Survivorship gap
- [ ] Try each delisted symbol in yfinance
- [ ] For missing ones, ask the university library about LSEG or Bloomberg
- [ ] If still missing, record the count and dates for the limitations section

### 1.4 Cleaning rules (apply in this order, count rows dropped by each)
1. Drop days with zero volume (stale prices)
2. Flag daily returns beyond plus or minus 10% outside a stock's first 3 trading days (likely a corporate action the adjustment missed). Check 5 by hand against Saudi Exchange announcements
3. Keep a stock in a week only if it traded at least 3 days that week
4. Require at least 60 trading days of history before a stock enters the universe
5. Define each week by its last trading day (the Saudi weekend changed in mid 2013)

### 1.5 Weekly panel
- [ ] One row per (stock, week): last close, weekly return, weekly traded value, days traded, sector, market value (or traded value as size if share counts are missing)
- [ ] Align Brent and VIX: for each Saudi week, use the last US close strictly before the Saudi rebalance day

**Outputs:** `data/processed/daily.parquet`, `data/processed/weekly_panel.parquet`, `results/tables/dataset_summary.csv` (stocks per year, rows dropped per rule, first and last date per series)

**Done when:** the dataset summary table exists and looks sensible.

---

## Step 2: Features, target, walk-forward folds (weeks 2 to 3)

**Goal:** a model-ready table and a fold calendar, with tests that prove there is no future data leakage.

**Files:** `src/features.py`, `src/market_features.py`, `src/target.py`, `src/splits.py`, `tests/test_leakage.py`

### 2.1 Stock features (from past data only)

| Feature | Definition |
| --- | --- |
| ret_1w, ret_1m, ret_3m, ret_6m | log price change over 5, 21, 63, 126 trading days |
| ret_12m | log price change from 252 to 21 trading days ago |
| vol_20d, vol_60d | std of daily returns |
| beta_oil, beta_tasi | 120-day rolling regression slope of stock return on Brent return / TASI return |
| turnover | traded value / market value |
| amihud | mean of abs(return) / traded value over 20 days |
| vol_chg | 20-day mean volume / 60-day mean volume |
| log_mktval | log market value |

- [ ] Inside each week, rank each feature across stocks and scale to [-1, 1]. Fill missing with 0 (the median after ranking)

### 2.2 Market features (one row per week)
- [ ] oil_ret_1m, oil_vol_20d, tasi_vol_20d, vix_level, vix_chg_1w
- [ ] Leave space for recent model IC columns (filled in Step 6)

### 2.3 Target
- [ ] target = next 5-day return minus mean next 5-day return of the stock's sector
- [ ] Merge sectors with fewer than 5 stocks that week into a broader group; save the mapping table
- [ ] Also store the cross-sectional rank of the target

### 2.4 Walk-forward folds (expanding window)

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| 1 | 2008 to 2009 | 2010 | 2011 |
| 2 | 2008 to 2010 | 2011 | 2012 |
| ... | ... | ... | ... |
| last | 2008 to 2023 | 2024 | 2025 |

- [ ] Drop the last 5 trading days of each block before the next block (embargo)

### 2.5 Leakage tests
- [ ] Shift test: recompute features for 20 random rows using only data up to that date; values must match
- [ ] Target test: target at week t uses prices strictly after week t's close
- [ ] Fold test: no date in both train and test; gap of at least 5 trading days
- [ ] Oil alignment test: every Brent and VIX value is dated before the Saudi rebalance day
- [ ] Random-label test: LightGBM on shuffled targets gives mean Rank IC within about plus or minus 0.01

**Outputs:** `data/processed/model_table.parquet`, `data/processed/market_weekly.parquet`, `data/processed/folds.json`, `results/tables/sector_mapping.csv`

**Done when:** all five leakage tests pass.

---

## Step 3: Backtester (week 3)

**Goal:** a trusted backtester, tested before any model exists.

**Files:** `src/backtest.py`, `tests/test_backtest.py`

**Input:** a table of (week, stock, score) and an exposure value per week (default 1.0).

**Each week**
1. Remove stocks that closed at the +10% limit on rebalance day, or traded fewer than 3 days
2. Buy the top 10% by score, equal weight, long only
3. Invest `exposure` of capital, rest in cash at 0%
4. Hold one week. Cost = cost_bps x turnover (sum of absolute weight changes)
5. Record gross return, net return, turnover, holdings

**Benchmarks built here:** TASI buy and hold, equal-weight universe.

**Sanity checks**
- [ ] Random scores: net excess return over equal-weight universe is about zero minus costs
- [ ] Perfect foresight (real next-week target as score): very large returns
- [ ] Momentum only (ret_12m): small, believable result

**Done when:** all three checks behave as expected.

---

## Step 4: Base models (weeks 3 to 5)

**Goal:** out-of-sample predictions from Ridge, LightGBM and MLP for every test year.

**Files:** `src/models/ridge.py`, `src/models/lgbm.py`, `src/models/mlp.py`

| Model | Setup | Tuning on validation year |
| --- | --- | --- |
| Ridge | scikit-learn `Ridge` on ranked target | alpha on log grid 1e-3 to 1e3 |
| LightGBM | `LGBMRegressor` on ranked target, early stopping | Optuna, 30 to 50 trials: num_leaves 8 to 64, learning_rate 0.01 to 0.1, min_child_samples 50 to 500, feature_fraction 0.5 to 1, lambda_l2 |
| MLP | PyTorch, 13 -> 64 -> 32 -> 1, ReLU, dropout 0.2 to 0.3, Adam, weight decay, MSE loss | early stopping on validation Rank IC; 5 seeds averaged |

- [ ] Save `results/preds/{model}_fold{k}.parquet` with week, stock, score
- [ ] For MLP, also save the std across the 5 seeds per stock (used in Step 7)
- [ ] Run each model's predictions through the backtester

**Expected size:** weekly Rank IC between about 0.01 and 0.05 is realistic. Above 0.10 means check for leakage.

**Done when:** a first results table (mean Rank IC, ICIR, net Sharpe per model) is saved to `results/tables/base_models.csv`.

---

## Step 5: MASTER benchmark (weeks 5 to 6, hard limit 2 weeks)

**Goal:** MASTER predictions in the same format as the other models.

**Files:** `src/models/master_adapter.py`, `notebooks/master_colab.ipynb`

**Background:** the official repo (https://github.com/SJTU-DMTai/MASTER) ships preprocessed Chinese data with 158 stock factors, 63 market features and 1 label per sample. Reuse its model class and training loop. Write your own data loader.

- [ ] Clone the repo and run their CSI300 example once to confirm the environment
- [ ] Build daily inputs: last 8 days of the 13 stock features per stock
- [ ] Build daily market vector: mean and std of TASI return and of traded value over 5, 10, 20, 30, 60 days; same for Brent return (about 30 features)
- [ ] Label: sector-relative next 5-day return, z-scored across stocks each day
- [ ] Batch = one day, shape [n_stocks, 8, n_stock_features + n_market_features]
- [ ] Train on the same folds with the paper's default settings, 5 seeds, on Colab or Kaggle GPU
- [ ] Take the prediction on each week's last trading day; save `results/preds/master_fold{k}.parquet`

**Done when:** MASTER has a row in the results table, or two weeks have passed. If stopped, record it as a limitation; the MLP stays as the deep learning model.

---

## Step 6: Oil-aware selector (weeks 6 to 7)

**Goal:** weekly model weights from oil and market conditions, and the blended ranking.

**Files:** `src/selector.py`

### 6.1 Selector table (one row per test week, out-of-sample predictions only)
- [ ] ic_ridge, ic_lgbm, ic_mlp: realised Rank IC that week
- [ ] best_model: model with the highest IC (label, 3 classes)
- [ ] Inputs: oil_ret_1m, oil_vol_20d, tasi_vol_20d, vix_level, vix_chg_1w, each model's mean IC over past 4 and 12 weeks (lagged one week)

### 6.2 Train with its own walk-forward
- [ ] For test year T, train on selector rows before year T
- [ ] LightGBM kept small and fixed in advance: max_depth 2 to 3, num_leaves at most 7, min_child_samples at least 20, about 100 trees, learning_rate 0.05, strong L2
- [ ] Multinomial logistic regression on the same inputs as a check

### 6.3 Blend
- [ ] Convert each model's scores to ranks in [0, 1]
- [ ] score = p_ridge x rank_ridge + p_lgbm x rank_lgbm + p_mlp x rank_mlp

### 6.4 Comparison versions (built from the same files)

| Version | Blend |
| --- | --- |
| Simple average | 1/3 each (key benchmark for RQ2) |
| Selector without oil | oil inputs removed |
| Hard switch | 100% on the most likely model |
| Recent-IC weights | weights from past 12-week IC, no learning |
| Frequent retraining | each base model refit every quarter |

**Outputs:** `results/preds/selector_fold{k}.parquet`, `results/tables/selector_weights.csv`

**Done when:** scores for the oil-aware selector and all five comparison versions are saved.

---

## Step 7: Warning rule (weeks 7 to 8)

**Goal:** a weekly exposure level (100%, 50% or 0%).

**Files:** `src/warning.py`

**Signals (known at the start of the week)**
1. Bad-week probability: a second small LightGBM classifier (same inputs and settings as the selector). Label bad_week = 1 when the blended portfolio's next-week net return is below the equal-weight universe
2. Disagreement: 1 minus mean pairwise Spearman correlation of the three models' rankings, plus the MLP seed spread

**Rule** (thresholds = percentile of each signal on past weeks only)

| Condition | Exposure |
| --- | --- |
| neither signal above threshold | 100% |
| one signal above threshold | 50% |
| both above threshold | 0% |

- [ ] Run with the 20% threshold as the main setting
- [ ] Sensitivity table with 10%, 20%, 30%

**Outputs:** `results/tables/exposure.csv`, `results/tables/warning_sensitivity.csv`

**Done when:** the selector-with-warning backtest is saved and the sensitivity table exists.

---

## Step 8: Evaluation (weeks 8 to 9)

**Goal:** every table and figure in the report produced by one script.

**Files:** `src/evaluate.py`, `src/regimes.py`

- [ ] Metrics per method: Rank IC, ICIR, precision at top 10%, NDCG, annual return, Sharpe, Sortino, max drawdown, turnover
- [ ] Warning metrics: AUROC, precision, recall, F1, Brier
- [ ] Diebold-Mariano: loss = negative weekly Rank IC, Newey-West errors. Selector vs simple average first, then vs each single model
- [ ] Block bootstrap on Sharpe differences (`arch`, block length about 8 weeks, 5,000 draws, 95% interval)
- [ ] All metrics per crisis window (dates in `config.yaml`)
- [ ] GMM with 3 components on oil_ret_1m, oil_vol_20d, tasi_vol_20d, vix_level; name them calm, oil stress, panic; report results and average selector weights per regime
- [ ] Robustness: liquid stocks only, without 10 largest, costs 10, 25, 50 bps
- [ ] Figures: cumulative net value with crisis windows shaded; selector weights over time with Brent below; drawdowns with and without warning; Rank IC by year

**Outputs:** `results/tables/table1.csv`, crisis, regime and robustness tables, `results/figures/*.png`

**Done when:** every table and figure needed in the report exists in `results/`.

---

## Step 9: Hosted demo (weeks 9 to 10)

**Goal:** a public Streamlit app that reads only precomputed files.

**Files:** `app/streamlit_app.py`, `app/data/`, `app/Dockerfile`, `app/requirements.txt`

**Pages**
1. This week's ranking: pick a test week; top 10% stocks with sector, each model's rank, blended score
2. Model trust over time: selector weights with Brent price and crisis shading
3. Warning status: bad-week probability, disagreement, exposure for the chosen week
4. Backtest: cumulative net value of all methods with a cost selector (10, 25, 50 bps)
5. Crisis explorer: metrics for a chosen crisis window

**Hosting**
- Default: Hugging Face Spaces with the Docker SDK and the Streamlit template (the built-in Streamlit SDK was deprecated in April 2025)
- Backup: Streamlit Community Cloud connected to the GitHub repo

**Done when:** the public link opens in a private browser window and each page loads in under 5 seconds.

---

## Step 10: Final report (weeks 3 to 10, written alongside)

| Report section | Write after | Source |
| --- | --- | --- |
| Dataset | Step 1 | dataset_summary.csv, survivorship count |
| Features and target | Step 2 | feature table, sector mapping, tests passed |
| Results, Table 1 | Step 8 | table1.csv |
| Crisis and regime results | Step 8 | crisis and regime tables |
| Discussion | Step 8 | deep models vs simple, selector weights by regime, liquid-only results, drawdown saved vs return given up |
| Limitations | end | survivorship, few selector rows, reduced MASTER inputs, price-only features |

**State these changes from the midsem report openly:** data start date (if earlier than 2012), equal-weight universe benchmark, separate bad-week classifier, logistic regression selector check, reduced MASTER inputs.

**Open questions for Dr. Shukla:** is long-only evaluation acceptable, and is price-only data enough without fundamentals?

---

## Cut order if time runs short

1. MASTER
2. Frequent retraining comparison
3. GMM regimes (keep fixed crisis windows)
4. Robustness checks other than costs
5. Demo pages 3 to 5

**Never cut:** leakage tests, trading costs, simple average benchmark, selector without oil.
