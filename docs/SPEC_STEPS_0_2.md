# Build Spec for Claude Code: Steps 0 to 2

Project: Ranking Saudi Stocks with Machine Learning (repo `Manognaaaaaa/stock_trend_prediction`)
Scope of this spec: Step 0 (finish the setup), Step 1 (data download and cleaning), Step 2 (features, target, folds, leakage tests).
Put this file in the repo as `docs/SPEC_STEPS_0_2.md`. It adds detail to `docs/STEPS.md`. Where the two disagree, this file wins, and the "Decisions" section says why.

---

## 0. Rules for Claude Code (apply to every step)

1. Build one step at a time. Stop at the end of each step, run the checks in its "Done when" list, show the output, and wait for approval before starting the next step.
2. Every number, date, window length and threshold comes from `config.yaml` through `src.utils.load_config()`. No magic numbers inside functions. If a new setting is needed, add it to `config.yaml` with a comment.
3. Split every module into pure functions (DataFrame in, DataFrame out, no file reading or writing) and a `main()` that does the file input and output. Tests call the pure functions directly. This is what makes the leakage tests possible.
4. Never overwrite anything in `data/raw/`. If a raw file exists, skip it and log "skipped, exists". To redownload, the user deletes the file by hand.
5. Use only past data for any value dated week `t`. When in doubt, use the stricter choice and write a comment.
6. Notebooks are for exploration only. No reported number may come from a notebook.
7. Use Python `logging` (INFO level) instead of `print`. Every `main()` logs row counts in and out.
8. Use pandas 3 style: no chained assignment (`df[a][b] = x`), use `.loc`. String columns use the default string dtype. Dates are tz-naive `datetime64[ns]` (see 1.4).
9. Save tables as Parquet with `pyarrow`. Save small human-readable tables as CSV in `results/tables/`.
10. Add type hints and a one-line docstring to every function. Keep functions short.
11. Do not add features, models or steps that are not in this spec.

---

## 1. Decisions this spec makes (gaps in STEPS.md)

| # | Topic | Decision | Reason |
| --- | --- | --- | --- |
| D1 | Where `universe.csv` lives | `data/reference/universe.csv` and `data/reference/sector_map.csv`, committed to git | They are made by hand and small. `data/raw/` is git-ignored, so they would be lost there. |
| D2 | How the stock list is built | Made by hand from the Saudi Exchange website (see Section 5). The code reads it, it does not scrape the website. | The Saudi Exchange site is built with JavaScript and blocks scripts. Scraping would be fragile. |
| D3 | REITs and ETFs | Excluded from the universe | They are funds, not companies. Their returns follow different drivers. Write this in the report. |
| D4 | Price columns | Download with `auto_adjust=False`. Use `Adj Close` for all returns. Use raw `Close` times `Volume` for traded value and for the price limit check. | Adjusted prices give correct returns across dividends and bonus shares. Raw prices give the real traded value and the real limit move. |
| D5 | Market value | First try `Ticker.get_shares_full()` (share count with dates, used only on or before each date). If missing, use the latest `sharesOutstanding` as a constant and set `shares_source = "snapshot"`. If both are missing, `mktval` is NaN. | Yahoo has no full share history for most Saudi stocks. The snapshot fallback uses a future share count, so report the share of rows that use it as a limitation. |
| D6 | Week definition | `week_id = date.to_period("W-FRI")`. `week_end` = last TASI trading day inside that period. | Friday is a non-trading day both before mid 2013 (Saturday to Wednesday week) and after (Sunday to Thursday week). So a week ending on Friday holds exactly one Saudi trading week in both periods. |
| D7 | Master trading calendar | The dates on which `^TASI.SR` has a close | One calendar for all stocks. Stock rows on dates not in this calendar are dropped and counted. |
| D8 | Target horizon | From this week's `week_end` close to next week's `week_end` close. Usually 5 trading days, fewer in holiday weeks. Store the day count as `target_days`. | Matches the one-week holding period in the backtester. A fixed 5-day horizon would not line up with rebalance days in Eid weeks. In config, replace `horizon_days: 5` with `horizon: "next_week"`. |
| D9 | Corporate action flags | A day is flagged when the absolute adjusted simple return is above 0.105 and it is not in the stock's first 3 trading days. On flagged days, set the daily return `ret` to NaN. A target window that contains a flagged day gets target NaN. | Main market moves are capped at 10%, so a bigger adjusted move means Yahoo missed an adjustment. Setting the return to NaN stops one bad day from corrupting a whole year of momentum. 0.105 leaves room for tick rounding. |
| D10 | Oil timing inside daily betas | For Saudi day `d`, use the Brent daily return of the last US trading date strictly before `d`. | That US close is known before the Saudi open on day `d`. The Saudi market reacts to the previous US session's oil move. |
| D11 | Rolling windows | Counted in the stock's own trading rows, not calendar days. Each window needs at least 80% non-missing values (`features.min_coverage: 0.8`), else NaN. | Zero-volume days are dropped, so row counts are the honest measure of history. |
| D12 | Rows kept in the weekly panel | All stock-weeks are kept with flag columns (`eligible`, `traded_on_week_end`, `limit_up_on_week_end`). The model table keeps only `eligible` rows. | The backtester in Step 3 needs the flags. Counting rows that fail each rule is easy with flags. |
| D13 | Target needs a trade on `week_end` | If the stock did not trade on `week_end` itself, its target is NaN. | The buy price would be stale otherwise. |
| D14 | Embargo | Defined by a condition, not a fixed number of weeks (see 3.4) | Holiday weeks have fewer days, so a fixed week count can fail. |
| D15 | First test year | `first_test_year` = first year with at least `splits.min_train_years` (default 2) full years of features before the validation year | Features need about 252 trading days of history, so the first usable year is one year after the data starts. |
| D16 | Tests without real data | Tests run on a small synthetic dataset by default. The same tests run on the real data with `pytest -m data`. | `pytest` must pass on a fresh clone and on Colab before any data is downloaded. |
| D17 | Daily features for MASTER | Step 2 also saves raw daily features (`features_daily_raw.parquet`) | Step 5 needs the last 8 days of features per stock. Saving them now avoids recomputing. |
| D18 | Model IC columns in market features | Not added now. Step 6 joins them on `week_end`. | Empty placeholder columns add nothing. |

---

## 2. Step 0: finish the setup

The skeleton commit already has the folders, config, requirements and stubs. Remaining tasks:

1. **Make `src` importable in tests.** Add `pytest.ini` at the repo root:
   ```ini
   [pytest]
   pythonpath = .
   testpaths = tests
   markers =
       data: needs the real processed data in data/processed (run with: pytest -m data)
       slow: takes more than 30 seconds
   addopts = -m "not data"
   ```
2. **Add `data/reference/`** with two committed files:
   - `universe.csv` with the header only for now: `symbol,name,industry_group,status,listed_date,delisted_date`
   - `sector_map.csv` filled with the default mapping in Section 5.2
3. **Extend `src/utils.py`:**
   - `setup_logging() -> None`: INFO level, format `%(asctime)s %(name)s %(levelname)s %(message)s`
   - `write_raw(df, path) -> bool`: writes Parquet only if `path` does not exist, returns False and logs if it exists
   - `write_table(df, name) -> Path`: writes CSV to `results/tables/{name}.csv` (overwriting is fine here, these are outputs)
   - `get_path` keeps working; add `"reference": "data/reference"` to `paths` in config
4. **Update `config.yaml`** with the new keys in Section 6.
5. **Fix the README:** the clone URL and folder name say `tadawul-ranking`; change to `https://github.com/Manognaaaaaa/stock_trend_prediction.git` and `stock_trend_prediction`.
6. **Add `notebooks/colab_setup.ipynb`** with 3 cells: clone the repo, `pip install -r requirements.txt`, run `pytest`.
7. **Optional:** `.github/workflows/tests.yml` that installs only `pandas numpy pyarrow pyyaml scikit-learn lightgbm scipy pytest` (skip torch to keep it fast) and runs `pytest`.

**Done when:** `pytest` exits with code 0 on a fresh clone (all current tests skipped or passing), and the changes are pushed.

---

## 3. Step 1: data download and cleaning

### 3.1 `src/data_download.py`

**Inputs:** `data/reference/universe.csv`, config.

**Functions**
- `load_universe(path) -> DataFrame`: reads the CSV with `symbol` as a 4-digit string (keep leading zeros if any), parses dates, drops rows whose `industry_group` is in `universe.exclude_groups`. Raises a clear error if the file has no rows, telling the user to fill it (Section 5.1).
- `yahoo_symbol(symbol) -> str`: returns `f"{symbol}{stock_suffix}"`, for example `"2222.SR"`.
- `download_prices(ticker, start, end) -> DataFrame | None`: uses `yf.Ticker(ticker).history(start=..., end=..., auto_adjust=False, actions=True)`. Retries `download.retries` times with a wait of `download.sleep_sec * 2**attempt`. Returns None if empty after retries.
- `download_shares(ticker) -> tuple[DataFrame | None, float | None]`: returns `get_shares_full(start=start_date)` as a DataFrame (`date, shares`) and `info.get("sharesOutstanding")`. Wrap both in try/except; Yahoo often fails on these.

**`main()`**
1. For each stock in the universe (listed and delisted): download prices and shares, sleep `download.sleep_sec` between symbols.
2. Save untouched raw output, with the index reset so the date is a column:
   - `data/raw/prices/{symbol}.parquet`
   - `data/raw/shares/{symbol}.parquet` (only if not empty)
   - `data/raw/market/tasi.parquet`, `brent.parquet`, `vix.parquet`
3. Save `data/raw/shares_snapshot.csv`: `symbol, shares_outstanding, fetched_at` (only written if it does not exist).
4. Save `data/raw/download_log.csv` (append mode is fine): `symbol, yahoo_ticker, status (ok / empty / error / skipped_exists), first_date, last_date, n_rows, has_shares_full, has_snapshot, fetched_at, yfinance_version`.
5. Log a final summary: how many listed and delisted symbols were found and how many were empty.

**CLI:** `python -m src.data_download` (all), `python -m src.data_download --symbols 2222 1120` (a few, for testing), `--market-only` (only TASI, Brent, VIX).

### 3.2 `src/clean.py`

**Pure functions, in the order they run**

1. `to_local_dates(df) -> DataFrame`: convert the tz-aware Yahoo index to a tz-naive date in the exchange's own time zone (`Asia/Riyadh` for `.SR`, `America/New_York` for `BZ=F` and `^VIX`), then normalise to midnight. Drop duplicate dates (keep last) and count them.
2. `build_calendar(tasi) -> DatetimeIndex`: sorted TASI trading dates.
3. `stack_prices(raw: dict[str, DataFrame], universe) -> DataFrame`: one long table, columns `date, symbol, open, high, low, close, adj_close, volume`. Lower-case column names.
4. `apply_cleaning_rules(daily, calendar, cfg) -> tuple[DataFrame, dict]`: applies the rules below in this order and returns the cleaned table plus a dict of counts per rule.
   - R0: drop rows with NaN `close` or `adj_close`, or dates not in the TASI calendar
   - R1: drop rows with `volume == 0`
   - Then compute, per stock, sorted by date:
     - `row_num`: trading row number of the stock, starting at 1
     - `raw_ret = close / close.shift(1) - 1`
     - `adj_ret_simple = adj_close / adj_close.shift(1) - 1`
     - `ret = log(adj_close / adj_close.shift(1))`
     - `traded_value = close * volume` (SAR)
     - `limit_up = raw_ret >= universe.limit_up_tolerance` (0.0995)
   - R2: `ca_flag = abs(adj_ret_simple) > universe.ca_threshold` and `row_num > universe.first_days_no_limit`. Set `ret` to NaN where `ca_flag` is True (D9). Count flags. Do not drop the rows.
5. `attach_market_value(daily, shares_full, snapshot) -> DataFrame`: adds `shares_out`, `shares_source` (`"full"`, `"snapshot"`, or `"none"`) and `mktval = close * shares_out`. For `"full"`, use `merge_asof` on date, backward direction, per symbol, so only share counts dated on or before each day are used.
6. `attach_sectors(daily, universe, sector_map) -> DataFrame`: adds `industry_group` and `broad_group`. Any group missing from `sector_map.csv` raises an error naming the group.
7. `build_weekly_panel(daily, calendar, cfg) -> DataFrame`, one row per (stock, week):

   | Column | Definition |
   | --- | --- |
   | `week_end` | last TASI trading day in the `W-FRI` period (D6) |
   | `symbol` | stock code |
   | `last_date` | stock's last trading row in that week |
   | `close`, `adj_close` | values on `last_date` |
   | `week_ret` | log of this week's `adj_close` over last week's `adj_close` for the same stock (NaN if it did not trade last week) |
   | `traded_value_week` | sum of daily traded value |
   | `days_traded` | count of trading rows in the week |
   | `traded_on_week_end` | `last_date == week_end` |
   | `limit_up_on_week_end` | `traded_on_week_end` and `limit_up` on that day |
   | `history_rows` | the stock's `row_num` on `last_date` |
   | `industry_group`, `broad_group`, `mktval`, `shares_source` | from the daily table on `last_date` |
   | `eligible` | `days_traded >= min_days_traded_per_week` and `history_rows >= min_history_days` |

   Record counts for R3 (fails days traded) and R4 (fails history), applied in that order.
8. `build_market_daily(tasi, brent, vix) -> DataFrame`: columns `date, series ("tasi", "brent", "vix"), close, ret` where `ret` is the daily log return (for VIX, keep `close` as the level and still compute `ret`).
9. `make_dataset_summary(...) -> DataFrame`: long format, columns `section, item, value`. Sections:
   - `series`: first date, last date and row count for TASI, Brent, VIX and the earliest and latest stock
   - `universe`: listed count, delisted count, delisted found on Yahoo, delisted missing (this is the survivorship number for the report), REITs/ETFs excluded
   - `cleaning`: rows dropped by R0, R1, R3, R4 and flags from R2
   - `coverage`: share of stock-days with `shares_source` = full / snapshot / none
   - `per_year`: for each year, mean eligible stocks per week and number of distinct eligible stocks
10. `sample_ca_flags(daily, n=5, seed=0) -> DataFrame`: 5 random flagged rows (symbol, date, raw_ret, adj_ret_simple) saved to `results/tables/ca_flags_to_check.csv` so the user can check them by hand against Saudi Exchange announcements.

**`main()`** reads all raw files, runs the functions above, and writes:
- `data/processed/daily.parquet` (columns: `date, symbol, open, high, low, close, adj_close, volume, traded_value, raw_ret, ret, limit_up, ca_flag, row_num, shares_out, shares_source, mktval, industry_group, broad_group`)
- `data/processed/market_daily.parquet`
- `data/processed/weekly_panel.parquet`
- `results/tables/dataset_summary.csv`
- `results/tables/ca_flags_to_check.csv`

**Note:** Brent and VIX are aligned to Saudi weeks in Step 2 (`market_features.py`), not here, so the alignment code and its test live in one place.

### 3.3 Step 1 checks (Claude Code runs these and shows the output)

- Run on 5 symbols first (`--symbols 2222 1120 2010 7010 4190`), then on all.
- `week_end` values are never a Friday, and every `week_end` is in the TASI calendar.
- Before 2013-06-29 no `week_end` falls on a Thursday; after mid 2013 no `week_end` falls on a Wednesday unless the Thursday was a holiday. Log how many exceptions there are.
- No duplicate (`symbol`, `date`) in `daily` and no duplicate (`symbol`, `week_end`) in the panel.
- Print the `per_year` section of the summary. Expect roughly 100+ eligible stocks in early years rising to about 200+ recently. Flag anything far off.

**Done when:** `dataset_summary.csv` exists, the checks above pass, and the user has looked at the 5 corporate action flags.

---

## 4. Step 2: features, target, walk-forward folds

### 4.1 `src/features.py`

**Pure functions**

1. `align_brent_to_saudi_days(daily, market_daily) -> DataFrame`: adds `brent_ret` to each Saudi stock-day using `pd.merge_asof(..., direction="backward", allow_exact_matches=False)` on date against Brent dates, so the Brent date used is strictly before the Saudi date (D10). Also keeps `brent_date` for the test.
2. `add_tasi_ret(daily, market_daily) -> DataFrame`: same-day TASI log return (TASI closes at the same time as the stocks, so this is known).
3. `compute_daily_features(daily, cfg) -> DataFrame`: per stock, sorted by date, on its own trading rows (D11). With `k` the window and `mp = ceil(0.8 * k)`:

   | Feature | Formula |
   | --- | --- |
   | `ret_1w` | rolling sum of `ret` over 5 rows |
   | `ret_1m` | rolling sum over 21 rows |
   | `ret_3m` | rolling sum over 63 rows |
   | `ret_6m` | rolling sum over 126 rows |
   | `ret_12m` | rolling sum over 231 rows, then `shift(21)` (covers rows t-251 to t-21) |
   | `vol_20d`, `vol_60d` | rolling std of `ret` over 20 and 60 rows |
   | `beta_oil` | rolling cov(`ret`, `brent_ret`) / rolling var(`brent_ret`) over 120 rows |
   | `beta_tasi` | same with `tasi_ret` |
   | `turnover` | rolling mean of `traded_value` over 20 rows / `mktval` |
   | `amihud` | rolling mean of `abs(ret) / (traded_value / 1e6)` over 20 rows |
   | `vol_chg` | rolling mean of `volume` over 20 rows / rolling mean over 60 rows |
   | `log_mktval` | `log(mktval)` |

   Put all window lengths in `config.yaml` under `features.windows`. Replace infinite values with NaN.
4. `sample_weekly(daily_feats, panel) -> DataFrame`: for each (stock, `week_end`) in the panel, take the feature row on that stock's `last_date`. Keep only `eligible` rows.
5. `rank_scale(df, cols, by="week_end") -> DataFrame`: inside each week, rank each column with `method="average"` over non-missing values, then `x = 2 * (rank - 1) / (n - 1) - 1`, giving values in [-1, 1]. If `n == 1`, set 0. Fill NaN with 0 after ranking.
6. `build_features(daily, market_daily, panel, cfg, as_of=None) -> tuple[DataFrame, DataFrame]`: runs 1 to 5. If `as_of` is given, first drops every row of `daily`, `market_daily` and `panel` dated after `as_of`. Returns (ranked weekly features, raw weekly features). The shift test calls this with `as_of`.

**`main()`** writes:
- `data/processed/features_daily_raw.parquet` (D17)
- `data/processed/features_weekly_raw.parquet` (unranked, for checks and the report)
- then calls `market_features`, `target` and `splits` so `python -m src.features` builds all of Step 2, as the README says, and finally writes `data/processed/model_table.parquet`

### 4.2 `src/market_features.py`

`build_market_weekly(market_daily, week_ends, cfg) -> DataFrame`, one row per `week_end`:

| Column | Definition |
| --- | --- |
| `brent_asof_date` | last Brent date strictly before `week_end` |
| `vix_asof_date` | last VIX date strictly before `week_end` |
| `oil_ret_1m` | sum of Brent daily log returns over the 21 US trading days ending on `brent_asof_date` |
| `oil_vol_20d` | std of Brent daily log returns over the 20 US trading days ending on `brent_asof_date` |
| `tasi_vol_20d` | std of TASI daily log returns over the 20 Saudi trading days ending on `week_end` |
| `vix_level` | VIX close on `vix_asof_date` |
| `vix_chg_1w` | `vix_level` minus VIX close 5 US trading days before `vix_asof_date` |

Compute the rolling values on each series' own daily calendar first, then pick the row with `merge_asof(..., allow_exact_matches=False)` for Brent and VIX and an exact match for TASI.

Writes `data/processed/market_weekly.parquet`.

### 4.3 `src/target.py`

1. `assign_sector_groups(panel_week, cfg) -> DataFrame`: for each week, on eligible rows:
   - industry groups with at least `min_sector_size` stocks keep their own name as `sector_group`
   - smaller groups are pooled by `broad_group`; a pool with at least `min_sector_size` stocks gets the `broad_group` name
   - anything still smaller goes into `"Other"`
   - returns the panel with `sector_group` and a mapping table `week_end, industry_group, sector_group, n_stocks`
2. `compute_forward_returns(daily, panel, calendar) -> DataFrame`: for each (stock, `week_end` W), with W' the next `week_end` in the calendar:
   - `target_end_date = W'`
   - `fwd_ret` = sum of the stock's daily `ret` over dates in (W, W']
   - `target_days` = number of TASI trading days in (W, W']
   - `fwd_ret` is NaN if: the stock did not trade on W (D13), it has no trading row in (W, W'], any `ret` in that window is NaN because of a flag (D9), or W is the last week in the data
3. `make_target(panel, fwd) -> DataFrame`: `target = fwd_ret - mean(fwd_ret)` over rows in the same `week_end` and `sector_group` with non-NaN `fwd_ret`. Then `target_rank` = `rank_scale` of `target` within each week (reuse the function from `features.py`, but leave NaN targets as NaN, do not fill with 0).

Writes `results/tables/sector_mapping.csv` (the mapping table, plus a short summary of how often each industry group was merged) and returns the target columns for the model table.

### 4.4 `src/splits.py`

`make_folds(model_table, calendar, cfg) -> list[dict]`

1. `year = week_end.year`. Usable years are those with eligible rows and non-NaN `target_rank`.
2. For each test year T from `first_test_year` (D15) to the last year:
   - test block: weeks in year T
   - validation block: weeks in year T-1
   - train block: all weeks from the first usable week up to the end of year T-2 (expanding)
3. Embargo (D14), applied to the end of the train block and the end of the validation block: drop weeks from the end of the block, one at a time, until the last kept week satisfies
   `count of TASI trading days d with target_end_date < d <= first week_end of the next block >= embargo_days`
   Usually this drops 1 week; in holiday weeks it may drop 2.
4. Return a list of dicts and save `data/processed/folds.json`:
   ```json
   {"fold": 1, "test_year": 2012, "val_year": 2011,
    "train_start": "2009-03-05", "train_end": "2010-12-23",
    "val_start": "2011-01-05", "val_end": "2011-12-22",
    "test_start": "2012-01-04", "test_end": "2012-12-27",
    "n_train_rows": 0, "n_val_rows": 0, "n_test_rows": 0}
   ```
   (Dates above are only an example of the format.)
5. Helper `get_fold_rows(model_table, fold) -> tuple[DataFrame, DataFrame, DataFrame]` used by every model in Step 4.

### 4.5 Final model table

`data/processed/model_table.parquet`, eligible rows only, one row per (stock, `week_end`):

`week_end, symbol, year, industry_group, sector_group, ret_1w, ret_1m, ret_3m, ret_6m, ret_12m, vol_20d, vol_60d, beta_oil, beta_tasi, turnover, amihud, vol_chg, log_mktval, fwd_ret, target, target_rank, target_end_date, target_days, traded_on_week_end, limit_up_on_week_end, days_traded`

Keep the 13 feature names in one list, `FEATURES`, in `src/features.py`, and import it everywhere.

### 4.6 Tests

**`tests/conftest.py`: synthetic data fixture**

`make_synthetic(seed=0) -> dict` returns `daily`, `market_daily`, `universe`, `sector_map`, built like the real data:
- 40 stocks in 6 industry groups (one group with only 2 stocks, so the merge rule is tested)
- dates from 2011-01-01 to 2015-12-31; Saturday to Wednesday trading before 2013-06-29, Sunday to Thursday after; a few holiday gaps
- prices: random walk with small drift; 2% of days with zero volume; 2 stocks delisted mid-way; 1 stock listed in 2013; 2 fake corporate action jumps of +40%
- Brent and VIX on a Monday to Friday calendar with a few US holidays
- run it through the real `clean.py`, `features.py`, `target.py` and `splits.py` functions (with `min_train_years: 1` so there are folds) and return everything as a session-scoped fixture

**`tests/test_leakage.py`: the five required tests.** Each test is written once and run twice with `@pytest.mark.parametrize("source", ["synthetic", pytest.param("real", marks=pytest.mark.data)])`. The `real` case loads files from `data/processed/` and skips if they are missing.

1. `test_shift_features_use_only_past_data`: pick 20 random (symbol, `week_end`) rows. For each, call `build_features(..., as_of=week_end)` and compare that week's ranked and raw features with the stored ones using `np.allclose(atol=1e-9, equal_nan=True)`. Also: change all prices after `week_end` (multiply by 2) and check the features for that week do not change.
2. `test_target_uses_only_future_prices`: every `target_end_date > week_end`. Recompute `fwd_ret` for 20 rows from `daily` and match. Changing prices after `target_end_date` does not change `fwd_ret`; changing the price on `target_end_date` does.
3. `test_folds_do_not_overlap_and_respect_embargo`: train, validation and test weeks are disjoint in every fold. The embargo condition in 4.4 holds at both block ends. `max(target_end_date)` in train is before `min(week_end)` in validation, and the same for validation and test.
4. `test_oil_and_vix_dated_before_rebalance_day`: in `market_weekly`, every `brent_asof_date < week_end` and `vix_asof_date < week_end`. In the daily table from `align_brent_to_saudi_days`, every `brent_date < date`.
5. `test_random_labels_give_zero_rank_ic`: shuffle `target_rank` across stocks inside each week (seed 0). Train `LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, verbose=-1)` on the first fold's train rows, predict its test rows, compute the weekly Spearman correlation between prediction and the shuffled target, and take the mean. Pass if `abs(mean) <= max(0.01, 3 * std / sqrt(n_weeks))` (the second term is for the small synthetic data). Mark the real case `slow`.

**`tests/test_pipeline.py`: extra checks (cheap, synthetic only)**
- no `week_end` is a Friday; every `week_end` is a TASI trading day
- ranked features are inside [-1, 1] and have no NaN
- every (`week_end`, `sector_group`) has at least `min_sector_size` stocks, except `"Other"`
- the two fake +40% jumps are flagged and their `ret` is NaN
- the delisted stocks stop appearing after their last date; the 2013 listing only becomes eligible after 60 trading rows

### 4.7 Step 2 checks (Claude Code runs these and shows the output)

- `pytest` (synthetic) passes, then `pytest -m data` (real) passes
- print the mean and std of each raw feature by year, and the share of zeros after ranking per feature (a feature that is mostly zeros means it is mostly missing, which is likely for `turnover` and `log_mktval` if share data is poor)
- print the folds table: test year, row counts, train start and end
- print the mean weekly Spearman correlation of each single ranked feature with `target_rank` over all weeks. Values should be small (absolute value under about 0.05). A value above 0.10 means look for leakage before going on

**Done when:** all five leakage tests pass on synthetic and real data, the four output files exist (`model_table.parquet`, `market_weekly.parquet`, `folds.json`, `sector_mapping.csv`), and the README progress boxes for Steps 0, 1 and 2 are ticked.

---

## 5. Things Manogna does by hand (Claude Code cannot do these)

### 5.1 Fill `data/reference/universe.csv`
- From saudiexchange.sa, Market Watch or Issuers list for the Main Market: copy every company's code, name and industry group.
- From the "Suspended / Delisted / Transformed Companies" page: add each delisted company with `status = delisted` and the delisting date.
- Leave `listed_date` blank if unknown; the code uses the first Yahoo date.
- Columns: `symbol,name,industry_group,status,listed_date,delisted_date` (dates as YYYY-MM-DD).

### 5.2 Default `sector_map.csv` (check the group names match the website exactly)

| industry_group | broad_group |
| --- | --- |
| Energy | Energy |
| Materials | Materials |
| Capital Goods | Industrials |
| Commercial & Professional Svc | Industrials |
| Transportation | Industrials |
| Consumer Durables & Apparel | Consumer Discretionary |
| Consumer Services | Consumer Discretionary |
| Consumer Discretionary Distribution & Retail | Consumer Discretionary |
| Media and Entertainment | Communication |
| Telecommunication Services | Communication |
| Consumer Staples Distribution & Retail | Consumer Staples |
| Food & Beverages | Consumer Staples |
| Household & Personal Products | Consumer Staples |
| Health Care Equipment & Svc | Health Care |
| Pharma, Biotech & Life Science | Health Care |
| Banks | Financials |
| Financial Services | Financials |
| Insurance | Financials |
| Real Estate Mgmt & Dev't | Real Estate |
| Utilities | Utilities |
| Software & Services | Information Technology |
| REITs | Real Estate (excluded by D3) |

### 5.3 Other manual tasks
- Check the 5 rows in `ca_flags_to_check.csv` against Saudi Exchange announcements and note what you found.
- Email the library about LSEG or Bloomberg access for delisted stock prices, using the "delisted missing" count from the summary.
- Note for the report: sector labels are today's labels applied to all years (Tadawul moved to the current groups in 2017).

---

## 6. New and changed `config.yaml` keys

```yaml
download:
  sleep_sec: 0.5              # pause between Yahoo calls
  retries: 3                  # attempts per symbol

universe:
  # existing keys stay
  exclude_groups: ["REITs"]   # D3
  ca_threshold: 0.105         # D9: abs adjusted daily move above this is flagged
  first_days_no_limit: 3      # IPO days with no price limit
  limit_up_tolerance: 0.0995  # raw close-to-close move counted as limit up

features:
  min_coverage: 0.8           # D11: share of a window that must be non-missing
  windows:
    ret_1w: 5
    ret_1m: 21
    ret_3m: 63
    ret_6m: 126
    ret_12m_total: 252
    ret_12m_skip: 21
    vol_short: 20
    vol_long: 60
    beta: 120
    liquidity: 20
    vol_chg_long: 60

market_features:
  oil_ret_days: 21
  vol_days: 20
  vix_chg_days: 5

target:
  horizon: "next_week"        # D8, replaces horizon_days
  type: "sector_relative"

splits:
  scheme: "expanding"
  embargo_days: 5
  min_train_years: 2          # D15

paths:
  # existing keys stay
  reference: "data/reference"
```

---

## 7. Prompt to paste into Claude Code

> Read `README.md`, `docs/STEPS.md` and `docs/SPEC_STEPS_0_2.md`. The spec file adds detail to STEPS.md and wins where they disagree.
> Build Step 0 only, following Section 2 of the spec. Then run `pytest`, show me the output, and stop.
> Follow the rules in Section 0 of the spec for all code. Do not start Step 1 until I say so.

After Step 0 is approved:

> Build Step 1 following Section 3 of the spec. First write the code and the synthetic data fixture from Section 4.6 (`tests/conftest.py`) so the cleaning functions can be tested without downloads. Then run the download for the 5 test symbols only, run `clean.py`, show me the Step 1 checks from 3.3, and stop.

After Step 1 is approved (and `universe.csv` is filled):

> Run the full download and cleaning, show me `dataset_summary.csv`, then build Step 2 following Section 4. Run `pytest` and `pytest -m data`, show me the Step 2 checks from 4.7, and stop.
