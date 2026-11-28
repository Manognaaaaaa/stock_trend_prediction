"""Synthetic data fixture (docs/SPEC_STEPS_0_2.md, Section 4.6).

make_synthetic() builds Yahoo-shaped raw tables that look like the real data:
Saturday-to-Wednesday trading before 2013-06-29 and Sunday-to-Thursday after,
holiday gaps, zero-volume days, delistings, a 2013 listing, fake corporate
action jumps and a US calendar for Brent and VIX. The session fixture runs it
through the real cleaning functions and the Step 2 functions (features,
market features, target, folds).
"""
import numpy as np
import pandas as pd
import pytest

from src.clean import clean_all
from src.features import build_step2
from src.utils import get_path, load_config

START, END = "2011-01-01", "2015-12-31"
SAUDI_TZ, US_TZ = "Asia/Riyadh", "America/New_York"

# 40 stocks in 6 industry groups; Energy has only 2 stocks so the merge rule is tested.
GROUPS = {"Banks": 12, "Materials": 10, "Food & Beverages": 8, "Financial Services": 5,
          "Insurance": 3, "Energy": 2}
DELISTED = {"1005": "2013-12-31", "2004": "2014-09-30"}  # symbol -> last trading day
LISTED_2013 = ("3001", "2013-03-03")                     # symbol, first trading day
CA_JUMPS = [("1002", "2012-05-15"), ("2002", "2014-11-04")]  # fake +40% missed adjustments
IPO_JUMP = 0.30                                          # day-2 move of the 2013 listing (not flagged)
REIT_SYMBOL = "4330"                                     # excluded by universe.exclude_groups

# Holidays: (first day, number of calendar days) removed from the Saudi calendar.
SAUDI_HOLIDAYS = [("2011-08-28", 9), ("2012-10-24", 7), ("2013-08-08", 5), ("2013-09-26", 1),
                  ("2014-07-27", 6), ("2014-10-02", 8), ("2015-07-16", 6), ("2015-09-24", 4)]
US_HOLIDAYS = ["2011-07-04", "2011-12-26", "2012-07-04", "2012-12-25", "2013-07-04",
               "2013-12-25", "2014-07-04", "2014-12-25", "2015-07-03", "2015-12-25"]


def saudi_calendar() -> pd.DatetimeIndex:
    """Saudi trading days: Sat-Wed before 2013-06-29, Sun-Thu after, minus holidays."""
    days = pd.date_range(START, END, freq="D")
    change = pd.Timestamp("2013-06-29")
    old = days[(days < change) & days.day_of_week.isin([5, 6, 0, 1, 2])]
    new = days[(days >= change) & days.day_of_week.isin([6, 0, 1, 2, 3])]
    cal = old.append(new)
    for first, n in SAUDI_HOLIDAYS:
        cal = cal[~cal.isin(pd.date_range(first, periods=n, freq="D"))]
    return cal


def us_calendar() -> pd.DatetimeIndex:
    """US trading days: Mon-Fri minus a few holidays."""
    days = pd.bdate_range(START, END)
    return days[~days.isin(pd.to_datetime(US_HOLIDAYS))]


def symbols() -> pd.DataFrame:
    """Return the synthetic universe (40 stocks plus one REIT that must be excluded)."""
    rows, i = [], 0
    for g_idx, (group, n) in enumerate(GROUPS.items(), start=1):
        for k in range(1, n + 1):
            rows.append({"symbol": f"{g_idx}{k:03d}", "name": f"Stock {g_idx}{k:03d}", "industry_group": group})
            i += 1
    uni = pd.DataFrame(rows)
    uni = pd.concat([uni, pd.DataFrame([{"symbol": REIT_SYMBOL, "name": "Fund", "industry_group": "REITs"}])])
    uni["status"] = np.where(uni["symbol"].isin(DELISTED), "delisted", "listed")
    uni["listed_date"] = pd.NaT
    uni["delisted_date"] = pd.to_datetime(uni["symbol"].map(DELISTED))
    uni.loc[uni["symbol"] == LISTED_2013[0], "listed_date"] = pd.Timestamp(LISTED_2013[1])
    uni["symbol"] = uni["symbol"].astype("string")
    return uni.reset_index(drop=True)


def yahoo_frame(dates: pd.DatetimeIndex, close: np.ndarray, adj: np.ndarray, volume: np.ndarray, tz: str) -> pd.DataFrame:
    """Shape arrays like a yfinance history() table with a tz-aware Date column."""
    return pd.DataFrame({
        "Date": dates.tz_localize(tz), "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Adj Close": adj, "Volume": volume, "Dividends": 0.0, "Stock Splits": 0.0,
    })


def stock_prices(symbol: str, cal: pd.DatetimeIndex, rng: np.random.Generator) -> pd.DataFrame:
    """Random-walk prices for one stock with a dividend, zero-volume days and planted events."""
    dates = cal
    if symbol in DELISTED:
        dates = dates[dates <= pd.Timestamp(DELISTED[symbol])]
    if symbol == LISTED_2013[0]:
        dates = dates[dates >= pd.Timestamp(LISTED_2013[1])]
    n = len(dates)
    log_ret = np.clip(rng.normal(0.0003, 0.015, n), -0.09, 0.09)
    log_ret[0] = 0.0
    if symbol == LISTED_2013[0]:
        log_ret[1] = np.log1p(IPO_JUMP)  # IPO days have no limit, must not be flagged
    for sym, day in CA_JUMPS:
        if sym == symbol:
            log_ret[dates.get_loc(pd.Timestamp(day))] = np.log1p(0.40)
    adj = 20.0 * np.exp(np.cumsum(log_ret))
    # One dividend of 3% mid-sample: raw close drops, adjusted close does not.
    div_idx = n // 2
    factor = np.where(np.arange(n) < div_idx, 1 / 0.97, 1.0)
    close = adj * factor  # raw close equals adjusted close after the dividend
    volume = rng.integers(10_000, 1_000_000, n).astype("int64")
    volume[rng.random(n) < 0.02] = 0
    volume[:5] = np.maximum(volume[:5], 1)  # keep IPO days tradeable
    return yahoo_frame(dates, close, adj, volume, SAUDI_TZ)


def add_raw_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Add one NaN close, one off-calendar Friday row and one duplicate date (R0 and dedup tests)."""
    out = df.copy()
    out.loc[10, "Close"] = np.nan
    friday = out.loc[20, "Date"] + pd.offsets.Week(weekday=4)
    extra = out.loc[[20]].assign(Date=friday)
    dup = out.loc[[30]]
    return pd.concat([out, extra, dup], ignore_index=True)


def market_series(dates: pd.DatetimeIndex, tz: str, level: float, vol: float, rng: np.random.Generator) -> pd.DataFrame:
    """Random-walk index or futures series shaped like yfinance output."""
    close = level * np.exp(np.cumsum(rng.normal(0, vol, len(dates))))
    vol_arr = rng.integers(1_000, 10_000, len(dates)).astype("int64")
    return yahoo_frame(dates, close, close, vol_arr, tz)


def shares_tables(universe: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dated share counts for the first 10 stocks, snapshots for the next 20, none for the rest."""
    syms = list(universe.loc[universe["industry_group"] != "REITs", "symbol"])
    full_parts = []
    for s in syms[:10]:
        dates = pd.date_range("2012-01-01", END, freq="QS").tz_localize(SAUDI_TZ)
        shares = np.round(rng.uniform(1e8, 1e9)) * np.linspace(1.0, 1.2, len(dates))
        full_parts.append(pd.DataFrame({"symbol": s, "date": dates, "shares": shares}))
    full = pd.concat(full_parts, ignore_index=True)
    snapshot = pd.DataFrame({"symbol": syms[:30], "shares_outstanding": rng.uniform(1e8, 1e9, 30)})
    snapshot["symbol"] = snapshot["symbol"].astype("string")
    return full, snapshot


def make_synthetic(seed: int = 0) -> dict:
    """Build Yahoo-shaped raw tables plus universe and sector map for the pipeline tests."""
    rng = np.random.default_rng(seed)
    cal, us = saudi_calendar(), us_calendar()
    universe = symbols()
    raw_prices = {s: stock_prices(s, cal, rng) for s in universe["symbol"]}
    raw_prices["1001"] = add_raw_noise(raw_prices["1001"])
    raw_market = {
        "tasi": market_series(cal, SAUDI_TZ, 7000.0, 0.01, rng),
        "brent": market_series(us, US_TZ, 80.0, 0.02, rng),
        "vix": market_series(us, US_TZ, 18.0, 0.05, rng),
    }
    shares_full, snapshot = shares_tables(universe, rng)
    sector_map = pd.read_csv(get_path("reference") / "sector_map.csv")
    return {"raw_prices": raw_prices, "raw_market": raw_market, "shares_full": shares_full,
            "snapshot": snapshot, "universe": universe, "sector_map": sector_map}


def synthetic_config() -> dict:
    """Project config with the settings the synthetic data needs (fewer train years)."""
    cfg = load_config()
    cfg["splits"]["min_train_years"] = 1
    return cfg


@pytest.fixture(scope="session")
def cfg() -> dict:
    """Config used by the synthetic tests."""
    return synthetic_config()


@pytest.fixture(scope="session")
def synthetic(cfg: dict) -> dict:
    """Synthetic raw data run through the real cleaning and Step 2 functions."""
    raw = make_synthetic(seed=0)
    excluded = cfg["universe"]["exclude_groups"]
    universe = raw["universe"].loc[~raw["universe"]["industry_group"].isin(excluded)].reset_index(drop=True)
    res = clean_all(raw["raw_prices"], raw["raw_market"], raw["shares_full"], raw["snapshot"],
                    universe, raw["sector_map"], cfg)
    step2 = build_step2(res["daily"], res["market_daily"], res["panel"], res["calendar"], cfg)
    return {**raw, "universe_clean": universe, **res, **step2}


@pytest.fixture(scope="session")
def real() -> dict:
    """Step 1 and Step 2 outputs read from data/processed/ (skips if they are missing)."""
    import json

    from src.features import FEATURES, calendar_from_market

    processed = get_path("processed")
    names = ["daily", "market_daily", "weekly_panel", "features_weekly_raw", "model_table", "market_weekly"]
    paths = {n: processed / f"{n}.parquet" for n in names}
    if not all(p.exists() for p in paths.values()) or not (processed / "folds.json").exists():
        pytest.skip("real data not built: run python -m src.clean and python -m src.features")
    t = {n: pd.read_parquet(p) for n, p in paths.items()}
    return {"daily": t["daily"], "market_daily": t["market_daily"], "panel": t["weekly_panel"],
            "raw": t["features_weekly_raw"], "ranked": t["model_table"][["week_end", "symbol", *FEATURES]],
            "model_table": t["model_table"], "market_weekly": t["market_weekly"],
            "calendar": calendar_from_market(t["market_daily"]),
            "folds": json.loads((processed / "folds.json").read_text()), "cfg": load_config()}
