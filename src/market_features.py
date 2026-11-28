"""Market features (docs/SPEC_STEPS_0_2.md, Section 4.2).

One row per week_end. Brent and VIX use the last US close strictly before the
Saudi rebalance day; TASI uses week_end itself. Model IC columns are added in
Step 6 (D18).
"""
import logging

import pandas as pd

from src.utils import get_path, load_config, setup_logging

logger = logging.getLogger(__name__)

MARKET_FEATURES = ["oil_ret_1m", "oil_vol_20d", "tasi_vol_20d", "vix_level", "vix_chg_1w"]


def series_daily(market_daily: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return one series (date, close, ret) sorted by date."""
    part = market_daily.loc[market_daily["series"] == name, ["date", "close", "ret"]]
    return part.sort_values("date").reset_index(drop=True)


def asof_before(weeks: pd.DataFrame, series: pd.DataFrame, date_name: str) -> pd.DataFrame:
    """Attach the last series row dated strictly before each week_end (D10)."""
    right = series.rename(columns={"date": date_name})
    return pd.merge_asof(weeks, right, left_on="week_end", right_on=date_name,
                         direction="backward", allow_exact_matches=False)


def build_market_weekly(market_daily: pd.DataFrame, week_ends: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Build oil, TASI and VIX features for each week_end from each series' own daily calendar."""
    m = cfg["market_features"]
    brent, tasi, vix = (series_daily(market_daily, s) for s in ["brent", "tasi", "vix"])
    # Rolling values first, on each series' own trading days; full windows only.
    brent["oil_ret_1m"] = brent["ret"].rolling(m["oil_ret_days"]).sum()
    brent["oil_vol_20d"] = brent["ret"].rolling(m["vol_days"]).std()
    tasi["tasi_vol_20d"] = tasi["ret"].rolling(m["vol_days"]).std()
    vix["vix_level"] = vix["close"]
    vix["vix_chg_1w"] = vix["close"] - vix["close"].shift(m["vix_chg_days"])

    weeks = pd.DataFrame({"week_end": pd.DatetimeIndex(sorted(week_ends)).astype("datetime64[ns]")})
    out = asof_before(weeks, brent[["date", "oil_ret_1m", "oil_vol_20d"]], "brent_asof_date")
    out = asof_before(out, vix[["date", "vix_level", "vix_chg_1w"]], "vix_asof_date")
    out = out.merge(tasi[["date", "tasi_vol_20d"]].rename(columns={"date": "week_end"}), on="week_end", how="left")
    cols = ["week_end", "brent_asof_date", "vix_asof_date", *MARKET_FEATURES]
    return out[cols]


def main() -> None:
    """Build market_weekly.parquet from market_daily and the weekly panel."""
    setup_logging()
    cfg = load_config()
    processed = get_path("processed")
    market_daily = pd.read_parquet(processed / "market_daily.parquet")
    week_ends = pd.DatetimeIndex(pd.read_parquet(processed / "weekly_panel.parquet")["week_end"].unique())
    out = build_market_weekly(market_daily, week_ends, cfg)
    out.to_parquet(processed / "market_weekly.parquet", index=False)
    logger.info("market_weekly: %d weeks", len(out))


if __name__ == "__main__":
    main()
