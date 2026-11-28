"""Stock features (docs/SPEC_STEPS_0_2.md, Section 4.1).

Pure functions take and return DataFrames. main() reads data/processed/ (Step 1
output), builds all of Step 2 (features, market features, target, folds) and
writes the model table.
"""
import json
import logging
import math

import numpy as np
import pandas as pd

from src.utils import get_path, load_config, setup_logging, write_table

logger = logging.getLogger(__name__)

FEATURES = [
    "ret_1w", "ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_20d", "vol_60d",
    "beta_oil", "beta_tasi", "turnover", "amihud", "vol_chg", "log_mktval",
]
WEEKLY_KEYS = ["week_end", "symbol"]


# ---------------------------------------------------------------- market inputs


def market_series(market_daily: pd.DataFrame, name: str) -> pd.DataFrame:
    """Return one market series (date, close, ret) sorted by date."""
    part = market_daily.loc[market_daily["series"] == name, ["date", "close", "ret"]]
    return part.sort_values("date").reset_index(drop=True)


def align_brent_to_saudi_days(daily: pd.DataFrame, market_daily: pd.DataFrame) -> pd.DataFrame:
    """Add brent_ret and brent_date: the Brent return of the last US date strictly before each Saudi date (D10)."""
    brent = market_series(market_daily, "brent").rename(columns={"date": "brent_date", "ret": "brent_ret"})
    out = daily.sort_values("date", kind="stable")
    out = pd.merge_asof(out, brent[["brent_date", "brent_ret"]], left_on="date", right_on="brent_date",
                        direction="backward", allow_exact_matches=False)
    return out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def add_tasi_ret(daily: pd.DataFrame, market_daily: pd.DataFrame) -> pd.DataFrame:
    """Add tasi_ret: same-day TASI log return (TASI closes with the stocks, so it is known)."""
    tasi = market_series(market_daily, "tasi")[["date", "ret"]].rename(columns={"ret": "tasi_ret"})
    return daily.merge(tasi, on="date", how="left")


# ---------------------------------------------------------------- daily features


def min_rows(window: int, cfg: dict) -> int:
    """Minimum non-missing rows a window needs (D11)."""
    return math.ceil(cfg["features"]["min_coverage"] * window)


def rolling_beta(ret: pd.Series, factor: pd.Series, window: int, mp: int) -> pd.Series:
    """Rolling cov(ret, factor) / var(factor) on rows where both are present."""
    factor = factor.where(ret.notna())  # same rows in numerator and denominator
    cov = ret.rolling(window, min_periods=mp).cov(factor)
    var = factor.rolling(window, min_periods=mp).var()
    return cov / var


def stock_features(g: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute the 13 features for one stock, on its own trading rows (D11)."""
    w = cfg["features"]["windows"]

    def roll(s: pd.Series, k: int) -> pd.api.typing.Rolling:
        """Rolling window of k rows with the D11 coverage rule."""
        return s.rolling(k, min_periods=min_rows(k, cfg))

    ret = g["ret"]
    out = pd.DataFrame({"date": g["date"], "symbol": g["symbol"]})
    for name in ["ret_1w", "ret_1m", "ret_3m", "ret_6m"]:
        out[name] = roll(ret, w[name]).sum()
    k12 = w["ret_12m_total"] - w["ret_12m_skip"]
    out["ret_12m"] = roll(ret, k12).sum().shift(w["ret_12m_skip"])  # rows t-251 to t-21
    out["vol_20d"] = roll(ret, w["vol_short"]).std()
    out["vol_60d"] = roll(ret, w["vol_long"]).std()
    out["beta_oil"] = rolling_beta(ret, g["brent_ret"], w["beta"], min_rows(w["beta"], cfg))
    out["beta_tasi"] = rolling_beta(ret, g["tasi_ret"], w["beta"], min_rows(w["beta"], cfg))
    out["turnover"] = roll(g["traded_value"], w["liquidity"]).mean() / g["mktval"]
    illiq = ret.abs() / (g["traded_value"] / cfg["features"]["amihud_scale"])
    out["amihud"] = roll(illiq, w["liquidity"]).mean()
    out["vol_chg"] = roll(g["volume"], w["vol_short"]).mean() / roll(g["volume"], w["vol_chg_long"]).mean()
    out["log_mktval"] = np.log(g["mktval"])
    return out


def compute_daily_features(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Compute the raw daily features for every stock; infinite values become NaN."""
    d = daily.sort_values(["symbol", "date"], kind="stable")
    parts = [stock_features(g, cfg) for _, g in d.groupby("symbol", sort=False)]
    out = pd.concat(parts, ignore_index=True)
    out[FEATURES] = out[FEATURES].replace([np.inf, -np.inf], np.nan)
    return out


# ---------------------------------------------------------------- weekly features


def sample_weekly(daily_feats: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Take each eligible (stock, week_end) feature row from the stock's last trading day of the week."""
    elig = panel.loc[panel["eligible"], [*WEEKLY_KEYS, "last_date"]]
    feats = daily_feats.rename(columns={"date": "last_date"})
    out = elig.merge(feats, on=["symbol", "last_date"], how="left")
    return out.sort_values(WEEKLY_KEYS, kind="stable").reset_index(drop=True)


def rank_scale(df: pd.DataFrame, cols: list[str], by: str = "week_end", fill: bool = True) -> pd.DataFrame:
    """Rank each column within each `by` group and scale to [-1, 1]; NaN becomes 0 if fill."""
    out = df.copy()
    grp = out.groupby(by, sort=False)
    for col in cols:
        rank = grp[col].rank(method="average")
        n = grp[col].transform("count")
        scaled = 2 * (rank - 1) / (n - 1) - 1
        scaled = scaled.where(n > 1, 0.0).where(out[col].notna())
        out[col] = scaled.fillna(0.0) if fill else scaled
    return out


def daily_feature_table(daily: pd.DataFrame, market_daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run steps 1 to 3: align Brent and TASI, then compute the daily features."""
    d = align_brent_to_saudi_days(daily, market_daily)
    d = add_tasi_ret(d, market_daily)
    return compute_daily_features(d, cfg)


def cut_as_of(df: pd.DataFrame, col: str, as_of: pd.Timestamp | None) -> pd.DataFrame:
    """Drop rows dated after as_of (no-op when as_of is None)."""
    return df if as_of is None else df.loc[df[col] <= pd.Timestamp(as_of)]


def build_features(daily: pd.DataFrame, market_daily: pd.DataFrame, panel: pd.DataFrame, cfg: dict,
                   as_of: pd.Timestamp | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (ranked, raw) weekly features; with as_of, use only data dated on or before it."""
    daily = cut_as_of(daily, "date", as_of)
    market_daily = cut_as_of(market_daily, "date", as_of)
    panel = cut_as_of(panel, "week_end", as_of)
    raw = sample_weekly(daily_feature_table(daily, market_daily, cfg), panel)
    return rank_scale(raw, FEATURES), raw


# ---------------------------------------------------------------- whole Step 2


MODEL_COLUMNS = [
    "week_end", "symbol", "year", "industry_group", "sector_group", *FEATURES,
    "fwd_ret", "target", "target_rank", "target_end_date", "target_days",
    "traded_on_week_end", "limit_up_on_week_end", "days_traded", "date_suspect",
]


def build_step2(daily: pd.DataFrame, market_daily: pd.DataFrame, panel: pd.DataFrame,
                calendar: pd.DatetimeIndex, cfg: dict) -> dict:
    """Build features, market features, target, model table and folds from Step 1 tables."""
    from src.market_features import build_market_weekly
    from src.splits import make_folds
    from src.target import build_target

    daily_feats = daily_feature_table(daily, market_daily, cfg)
    raw = sample_weekly(daily_feats, panel)
    ranked = rank_scale(raw, FEATURES)
    market_weekly = build_market_weekly(market_daily, pd.DatetimeIndex(panel["week_end"].unique()), cfg)
    tgt, mapping = build_target(daily, panel, calendar, cfg)
    info = panel.loc[panel["eligible"], [*WEEKLY_KEYS, "industry_group", "traded_on_week_end",
                                         "limit_up_on_week_end", "days_traded", "date_suspect"]]
    table = ranked.drop(columns="last_date").merge(info, on=WEEKLY_KEYS).merge(tgt, on=WEEKLY_KEYS)
    table["year"] = table["week_end"].dt.year
    table = table[MODEL_COLUMNS].sort_values(WEEKLY_KEYS).reset_index(drop=True)
    folds = make_folds(table, calendar, cfg)
    return {"daily_feats": daily_feats, "raw": raw, "ranked": ranked, "market_weekly": market_weekly,
            "mapping": mapping, "model_table": table, "folds": folds}


def calendar_from_market(market_daily: pd.DataFrame) -> pd.DatetimeIndex:
    """Rebuild the master calendar (D7) from the TASI rows of market_daily."""
    dates = market_daily.loc[market_daily["series"] == "tasi", "date"]
    return pd.DatetimeIndex(sorted(dates.unique()), name="date").astype("datetime64[ns]")


def mapping_summary(mapping: pd.DataFrame) -> pd.DataFrame:
    """Share of weeks each industry group kept its own name, was pooled, or went to Other."""
    m = mapping.copy()
    m["outcome"] = np.where(m["sector_group"] == m["industry_group"], "own",
                            np.where(m["sector_group"] == "Other", "other", "pooled"))
    out = m.groupby("industry_group")["outcome"].value_counts(normalize=True).unstack(fill_value=0.0)
    out["weeks"] = m.groupby("industry_group").size()
    return out.round(3).reset_index()


def main() -> None:
    """Read Step 1 tables, build all of Step 2 and write its outputs."""
    setup_logging()
    cfg = load_config()
    processed = get_path("processed")
    daily = pd.read_parquet(processed / "daily.parquet")
    market_daily = pd.read_parquet(processed / "market_daily.parquet")
    panel = pd.read_parquet(processed / "weekly_panel.parquet")
    calendar = calendar_from_market(market_daily)
    logger.info("read daily %d rows, panel %d rows (%d eligible)", len(daily), len(panel), int(panel["eligible"].sum()))

    res = build_step2(daily, market_daily, panel, calendar, cfg)
    res["daily_feats"].to_parquet(processed / "features_daily_raw.parquet", index=False)
    res["raw"].to_parquet(processed / "features_weekly_raw.parquet", index=False)
    res["market_weekly"].to_parquet(processed / "market_weekly.parquet", index=False)
    res["model_table"].to_parquet(processed / "model_table.parquet", index=False)
    (processed / "folds.json").write_text(json.dumps(res["folds"], indent=2))
    write_table(res["mapping"], "sector_mapping")
    write_table(mapping_summary(res["mapping"]), "sector_mapping_summary")
    logger.info("wrote features_daily_raw %d, features_weekly_raw %d, market_weekly %d, model_table %d rows, %d folds",
                len(res["daily_feats"]), len(res["raw"]), len(res["market_weekly"]),
                len(res["model_table"]), len(res["folds"]))


if __name__ == "__main__":
    main()
