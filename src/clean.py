"""Cleaning and weekly panel (docs/SPEC_STEPS_0_2.md, Section 3.2).

Pure functions take and return DataFrames. main() reads data/raw/ and writes
data/processed/ and results/tables/.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_download import load_universe
from src.utils import get_path, load_config, setup_logging, write_table

logger = logging.getLogger(__name__)

PRICE_COLUMNS = {
    "Open": "open", "High": "high", "Low": "low", "Close": "close",
    "Adj Close": "adj_close", "Volume": "volume",
}
DAILY_COLUMNS = [
    "date", "symbol", "open", "high", "low", "close", "adj_close", "volume", "traded_value",
    "raw_ret", "ret", "limit_up", "ca_flag", "row_num", "shares_out", "shares_source",
    "mktval", "industry_group", "broad_group", "date_suspect",
]
PANEL_FROM_LAST = ["close", "adj_close", "industry_group", "broad_group", "mktval", "shares_source", "date_suspect"]


# ---------------------------------------------------------------- dates


def to_local_dates(df: pd.DataFrame, tz: str, date_col: str = "Date") -> tuple[pd.DataFrame, int]:
    """Turn Yahoo's tz-aware timestamps into tz-naive local dates; drop duplicate dates."""
    out = df.copy()
    dates = pd.to_datetime(out[date_col])
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert(tz).dt.tz_localize(None)
    out = out.drop(columns=[date_col])
    out.insert(0, "date", dates.dt.normalize().astype("datetime64[ns]"))
    out = out.sort_values("date", kind="stable")
    dup = out["date"].duplicated(keep="last")
    return out.loc[~dup].reset_index(drop=True), int(dup.sum())


def drop_tasi_filler(tasi: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop filler TASI rows: no volume and the same close as the previous row (not real sessions)."""
    prev = tasi["Close"].shift(1)
    filler = tasi["Volume"].isna() & (tasi["Close"] == prev)
    return tasi.loc[~filler].reset_index(drop=True), int(filler.sum())


def flag_suspect_dates(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add date_suspect: True for stock rows before Yahoo's Saudi dates become reliable."""
    out = daily.copy()
    out["date_suspect"] = out["date"] < pd.Timestamp(cfg["calendar"]["yahoo_dates_reliable_from"])
    return out


def build_calendar(tasi: pd.DataFrame) -> pd.DatetimeIndex:
    """Return the master trading calendar: sorted dates with a TASI close (D7)."""
    dates = tasi.loc[tasi["Close"].notna(), "date"]
    return pd.DatetimeIndex(sorted(dates.unique()), name="date").astype("datetime64[ns]")


def week_end_map(calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Map every calendar date to its W-FRI week and that week's last trading day (D6)."""
    cal = pd.DataFrame({"date": calendar})
    cal["week_id"] = cal["date"].dt.to_period("W-FRI")
    cal["week_end"] = cal.groupby("week_id")["date"].transform("max")
    return cal


def week_end_list(calendar: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the sorted unique week_end dates of the calendar."""
    return pd.DatetimeIndex(week_end_map(calendar)["week_end"].unique(), name="week_end")


# ---------------------------------------------------------------- stock table


def stack_prices(raw: dict[str, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    """Stack per-symbol price tables into one long table with lower-case columns."""
    keep = set(universe["symbol"])
    parts = []
    for symbol, df in raw.items():
        if symbol not in keep:
            continue
        part = df.rename(columns=PRICE_COLUMNS)[["date", *PRICE_COLUMNS.values()]].copy()
        part.insert(1, "symbol", symbol)
        parts.append(part)
    cols = ["date", "symbol", *PRICE_COLUMNS.values()]
    if not parts:
        return pd.DataFrame(columns=cols)
    out = pd.concat(parts, ignore_index=True)
    out["symbol"] = out["symbol"].astype("string")
    out["volume"] = out["volume"].astype("float64")
    return out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def add_returns(daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Add row_num, raw/adjusted returns, traded value and the limit-up flag per stock."""
    out = daily.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    grp = out.groupby("symbol", sort=False)
    out["row_num"] = grp.cumcount() + 1
    prev_close = grp["close"].shift(1)
    prev_adj = grp["adj_close"].shift(1)
    out["raw_ret"] = out["close"] / prev_close - 1
    out["adj_ret_simple"] = out["adj_close"] / prev_adj - 1
    out["ret"] = np.log(out["adj_close"] / prev_adj)
    out["traded_value"] = out["close"] * out["volume"]  # raw close, SAR (D4)
    out["limit_up"] = (out["raw_ret"] >= cfg["universe"]["limit_up_tolerance"]).fillna(False)
    return out


def apply_cleaning_rules(daily: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Apply R0 (bad rows), R1 (zero volume) and R2 (corporate action flags); return counts."""
    counts = {"rows_in": len(daily)}
    nan_price = daily["close"].isna() | daily["adj_close"].isna()
    off_cal = ~daily["date"].isin(calendar)
    counts["r0_nan_price"] = int(nan_price.sum())
    counts["r0_not_in_calendar"] = int((off_cal & ~nan_price).sum())
    out = daily.loc[~(nan_price | off_cal)]
    # Missing volume is treated like zero volume: no evidence of a real trade.
    no_trade = out["volume"].isna() | (out["volume"] == 0)
    counts["r1_zero_volume"] = int(no_trade.sum())
    out = add_returns(out.loc[~no_trade], cfg)
    uni = cfg["universe"]
    out["ca_flag"] = (out["adj_ret_simple"].abs() > uni["ca_threshold"]) & (out["row_num"] > uni["first_days_no_limit"])
    out.loc[out["ca_flag"], "ret"] = np.nan  # D9
    counts["r2_ca_flags"] = int(out["ca_flag"].sum())
    counts["rows_out"] = len(out)
    return out, counts


def attach_market_value(daily: pd.DataFrame, shares_full: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    """Add shares_out, shares_source and mktval, using only share counts dated on or before each day (D5)."""
    out = daily.sort_values("date", kind="stable")
    full = shares_full.loc[shares_full["shares"].notna() & (shares_full["shares"] > 0), ["symbol", "date", "shares"]]
    full = full.astype({"symbol": "string"}).sort_values("date", kind="stable")
    out = pd.merge_asof(out, full, on="date", by="symbol", direction="backward")
    snap = snapshot.loc[snapshot["shares_outstanding"].notna(), ["symbol", "shares_outstanding"]]
    out = out.merge(snap.astype({"symbol": "string"}), on="symbol", how="left")
    has_full = out["shares"].notna()
    has_snap = out["shares_outstanding"].notna() & ~has_full
    out["shares_out"] = out["shares"].where(has_full, out["shares_outstanding"]).astype("float64")
    out["shares_source"] = pd.Series("none", index=out.index, dtype="string")
    out.loc[has_full, "shares_source"] = "full"
    out.loc[has_snap, "shares_source"] = "snapshot"
    out["mktval"] = out["close"] * out["shares_out"]
    out = out.drop(columns=["shares", "shares_outstanding"])
    return out.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)


def attach_sectors(daily: pd.DataFrame, universe: pd.DataFrame, sector_map: pd.DataFrame) -> pd.DataFrame:
    """Add industry_group and broad_group; raise if a group is missing from sector_map."""
    groups = set(universe.loc[universe["symbol"].isin(daily["symbol"].unique()), "industry_group"])
    missing = sorted(groups - set(sector_map["industry_group"]))
    if missing:
        raise ValueError(f"industry groups missing from sector_map.csv: {missing}")
    uni = universe[["symbol", "industry_group"]].astype({"symbol": "string"})
    out = daily.merge(uni, on="symbol", how="left")
    return out.merge(sector_map[["industry_group", "broad_group"]], on="industry_group", how="left")


# ---------------------------------------------------------------- weekly panel


def build_weekly_panel(daily: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict) -> tuple[pd.DataFrame, dict]:
    """Build one row per (stock, week) with flags and the eligible column (D6, D12)."""
    cal = week_end_map(calendar)[["date", "week_end"]]
    d = daily.merge(cal, on="date", how="left").sort_values(["symbol", "date"], kind="stable")
    grp = d.groupby(["symbol", "week_end"], sort=True)
    last = grp.tail(1).set_index(["symbol", "week_end"])
    panel = pd.DataFrame({
        "last_date": last["date"],
        **{col: last[col] for col in PANEL_FROM_LAST},
        "limit_up_last": last["limit_up"],
        "history_rows": last["row_num"],
        "traded_value_week": grp["traded_value"].sum(),
        "days_traded": grp.size(),
    }).reset_index()
    panel = add_week_ret(panel, calendar)
    panel["traded_on_week_end"] = panel["last_date"] == panel["week_end"]
    panel["limit_up_on_week_end"] = panel["traded_on_week_end"] & panel["limit_up_last"]
    uni = cfg["universe"]
    fails_days = panel["days_traded"] < uni["min_days_traded_per_week"]
    fails_hist = panel["history_rows"] < uni["min_history_days"]
    panel["eligible"] = ~fails_days & ~fails_hist
    counts = {
        "panel_rows": len(panel),
        "r3_fails_days_traded": int(fails_days.sum()),
        "r4_fails_history": int((fails_hist & ~fails_days).sum()),
        "eligible_rows": int(panel["eligible"].sum()),
    }
    cols = ["week_end", "symbol", "last_date", "close", "adj_close", "week_ret", "traded_value_week",
            "days_traded", "traded_on_week_end", "limit_up_on_week_end", "history_rows",
            "industry_group", "broad_group", "mktval", "shares_source", "date_suspect", "eligible"]
    return panel[cols].sort_values(["week_end", "symbol"]).reset_index(drop=True), counts


def add_week_ret(panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Add week_ret: log adj_close change from the previous calendar week (NaN if no trade then)."""
    weeks = week_end_list(calendar)
    prev_week = pd.Series(weeks[:-1], index=weeks[1:])
    out = panel.sort_values(["symbol", "week_end"], kind="stable").reset_index(drop=True)
    grp = out.groupby("symbol", sort=False)
    prev_adj = grp["adj_close"].shift(1)
    prev_we = grp["week_end"].shift(1)
    consecutive = prev_we == out["week_end"].map(prev_week)
    out["week_ret"] = np.log(out["adj_close"] / prev_adj).where(consecutive)
    return out


# ---------------------------------------------------------------- market series


def build_market_daily(tasi: pd.DataFrame, brent: pd.DataFrame, vix: pd.DataFrame) -> pd.DataFrame:
    """Stack TASI, Brent and VIX into date, series, close, ret (daily log return)."""
    parts = []
    for name, df in [("tasi", tasi), ("brent", brent), ("vix", vix)]:
        part = df.loc[df["Close"].notna() & (df["Close"] > 0), ["date", "Close"]].rename(columns={"Close": "close"})
        part = part.sort_values("date").reset_index(drop=True)
        part["ret"] = np.log(part["close"] / part["close"].shift(1))
        part.insert(1, "series", name)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["series"] = out["series"].astype("string")
    return out


# ---------------------------------------------------------------- checks and summary


def check_week_ends(panel: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict) -> dict:
    """Count week_end values that break the calendar rules in spec Section 3.3."""
    we = pd.Series(panel["week_end"].unique())
    change = pd.Timestamp(cfg["calendar"]["weekend_change_date"])
    early, late = we[we < change], we[we >= change]
    return {
        "n_week_ends": len(we),
        "friday": int((we.dt.day_of_week == 4).sum()),
        "not_in_calendar": int((~we.isin(calendar)).sum()),
        "thursday_before_change": int((early.dt.day_of_week == 3).sum()),
        "wednesday_after_change": int((late.dt.day_of_week == 2).sum()),
        "dup_symbol_week": int(panel.duplicated(["symbol", "week_end"]).sum()),
    }


def series_rows(name: str, dates: pd.Series) -> list[tuple]:
    """Return summary rows (first date, last date, rows) for one date series."""
    return [("series", f"{name}_first_date", dates.min().date().isoformat()),
            ("series", f"{name}_last_date", dates.max().date().isoformat()),
            ("series", f"{name}_rows", len(dates))]


def universe_rows(universe: pd.DataFrame, found: set[str], n_excluded: int) -> list[tuple]:
    """Return summary rows for listed and delisted counts and the survivorship gap."""
    delisted = universe["status"].str.lower().eq("delisted")
    found_del = universe.loc[delisted, "symbol"].isin(found)
    return [("universe", "listed", int((~delisted).sum())),
            ("universe", "delisted", int(delisted.sum())),
            ("universe", "delisted_found_on_yahoo", int(found_del.sum())),
            ("universe", "delisted_missing", int((~found_del).sum())),
            ("universe", "listed_missing", int((~universe.loc[~delisted, "symbol"].isin(found)).sum())),
            ("universe", "excluded_groups", n_excluded)]


def per_year_rows(panel: pd.DataFrame) -> list[tuple]:
    """Return per-year mean eligible stocks per week and distinct eligible stocks."""
    elig = panel.loc[panel["eligible"]]
    rows = []
    for year, grp in elig.groupby(elig["week_end"].dt.year):
        rows.append(("per_year", f"{year}_mean_eligible_per_week", round(grp.groupby("week_end").size().mean(), 1)))
        rows.append(("per_year", f"{year}_distinct_eligible", grp["symbol"].nunique()))
    return rows


def make_dataset_summary(daily: pd.DataFrame, panel: pd.DataFrame, market_daily: pd.DataFrame,
                         universe: pd.DataFrame, found: set[str], counts: dict, n_excluded: int) -> pd.DataFrame:
    """Build the long-format dataset summary (section, item, value)."""
    rows = []
    for name in ["tasi", "brent", "vix"]:
        rows += series_rows(name, market_daily.loc[market_daily["series"] == name, "date"])
    first = daily.groupby("symbol")["date"].min()
    last = daily.groupby("symbol")["date"].max()
    rows += [("series", "stocks_earliest_first_date", f"{first.min().date()} ({first.idxmin()})"),
             ("series", "stocks_latest_first_date", f"{first.max().date()} ({first.idxmax()})"),
             ("series", "stocks_last_date", last.max().date().isoformat()),
             ("series", "stock_rows", len(daily))]
    rows += universe_rows(universe, found, n_excluded)
    rows += [("cleaning", k, v) for k, v in counts.items()]
    share = daily["shares_source"].value_counts(normalize=True)
    rows += [("coverage", f"shares_{s}", round(float(share.get(s, 0.0)), 4)) for s in ["full", "snapshot", "none"]]
    rows += [("cleaning", "date_suspect_stock_days", int(daily["date_suspect"].sum())),
             ("cleaning", "date_suspect_eligible_weeks", int((panel["eligible"] & panel["date_suspect"]).sum()))]
    rows += per_year_rows(panel)
    out = pd.DataFrame(rows, columns=["section", "item", "value"])
    out["value"] = out["value"].astype("string")
    return out


def sample_ca_flags(daily: pd.DataFrame, n: int = 5, seed: int = 0) -> pd.DataFrame:
    """Return up to n random corporate-action-flagged rows for a manual check."""
    flagged = daily.loc[daily["ca_flag"], ["symbol", "date", "close", "raw_ret", "adj_ret_simple"]]
    return flagged.sample(n=min(n, len(flagged)), random_state=seed).sort_values(["symbol", "date"])


# ---------------------------------------------------------------- pipeline


def clean_all(raw_prices: dict[str, pd.DataFrame], raw_market: dict[str, pd.DataFrame],
              shares_full: pd.DataFrame, snapshot: pd.DataFrame, universe: pd.DataFrame,
              sector_map: pd.DataFrame, cfg: dict) -> dict:
    """Run every cleaning step on raw Yahoo-style tables; return all outputs and counts."""
    saudi, us = cfg["calendar"]["saudi_tz"], cfg["calendar"]["us_tz"]
    counts = {"duplicate_dates": 0}
    local = {}
    for symbol, df in raw_prices.items():
        local[symbol], n_dup = to_local_dates(df, saudi)
        counts["duplicate_dates"] += n_dup
    market = {}
    for name, df in raw_market.items():
        market[name], n_dup = to_local_dates(df, saudi if name == "tasi" else us)
        counts["duplicate_dates"] += n_dup
    if "Volume" in market["tasi"].columns:
        market["tasi"], counts["tasi_filler_rows"] = drop_tasi_filler(market["tasi"])
    calendar = build_calendar(market["tasi"])
    stacked = stack_prices(local, universe)
    daily, rule_counts = apply_cleaning_rules(stacked, calendar, cfg)
    counts.update(rule_counts)
    shares_local = local_shares(shares_full, saudi)
    daily = attach_market_value(daily, shares_local, snapshot)
    daily = attach_sectors(daily, universe, sector_map)
    daily = flag_suspect_dates(daily, cfg)
    panel, panel_counts = build_weekly_panel(daily, calendar, cfg)
    counts.update(panel_counts)
    market_daily = build_market_daily(market["tasi"], market["brent"], market["vix"])
    return {"daily": daily, "panel": panel, "market_daily": market_daily,
            "calendar": calendar, "counts": counts}


def local_shares(shares_full: pd.DataFrame, tz: str) -> pd.DataFrame:
    """Convert raw share-count dates to local tz-naive dates, keeping the last value per day."""
    if shares_full.empty:
        return pd.DataFrame({"symbol": pd.Series(dtype="string"), "date": pd.Series(dtype="datetime64[ns]"),
                             "shares": pd.Series(dtype="float64")})
    parts = []
    for symbol, grp in shares_full.groupby("symbol"):
        part, _ = to_local_dates(grp.drop(columns="symbol"), tz, date_col="date")
        part.insert(0, "symbol", symbol)
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    out["shares"] = out["shares"].astype("float64")
    return out


# ---------------------------------------------------------------- file input and output


VOLUME_SUFFIX = {"K": 1e3, "M": 1e6, "B": 1e9}
INVESTING_COLUMNS = {"Price": "Close", "Open": "Open", "High": "High", "Low": "Low"}


def parse_number(text: pd.Series) -> pd.Series:
    """Turn Investing.com number strings like '11,948.79' into floats ('-' becomes NaN)."""
    num = pd.to_numeric(text.astype("string").str.replace(",", "").str.strip().replace("-", None), errors="coerce")
    return num.astype("float64")


def parse_volume(text: pd.Series) -> pd.Series:
    """Turn Investing.com volumes like '245.67M' into floats ('-' becomes NaN)."""
    s = text.astype("string").str.replace(",", "").str.strip()
    mult = s.str[-1].map(VOLUME_SUFFIX).astype("float64").fillna(1.0)
    num = s.where(~s.str[-1].isin(list(VOLUME_SUFFIX)), s.str[:-1])
    return pd.to_numeric(num.replace("-", None), errors="coerce").astype("float64") * mult


def parse_investing_csv(df: pd.DataFrame, date_format: str) -> pd.DataFrame:
    """Turn an Investing.com historical export into a Yahoo-shaped table (tz-naive local Date)."""
    df = df.rename(columns=lambda c: str(c).strip().strip('"').lstrip("﻿"))
    missing = {"Date", *INVESTING_COLUMNS} - set(df.columns)
    if missing:
        raise ValueError(f"TASI CSV is missing columns {sorted(missing)}; expected an Investing.com export")
    out = pd.DataFrame({"Date": pd.to_datetime(df["Date"].astype("string").str.strip(), format=date_format)})
    for src, dst in INVESTING_COLUMNS.items():
        out[dst] = parse_number(df[src])
    out["Adj Close"] = out["Close"]  # a price index has no dividend adjustment
    out["Volume"] = parse_volume(df["Vol."]) if "Vol." in df.columns else np.nan
    out = out.dropna(subset=["Close"]).sort_values("Date", kind="stable")
    return out.drop_duplicates("Date", keep="last").reset_index(drop=True)


def read_tasi_reference(ref: Path, cfg: dict) -> pd.DataFrame:
    """Read and join every hand-downloaded TASI CSV part in data/reference/."""
    paths = sorted(ref.glob(cfg["tasi_csv"]["pattern"]))
    if not paths:
        raise FileNotFoundError(
            f"No TASI history: Yahoo returned none and no {cfg['tasi_csv']['pattern']} in {ref}. "
            "Download it from investing.com/indices/tasi-historical-data (Daily) and save it there."
        )
    parts = [parse_investing_csv(pd.read_csv(p, dtype="string"), cfg["tasi_csv"]["date_format"]) for p in paths]
    out = pd.concat(parts, ignore_index=True).sort_values("Date", kind="stable")
    out = out.drop_duplicates("Date", keep="last")
    # Keep the study period only, like the Yahoo downloads (files may run past end_date).
    start, end = pd.Timestamp(cfg["data"]["start_date"]), pd.Timestamp(cfg["data"]["end_date"])
    out = out.loc[out["Date"].between(start, end)]
    logger.info("TASI from %d reference file(s): %d rows in %s..%s", len(paths), len(out), start.date(), end.date())
    return out.reset_index(drop=True)


def read_market(raw: Path, ref: Path, cfg: dict) -> tuple[dict, str]:
    """Read TASI, Brent and VIX; TASI falls back to the reference CSV if Yahoo had none."""
    market = {}
    for name in ["tasi", "brent", "vix"]:
        path = raw / "market" / f"{name}.parquet"
        if path.exists():
            market[name] = pd.read_parquet(path)
    source = "yahoo"
    if "tasi" not in market:
        market["tasi"], source = read_tasi_reference(ref, cfg), "reference_csv"
    return market, source


def read_raw(raw: Path, ref: Path, cfg: dict) -> tuple[dict, dict, pd.DataFrame, pd.DataFrame, str]:
    """Read raw price, market and share files from data/raw/ (TASI may come from data/reference/)."""
    prices = {p.stem: pd.read_parquet(p) for p in sorted((raw / "prices").glob("*.parquet"))}
    market, tasi_source = read_market(raw, ref, cfg)
    share_parts = [pd.read_parquet(p).assign(symbol=p.stem) for p in sorted((raw / "shares").glob("*.parquet"))]
    shares = pd.concat(share_parts, ignore_index=True) if share_parts else pd.DataFrame(columns=["date", "shares", "symbol"])
    snap_path = raw / "shares_snapshot.csv"
    snapshot = (pd.read_csv(snap_path, dtype={"symbol": "string"}) if snap_path.exists()
                else pd.DataFrame({"symbol": pd.Series(dtype="string"), "shares_outstanding": pd.Series(dtype="float64")}))
    snapshot = snapshot.drop_duplicates("symbol", keep="first")
    return prices, market, shares, snapshot, tasi_source


def main() -> None:
    """Read data/raw/, clean it and write daily, market and weekly tables plus the summary."""
    setup_logging()
    cfg = load_config()
    ref, raw, processed = get_path("reference"), get_path("raw"), get_path("processed")
    full_uni = load_universe(ref / "universe.csv", cfg, keep_excluded=True)
    n_excluded = int(full_uni["industry_group"].isin(cfg["universe"]["exclude_groups"]).sum())
    universe = load_universe(ref / "universe.csv", cfg)
    sector_map = pd.read_csv(ref / "sector_map.csv")
    prices, market, shares, snapshot, tasi_source = read_raw(raw, ref, cfg)
    logger.info("TASI source: %s", tasi_source)
    logger.info("read %d raw price files, %d stock rows", len(prices), sum(len(df) for df in prices.values()))

    res = clean_all(prices, market, shares, snapshot, universe, sector_map, cfg)
    daily, panel, counts = res["daily"], res["panel"], res["counts"]
    logger.info("cleaning counts: %s", counts)
    checks = check_week_ends(panel, res["calendar"], cfg)
    checks["dup_symbol_date"] = int(daily.duplicated(["symbol", "date"]).sum())
    logger.info("week_end checks: %s", checks)

    processed.mkdir(parents=True, exist_ok=True)
    daily[DAILY_COLUMNS].to_parquet(processed / "daily.parquet", index=False)
    res["market_daily"].to_parquet(processed / "market_daily.parquet", index=False)
    panel.to_parquet(processed / "weekly_panel.parquet", index=False)
    logger.info("wrote daily %d rows, market_daily %d rows, weekly_panel %d rows",
                len(daily), len(res["market_daily"]), len(panel))
    have = set(daily["symbol"])
    found = {s for s in prices if s in have}
    summary = make_dataset_summary(daily, panel, res["market_daily"], universe, found, counts, n_excluded)
    summary.loc[len(summary)] = ["series", "tasi_source", tasi_source]
    write_table(summary, "dataset_summary")
    write_table(sample_ca_flags(daily), "ca_flags_to_check")


if __name__ == "__main__":
    main()
