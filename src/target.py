"""Target (docs/SPEC_STEPS_0_2.md, Section 4.3).

target = next-week return (week_end close to next week_end close, D8) minus the
mean next-week return of the stock's sector group that week. Small industry
groups are pooled by broad group. python -m src.features builds and saves it.
"""
import pandas as pd

from src.clean import week_end_list, week_end_map
from src.features import main as build_all
from src.features import rank_scale

KEYS = ["week_end", "symbol"]
OTHER = "Other"


def assign_sector_groups(panel_week: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add sector_group to eligible rows; return (rows, mapping week_end/industry_group/sector_group/n_stocks)."""
    min_n = cfg["universe"]["min_sector_size"]
    elig = panel_week.loc[panel_week["eligible"], [*KEYS, "industry_group", "broad_group"]]
    counts = (elig.groupby(["week_end", "industry_group", "broad_group"]).size()
              .rename("n_stocks").reset_index())
    small = counts["n_stocks"] < min_n
    pool_n = counts["n_stocks"].where(small, 0).groupby([counts["week_end"], counts["broad_group"]]).transform("sum")
    counts["sector_group"] = counts["industry_group"].where(~small, counts["broad_group"].where(pool_n >= min_n, OTHER))
    counts["sector_group"] = counts["sector_group"].astype("string")
    mapping = counts[["week_end", "industry_group", "sector_group", "n_stocks"]]
    rows = elig.merge(mapping[["week_end", "industry_group", "sector_group"]], on=["week_end", "industry_group"])
    return rows, mapping.sort_values(["week_end", "industry_group"]).reset_index(drop=True)


def compute_forward_returns(daily: pd.DataFrame, panel: pd.DataFrame, calendar: pd.DatetimeIndex) -> pd.DataFrame:
    """Return fwd_ret, target_end_date and target_days for each panel row (D8, D9, D13)."""
    weeks = week_end_list(calendar)
    next_week = pd.Series(weeks[1:], index=weeks[:-1])
    cal = week_end_map(calendar)[["date", "week_end"]]
    days_in_week = cal.groupby("week_end").size()
    # Dates in (W, W'] are exactly the trading days of week W'.
    d = daily[["symbol", "date", "ret"]].merge(cal, on="date")
    d["is_nan"] = d["ret"].isna()
    agg = d.groupby(["symbol", "week_end"]).agg(
        fwd_sum=("ret", "sum"), n_rows=("ret", "size"), n_nan=("is_nan", "sum"),
    ).reset_index().rename(columns={"week_end": "target_end_date"})
    out = panel[[*KEYS, "traded_on_week_end"]].copy()
    out["target_end_date"] = out["week_end"].map(next_week)
    out = out.merge(agg, on=["symbol", "target_end_date"], how="left")
    ok = out["traded_on_week_end"] & out["target_end_date"].notna() & (out["n_rows"] > 0) & (out["n_nan"] == 0)
    out["fwd_ret"] = out["fwd_sum"].where(ok)
    out["target_days"] = out["target_end_date"].map(days_in_week).astype("Int64")
    return out[[*KEYS, "fwd_ret", "target_end_date", "target_days"]]


def make_target(panel: pd.DataFrame, fwd: pd.DataFrame) -> pd.DataFrame:
    """target = fwd_ret minus its (week_end, sector_group) mean; target_rank in [-1, 1], NaN kept."""
    out = panel.merge(fwd, on=KEYS, how="left")
    sector_mean = out.groupby(["week_end", "sector_group"])["fwd_ret"].transform("mean")
    out["target"] = out["fwd_ret"] - sector_mean
    out["target_rank"] = out["target"]
    return rank_scale(out, ["target_rank"], fill=False)


def build_target(daily: pd.DataFrame, panel: pd.DataFrame, calendar: pd.DatetimeIndex,
                 cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return target columns for eligible rows and the sector mapping table."""
    rows, mapping = assign_sector_groups(panel, cfg)
    fwd = compute_forward_returns(daily, panel.loc[panel["eligible"]], calendar)
    tgt = make_target(rows, fwd)
    cols = [*KEYS, "sector_group", "fwd_ret", "target", "target_rank", "target_end_date", "target_days"]
    return tgt[cols], mapping


def main() -> None:
    """The target is built with the rest of Step 2: run python -m src.features."""
    build_all()


if __name__ == "__main__":
    main()
