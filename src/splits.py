"""Walk-forward folds (docs/SPEC_STEPS_0_2.md, Section 4.4).

Expanding window: train from the first usable week to the end of year T-2,
validate on T-1, test on T. The embargo is a condition on trading days (D14),
not a fixed number of weeks. python -m src.features builds and saves the folds.
"""
import pandas as pd

from src.clean import week_end_list
from src.features import main as build_all


def usable_weeks(model_table: pd.DataFrame) -> pd.DatetimeIndex:
    """Weeks with at least one eligible row with a non-NaN target_rank."""
    weeks = model_table.loc[model_table["target_rank"].notna(), "week_end"].unique()
    return pd.DatetimeIndex(sorted(weeks))


def full_years(weeks: pd.DatetimeIndex) -> set[int]:
    """Years with usable weeks in both January and December."""
    months = pd.DataFrame({"year": weeks.year, "month": weeks.month})
    has = months.groupby("year")["month"].agg(lambda m: {1, 12} <= set(m))
    return set(has[has].index)


def first_test_year(weeks: pd.DatetimeIndex, cfg: dict) -> int | None:
    """First year T with a validation year T-1 and min_train_years full years up to T-2 (D15)."""
    years, full = sorted(set(weeks.year)), full_years(weeks)
    for year in years:
        n_full = sum(1 for y in full if y <= year - 2)
        if year - 1 in years and n_full >= cfg["splits"]["min_train_years"]:
            return year
    return None


def embargo_gap(week: pd.Timestamp, next_start: pd.Timestamp, calendar: pd.DatetimeIndex,
                next_week: pd.Series) -> int:
    """Trading days d with target_end_date(week) < d <= next_start."""
    target_end = next_week.get(week)
    if pd.isna(target_end):
        return 0
    return int(calendar.searchsorted(next_start, "right") - calendar.searchsorted(target_end, "right"))


def apply_embargo(block: pd.DatetimeIndex, next_start: pd.Timestamp, calendar: pd.DatetimeIndex,
                  next_week: pd.Series, embargo_days: int) -> pd.DatetimeIndex:
    """Drop weeks from the end of a block until its last target ends embargo_days before the next block."""
    while len(block) and embargo_gap(block[-1], next_start, calendar, next_week) < embargo_days:
        block = block[:-1]
    return block


def count_rows(model_table: pd.DataFrame, block: pd.DatetimeIndex) -> int:
    """Number of model-table rows whose week_end is in the block's date range."""
    if not len(block):
        return 0
    return int(model_table["week_end"].between(block[0], block[-1]).sum())


def make_folds(model_table: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict) -> list[dict]:
    """Return the expanding-window folds with embargoed train and validation blocks."""
    weeks = usable_weeks(model_table)
    all_weeks = week_end_list(calendar)
    next_week = pd.Series(all_weeks[1:], index=all_weeks[:-1])
    first = first_test_year(weeks, cfg)
    if first is None:
        return []
    embargo = cfg["splits"]["embargo_days"]
    folds = []
    for i, test_year in enumerate(range(first, weeks.year.max() + 1), start=1):
        test = weeks[weeks.year == test_year]
        val = apply_embargo(weeks[weeks.year == test_year - 1], test[0], calendar, next_week, embargo)
        val_start = weeks[weeks.year == test_year - 1][0]
        train = apply_embargo(weeks[weeks.year <= test_year - 2], val_start, calendar, next_week, embargo)
        folds.append({
            "fold": i, "test_year": int(test_year), "val_year": int(test_year - 1),
            **{f"{name}_{edge}": block[pos].date().isoformat()
               for name, block in [("train", train), ("val", val), ("test", test)]
               for edge, pos in [("start", 0), ("end", -1)]},
            "n_train_rows": count_rows(model_table, train), "n_val_rows": count_rows(model_table, val),
            "n_test_rows": count_rows(model_table, test),
        })
    return folds


def get_fold_rows(model_table: pd.DataFrame, fold: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return the (train, validation, test) rows of one fold."""
    def block(name: str) -> pd.DataFrame:
        """Rows whose week_end lies inside the block's start and end dates."""
        start, end = pd.Timestamp(fold[f"{name}_start"]), pd.Timestamp(fold[f"{name}_end"])
        return model_table.loc[model_table["week_end"].between(start, end)]
    return block("train"), block("val"), block("test")


def main() -> None:
    """The folds are built with the rest of Step 2: run python -m src.features."""
    build_all()


if __name__ == "__main__":
    main()
