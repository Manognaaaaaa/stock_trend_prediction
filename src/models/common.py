"""Shared helpers for the base models (docs/STEPS.md, Step 4).

Every model trains on rows with a target, tunes on the validation year, and
predicts every test row (rows without a target are still needed by the
backtester). Predictions go to results/preds/{model}_fold{k}.parquet.
python -m src.models.common writes results/tables/base_models.csv.
"""
import json
import logging
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest import equal_weight_universe, run_backtest, summarize
from src.features import FEATURES
from src.splits import get_fold_rows
from src.utils import get_path, load_config, setup_logging, write_table

logger = logging.getLogger(__name__)

KEYS = ["week_end", "symbol"]
TARGET = "target_rank"
BASE_MODELS = ["ridge", "lgbm", "mlp"]


def load_inputs() -> tuple[pd.DataFrame, list[dict], dict]:
    """Read the model table, the folds and the config."""
    processed = get_path("processed")
    model_table = pd.read_parquet(processed / "model_table.parquet")
    folds = json.loads((processed / "folds.json").read_text())
    return model_table, folds, load_config()


def with_target(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that have a target (used for training and tuning)."""
    return df.loc[df[TARGET].notna()]


def weekly_rank_ic(df: pd.DataFrame, score: str = "score", target: str = TARGET) -> pd.Series:
    """Spearman correlation between score and target inside each week."""
    d = df.loc[df[target].notna() & df[score].notna(), ["week_end", score, target]]
    r = d.assign(s=d.groupby("week_end")[score].rank(), t=d.groupby("week_end")[target].rank())
    return r.groupby("week_end")[["s", "t"]].apply(lambda g: g["s"].corr(g["t"])).dropna().rename("ic")


def mean_ic(df: pd.DataFrame, score: str = "score") -> float:
    """Mean weekly Rank IC (NaN-safe)."""
    ic = weekly_rank_ic(df, score)
    return float(ic.mean()) if len(ic) else float("nan")


def preds_path(model: str, fold: int) -> Path:
    """Path of one model's predictions for one fold."""
    return get_path("preds") / f"{model}_fold{fold}.parquet"


def run_folds(model: str, fit_predict: Callable[[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, int], tuple[pd.DataFrame, dict]]) -> None:
    """Train one model on every fold, save its test predictions and the chosen settings."""
    model_table, folds, cfg = load_inputs()
    get_path("preds").mkdir(parents=True, exist_ok=True)
    chosen = []
    for fold in folds:
        train, val, test = get_fold_rows(model_table, fold)
        preds, info = fit_predict(with_target(train), with_target(val), test, cfg, fold["fold"])
        preds.to_parquet(preds_path(model, fold["fold"]), index=False)
        test_ic = mean_ic(preds.merge(test[[*KEYS, TARGET]], on=KEYS))
        chosen.append({"fold": fold["fold"], "test_year": fold["test_year"], **info, "test_ic": test_ic})
        logger.info("%s fold %d (%d): %s, test IC %.4f", model, fold["fold"], fold["test_year"], info, test_ic)
    write_table(pd.DataFrame(chosen), f"{model}_folds")


def load_preds(model: str) -> pd.DataFrame:
    """All saved test predictions of one model, every fold stacked."""
    paths = sorted(get_path("preds").glob(f"{model}_fold*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no predictions for {model}: run python -m src.models.{model}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def ic_stats(ic: pd.Series, cfg: dict) -> dict:
    """Mean Rank IC, its weekly std, ICIR and its t-statistic."""
    n = len(ic)
    return {"weeks": n, "mean_ic": ic.mean(), "ic_std": ic.std(), "icir": ic.mean() / ic.std(),
            "icir_annual": ic.mean() / ic.std() * math.sqrt(cfg["portfolio"]["weeks_per_year"]),
            "ic_t": ic.mean() / ic.std() * math.sqrt(n)}


def score_row(name: str, preds: pd.DataFrame, model_table: pd.DataFrame, cfg: dict) -> dict:
    """One results-table row: IC statistics and net backtest performance."""
    ic = weekly_rank_ic(preds.merge(model_table[[*KEYS, TARGET]], on=KEYS))
    weekly, _ = run_backtest(preds, model_table, cfg)
    weekly = weekly.set_index("week_end")
    ew = equal_weight_universe(model_table, cfg, pd.DatetimeIndex(weekly.index))
    perf = summarize(weekly["net"], cfg)
    return {"method": name, **ic_stats(ic, cfg), "net_ann_return": perf["ann_return"], "net_sharpe": perf["sharpe"],
            "mean_turnover": weekly["turnover"].mean(), "net_excess_vs_ew": (weekly["net"] - ew).mean()}


def base_models_table() -> pd.DataFrame:
    """Results table of the base models plus the equal-weight universe over the same test weeks."""
    model_table, _, cfg = load_inputs()
    rows = [score_row(m, load_preds(m), model_table, cfg) for m in BASE_MODELS if list(get_path("preds").glob(f"{m}_fold*.parquet"))]
    weeks = pd.DatetimeIndex(sorted(load_preds(BASE_MODELS[0])["week_end"].unique()))
    ew = equal_weight_universe(model_table, cfg, weeks)
    rows.append({"method": "ew_universe", **{k: v for k, v in summarize(ew, cfg).items() if k == "weeks"},
                 "net_ann_return": summarize(ew, cfg)["ann_return"], "net_sharpe": summarize(ew, cfg)["sharpe"]})
    return pd.DataFrame(rows)


def main() -> None:
    """Write results/tables/base_models.csv from the saved predictions."""
    setup_logging()
    table = base_models_table()
    logger.info("base models:\n%s", table.round(4).to_string(index=False))
    write_table(table, "base_models")


if __name__ == "__main__":
    main()
