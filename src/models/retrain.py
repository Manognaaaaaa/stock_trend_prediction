"""Frequent retraining comparison (docs/STEPS.md, Step 6.4).

Each base model is refit at the start of every quarter of the test year on all
rows whose target ends at least `embargo_days` trading days before that quarter
(the fold's validation year and earlier quarters included), using the settings
tuned for that fold in Step 4. Predicts that quarter's rows and saves
results/preds/{model}_q_fold{k}.parquet.
"""
import json
import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.clean import week_end_list
from src.features import FEATURES, calendar_from_market
from src.models.common import BASE_MODELS, KEYS, TARGET, load_inputs, with_target
from src.models.lgbm import make_model
from src.models.mlp import predict, tensor, train_fixed
from src.splits import apply_embargo
from src.utils import get_path, setup_logging

logger = logging.getLogger(__name__)


def fold_settings(model: str) -> pd.DataFrame:
    """Settings chosen per fold in Step 4 (results/tables/{model}_folds.csv)."""
    return pd.read_csv(get_path("tables") / f"{model}_folds.csv").set_index("fold")


def fit_and_score(model: str, s: pd.Series, train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> np.ndarray:
    """Refit one model with its fold settings and score the test rows."""
    if model == "ridge":
        return Ridge(alpha=s["alpha"]).fit(train[FEATURES], train[TARGET]).predict(test[FEATURES])
    if model == "lgbm":
        params = {k: s[k] for k in ["num_leaves", "min_child_samples", "learning_rate", "feature_fraction", "lambda_l2"]}
        params = {**params, "num_leaves": int(params["num_leaves"]), "min_child_samples": int(params["min_child_samples"])}
        return make_model(params, cfg, n_trees=int(s["n_trees"])).fit(train[FEATURES], train[TARGET]).predict(test[FEATURES])
    epochs = json.loads(s["epochs"])
    nets = [train_fixed(train, cfg, seed, ep) for seed, ep in zip(cfg["models"]["seeds"], epochs)]
    return np.mean([predict(net, tensor(test)) for net in nets], axis=0)


def quarter_blocks(model_table: pd.DataFrame, fold: dict) -> list[pd.DatetimeIndex]:
    """The test year's weeks split by calendar quarter."""
    test = model_table.loc[model_table["week_end"].between(pd.Timestamp(fold["test_start"]), pd.Timestamp(fold["test_end"]))]
    weeks = pd.DatetimeIndex(sorted(test["week_end"].unique()))
    return [weeks[weeks.quarter == q] for q in sorted(set(weeks.quarter))]


def main() -> None:
    """Quarterly refits of every base model on every fold."""
    setup_logging()
    model_table, folds, cfg = load_inputs()
    calendar = calendar_from_market(pd.read_parquet(get_path("processed") / "market_daily.parquet"))
    all_weeks = week_end_list(calendar)
    next_week = pd.Series(all_weeks[1:], index=all_weeks[:-1])
    usable = pd.DatetimeIndex(sorted(with_target(model_table)["week_end"].unique()))
    for model in BASE_MODELS:
        settings = fold_settings(model)
        for fold in folds:
            parts = []
            for quarter in quarter_blocks(model_table, fold):
                start = quarter[0]
                train_weeks = apply_embargo(usable[(usable >= pd.Timestamp(fold["train_start"])) & (usable < start)],
                                            start, calendar, next_week, cfg["splits"]["embargo_days"])
                train = with_target(model_table.loc[model_table["week_end"].between(train_weeks[0], train_weeks[-1])])
                test = model_table.loc[model_table["week_end"].isin(quarter)]
                parts.append(test[KEYS].assign(score=fit_and_score(model, settings.loc[fold["fold"]], train, test, cfg)))
            out = pd.concat(parts, ignore_index=True)
            out.to_parquet(get_path("preds") / f"{model}_q_fold{fold['fold']}.parquet", index=False)
            logger.info("%s quarterly refits, fold %d: %d rows", model, fold["fold"], len(out))


if __name__ == "__main__":
    main()
