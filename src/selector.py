"""Oil-aware selector (docs/STEPS.md, Step 6).

One row per test week, out-of-sample predictions only: each model's realised
Rank IC that week, the best model (label) and the inputs known at the start of
the week (oil and market features, each model's mean IC over the past 4 and 12
weeks, lagged one week). A small LightGBM classifier, trained with its own
walk-forward (test year T uses rows before T), gives each model's weight; the
blended score is sum(p_model x rank_model). Comparison versions are built from
the same files.
"""
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.market_features import MARKET_FEATURES
from src.models.common import BASE_MODELS, KEYS, TARGET, load_preds, weekly_rank_ic
from src.utils import get_path, load_config, setup_logging, write_table

logger = logging.getLogger(__name__)

OIL_INPUTS = ["oil_ret_1m", "oil_vol_20d"]
VERSIONS = ["selector", "simple_avg", "selector_logit", "selector_no_oil", "hard_switch", "recent_ic", "frequent_retrain"]


def ic_inputs(models: list[str], windows: list[int]) -> list[str]:
    """Names of the lagged mean-IC inputs."""
    return [f"ic{w}_{m}" for w in windows for m in models]


def selector_inputs(models: list[str], cfg: dict, oil: bool = True) -> list[str]:
    """Selector input columns (market features plus lagged IC means)."""
    market = [c for c in MARKET_FEATURES if oil or c not in OIL_INPUTS]
    return [*market, *ic_inputs(models, cfg["selector"]["ic_windows"])]


def stack_scores(models: list[str], suffix: str = "") -> pd.DataFrame:
    """One row per (week_end, symbol) with each model's score as a column."""
    out = None
    for m in models:
        p = load_preds(f"{m}{suffix}")[[*KEYS, "score"]].rename(columns={"score": m})
        out = p if out is None else out.merge(p, on=KEYS, how="outer")
    return out


def selector_table(scores: pd.DataFrame, model_table: pd.DataFrame, market_weekly: pd.DataFrame,
                   models: list[str], cfg: dict) -> pd.DataFrame:
    """Per-week realised IC per model, the best-model label and the lagged inputs."""
    d = scores.merge(model_table[[*KEYS, TARGET]], on=KEYS, how="left")
    ics = pd.concat({f"ic_{m}": weekly_rank_ic(d, m) for m in models}, axis=1).sort_index()
    ics.index.name = "week_end"
    table = ics.reset_index()
    table["best_model"] = ics[[f"ic_{m}" for m in models]].to_numpy().argmax(axis=1)
    for w in cfg["selector"]["ic_windows"]:
        for m in models:
            # IC of week t is known at the next week_end, so only weeks up to t-1 are used at t.
            table[f"ic{w}_{m}"] = table[f"ic_{m}"].rolling(w, min_periods=1).mean().shift(1)
    return table.merge(market_weekly[["week_end", *MARKET_FEATURES]], on="week_end", how="left")


def lgbm_classifier(cfg: dict) -> LGBMClassifier:
    """The small, fixed selector classifier from the config."""
    s = cfg["selector"]
    return LGBMClassifier(max_depth=s["max_depth"], num_leaves=s["num_leaves"], min_child_samples=s["min_child_samples"],
                          n_estimators=s["n_estimators"], learning_rate=s["learning_rate"], reg_lambda=s["reg_lambda"],
                          random_state=0, verbose=-1)


def logit_classifier(cfg: dict) -> object:
    """Multinomial logistic regression on standardised inputs (the check)."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=cfg["selector"]["logit_C"], max_iter=1000))


def walk_forward_proba(table: pd.DataFrame, inputs: list[str], label: str, n_classes: int,
                       make_model: Callable[[dict], object], cfg: dict, fill: bool = False) -> pd.DataFrame:
    """Class probabilities for each week from a model trained on rows before its year (NaN if too few rows)."""
    years = table["week_end"].dt.year
    x = table[inputs].fillna(0.0) if fill else table[inputs]
    out = np.full((len(table), n_classes), np.nan)
    for year in sorted(years.unique()):
        past = (years < year) & table[label].notna()
        if past.sum() < cfg["selector"]["min_train_weeks"]:
            continue
        model = make_model(cfg).fit(x.loc[past], table.loc[past, label].astype(int))
        classes = model.classes_
        now = (years == year).to_numpy()
        out[np.ix_(now, classes.astype(int))] = model.predict_proba(x.loc[now])
    return pd.DataFrame(out, index=table.index)


def fill_equal(p: pd.DataFrame) -> pd.DataFrame:
    """Rows without a trained model (or all zero) get equal weights."""
    n = p.shape[1]
    bad = p.isna().any(axis=1) | (p.fillna(0).sum(axis=1) <= 0)
    p = p.fillna(0.0)
    p.loc[bad] = 1.0 / n
    return p.div(p.sum(axis=1), axis=0)


def recent_ic_weights(table: pd.DataFrame, models: list[str], cfg: dict) -> pd.DataFrame:
    """Weights proportional to the positive part of each model's past 12-week mean IC."""
    w = max(cfg["selector"]["ic_windows"])
    raw = table[[f"ic{w}_{m}" for m in models]].clip(lower=0.0)
    raw.columns = range(len(models))
    return fill_equal(raw.where(raw.sum(axis=1) > 0))


def hard_switch(p: pd.DataFrame) -> pd.DataFrame:
    """All weight on the most likely model."""
    onehot = np.zeros(p.shape)
    onehot[np.arange(len(p)), p.to_numpy().argmax(axis=1)] = 1.0
    return pd.DataFrame(onehot, index=p.index, columns=p.columns)


def model_weights(table: pd.DataFrame, models: list[str], cfg: dict) -> dict[str, pd.DataFrame]:
    """Weekly model weights for every selector version (columns = model index)."""
    k = len(models)
    full, no_oil = selector_inputs(models, cfg), selector_inputs(models, cfg, oil=False)
    sel = fill_equal(walk_forward_proba(table, full, "best_model", k, lgbm_classifier, cfg))
    return {
        "selector": sel,
        "simple_avg": pd.DataFrame(1.0 / k, index=table.index, columns=range(k)),
        "selector_logit": fill_equal(walk_forward_proba(table, full, "best_model", k, logit_classifier, cfg, fill=True)),
        "selector_no_oil": fill_equal(walk_forward_proba(table, no_oil, "best_model", k, lgbm_classifier, cfg)),
        "hard_switch": hard_switch(sel),
        "recent_ic": recent_ic_weights(table, models, cfg),
    }


def rank01(scores: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    """Each model's score as a within-week rank scaled to [0, 1]."""
    out = scores[KEYS].copy()
    grp = scores.groupby("week_end")
    for m in models:
        n = grp[m].transform("count")
        out[m] = ((grp[m].rank(method="average") - 1) / (n - 1)).where(n > 1, 0.5)
    return out


def blend(ranks: pd.DataFrame, weights: pd.DataFrame, weeks: pd.Series, models: list[str]) -> pd.Series:
    """score = sum over models of weight x rank, with each week's weights."""
    w = weights.set_axis(models, axis=1).assign(week_end=weeks.to_numpy())
    joined = ranks[["week_end"]].merge(w, on="week_end", how="left")
    return (ranks[models].fillna(0.5).to_numpy() * joined[models].to_numpy()).sum(axis=1)


def build_selector(scores: pd.DataFrame, model_table: pd.DataFrame, market_weekly: pd.DataFrame,
                   models: list[str], cfg: dict, retrain_scores: pd.DataFrame | None = None) -> dict:
    """Selector table, weights per version and blended scores for every version."""
    table = selector_table(scores, model_table, market_weekly, models, cfg)
    weights = model_weights(table, models, cfg)
    ranks = rank01(scores, models)
    blended = scores[KEYS].copy()
    for name, w in weights.items():
        blended[name] = blend(ranks, w, table["week_end"], models)
    if retrain_scores is not None:
        r = rank01(retrain_scores, models)
        r["frequent_retrain"] = r[models].fillna(0.5).mean(axis=1)
        blended = blended.merge(r[[*KEYS, "frequent_retrain"]], on=KEYS, how="left")
    return {"table": table, "weights": weights, "scores": blended}


def weights_table(table: pd.DataFrame, weights: dict[str, pd.DataFrame], models: list[str]) -> pd.DataFrame:
    """Long-to-wide table of weekly weights: week_end, then {version}_{model} columns."""
    out = table[["week_end"]].copy()
    for name in ["selector", "selector_logit", "selector_no_oil", "recent_ic"]:
        for i, m in enumerate(models):
            out[f"{name}_{m}"] = weights[name][i].to_numpy()
    return out


def main() -> None:
    """Build the selector and every comparison version from the saved base-model predictions."""
    setup_logging()
    cfg = load_config()
    processed, preds_dir = get_path("processed"), get_path("preds")
    model_table = pd.read_parquet(processed / "model_table.parquet")
    market_weekly = pd.read_parquet(processed / "market_weekly.parquet")
    scores = stack_scores(BASE_MODELS)
    has_retrain = all(list(preds_dir.glob(f"{m}_q_fold*.parquet")) for m in BASE_MODELS)
    retrain = stack_scores(BASE_MODELS, "_q") if has_retrain else None
    if retrain is None:
        logger.warning("no quarterly-refit predictions: run python -m src.models.retrain for the frequent_retrain version")
    res = build_selector(scores, model_table, market_weekly, BASE_MODELS, cfg, retrain)
    folds = pd.read_json(processed / "folds.json")
    for _, f in folds.iterrows():
        part = res["scores"].loc[res["scores"]["week_end"].dt.year == f["test_year"]]
        part.to_parquet(preds_dir / f"selector_fold{f['fold']}.parquet", index=False)
    table = res["table"]
    write_table(table, "selector_table")
    write_table(weights_table(table, res["weights"], BASE_MODELS), "selector_weights")
    logger.info("selector: %d weeks, best-model counts %s", len(table),
                table["best_model"].map(dict(enumerate(BASE_MODELS))).value_counts().to_dict())


if __name__ == "__main__":
    main()
