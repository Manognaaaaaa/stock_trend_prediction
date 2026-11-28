"""LightGBM base model (docs/STEPS.md, Step 4).

LGBMRegressor on the ranked target with early stopping on the validation year.
Optuna tunes num_leaves, learning_rate, min_child_samples, feature_fraction and
lambda_l2 to maximise the validation mean weekly Rank IC.
Saves results/preds/lgbm_fold{k}.parquet.
"""
import logging

import optuna
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping

from src.features import FEATURES
from src.models.common import KEYS, TARGET, mean_ic, run_folds
from src.utils import setup_logging

logger = logging.getLogger(__name__)


def make_model(params: dict, cfg: dict, n_trees: int | None = None, seed: int = 0) -> LGBMRegressor:
    """LightGBM regressor with the given tuned parameters."""
    return LGBMRegressor(
        n_estimators=n_trees or cfg["lgbm"]["max_trees"], num_leaves=params["num_leaves"],
        learning_rate=params["learning_rate"], min_child_samples=params["min_child_samples"],
        colsample_bytree=params["feature_fraction"], reg_lambda=params["lambda_l2"],
        random_state=seed, verbose=-1, n_jobs=-1,
    )


def fit_early(train: pd.DataFrame, val: pd.DataFrame, params: dict, cfg: dict) -> LGBMRegressor:
    """Fit with early stopping on the validation year's squared error."""
    model = make_model(params, cfg)
    model.fit(train[FEATURES], train[TARGET], eval_set=[(val[FEATURES], val[TARGET])],
              callbacks=[early_stopping(cfg["lgbm"]["early_stopping"], verbose=False)])
    return model


def suggest(trial: optuna.Trial, cfg: dict) -> dict:
    """Sample one parameter set from the config search ranges."""
    s = cfg["lgbm"]["search"]
    return {"num_leaves": trial.suggest_int("num_leaves", *s["num_leaves"]),
            "learning_rate": trial.suggest_float("learning_rate", *s["learning_rate"], log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", *s["min_child_samples"]),
            "feature_fraction": trial.suggest_float("feature_fraction", *s["feature_fraction"]),
            "lambda_l2": trial.suggest_float("lambda_l2", *s["lambda_l2"], log=True)}


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: dict, fold: int) -> tuple[pd.DataFrame, dict]:
    """Tune on validation with Optuna, then refit on train with the best trees and predict test."""
    def objective(trial: optuna.Trial) -> float:
        """Validation mean Rank IC of one parameter set."""
        params = suggest(trial, cfg)
        model = fit_early(train, val, params, cfg)
        trial.set_user_attr("best_iteration", int(model.best_iteration_ or cfg["lgbm"]["max_trees"]))
        return mean_ic(val.assign(score=model.predict(val[FEATURES])))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=fold))
    study.optimize(objective, n_trials=cfg["models"]["optuna_trials"])
    best = study.best_trial
    n_trees = best.user_attrs["best_iteration"]
    model = make_model(best.params, cfg, n_trees=n_trees).fit(train[FEATURES], train[TARGET])
    preds = test[KEYS].assign(score=model.predict(test[FEATURES]))
    return preds, {**{k: round(v, 5) if isinstance(v, float) else v for k, v in best.params.items()},
                   "n_trees": n_trees, "val_ic": best.value}


def main() -> None:
    """Train LightGBM on every fold and save predictions."""
    setup_logging()
    run_folds("lgbm", fit_predict)


if __name__ == "__main__":
    main()
