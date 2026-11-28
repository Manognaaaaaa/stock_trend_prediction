"""Ridge base model (docs/STEPS.md, Step 4).

scikit-learn Ridge on the ranked target; alpha picked on a log grid by the
validation year's mean weekly Rank IC. Saves results/preds/ridge_fold{k}.parquet.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from src.features import FEATURES
from src.models.common import KEYS, TARGET, mean_ic, run_folds
from src.utils import setup_logging


def alpha_grid(cfg: dict) -> np.ndarray:
    """Alphas from the config log grid."""
    low, high, n = cfg["ridge"]["alpha_log10_grid"]
    return np.logspace(low, high, int(n))


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: dict, fold: int) -> tuple[pd.DataFrame, dict]:
    """Pick alpha on validation, refit on train with it, predict the test rows."""
    scores = {}
    for alpha in alpha_grid(cfg):
        model = Ridge(alpha=alpha).fit(train[FEATURES], train[TARGET])
        scores[alpha] = mean_ic(val.assign(score=model.predict(val[FEATURES])))
    best = max(scores, key=scores.get)
    model = Ridge(alpha=best).fit(train[FEATURES], train[TARGET])
    preds = test[KEYS].assign(score=model.predict(test[FEATURES]))
    return preds, {"alpha": float(best), "val_ic": scores[best]}


def main() -> None:
    """Train Ridge on every fold and save predictions."""
    setup_logging()
    run_folds("ridge", fit_predict)


if __name__ == "__main__":
    main()
