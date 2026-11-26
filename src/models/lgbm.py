"""LightGBM base model (docs/STEPS.md, Step 4).

Tasks:
- LGBMRegressor on the ranked target with early stopping
- Optuna 30 to 50 trials: num_leaves, learning_rate, min_child_samples, feature_fraction, lambda_l2
- Save results/preds/lgbm_fold{k}.parquet with week, stock, score
- Run predictions through the backtester
"""


def main():
    raise NotImplementedError("Step 4 not built yet")


if __name__ == "__main__":
    main()
