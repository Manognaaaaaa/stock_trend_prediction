"""Oil-aware selector (docs/STEPS.md, Step 6).

Tasks:
- Selector table: per-week realised IC per model, best_model label, oil/market inputs, lagged 4/12-week IC
- Own walk-forward: for test year T, train on rows before T
- Small fixed LightGBM classifier; multinomial logistic regression as a check
- Blend: score = sum of p_model x rank_model (ranks in [0, 1])
- Comparison versions: simple average, no oil, hard switch, recent-IC weights, frequent retraining
- Save results/preds/selector_fold{k}.parquet and results/tables/selector_weights.csv
"""


def main():
    raise NotImplementedError("Step 6 not built yet")


if __name__ == "__main__":
    main()
