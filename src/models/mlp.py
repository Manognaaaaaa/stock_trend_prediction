"""MLP base model (docs/STEPS.md, Step 4).

Tasks:
- PyTorch 13 -> 64 -> 32 -> 1, ReLU, dropout, Adam with weight decay, MSE loss
- Early stopping on validation Rank IC; average 5 seeds
- Save results/preds/mlp_fold{k}.parquet with week, stock, score
- Also save the std across the 5 seeds per stock (used in Step 7)
- Run predictions through the backtester
"""


def main():
    raise NotImplementedError("Step 4 not built yet")


if __name__ == "__main__":
    main()
