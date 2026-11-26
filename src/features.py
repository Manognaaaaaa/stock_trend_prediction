"""Stock features (docs/STEPS.md, Step 2).

Tasks:
- Momentum: ret_1w, ret_1m, ret_3m, ret_6m, ret_12m (skip latest month)
- Volatility: vol_20d, vol_60d; betas: 120-day beta_oil, beta_tasi
- Liquidity and size: turnover, amihud, vol_chg, log_mktval
- Rank each feature across stocks each week, scale to [-1, 1], fill missing with 0
- Use past data only; save data/processed/model_table.parquet
"""


def main():
    raise NotImplementedError("Step 2 not built yet")


if __name__ == "__main__":
    main()
