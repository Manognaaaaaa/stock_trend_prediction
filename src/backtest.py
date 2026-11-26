"""Backtester (docs/STEPS.md, Step 3).

Tasks:
- Remove stocks at the +10% limit on rebalance day or traded fewer than 3 days
- Buy the top 10% by score, equal weight, long only; invest `exposure`, rest in cash
- Hold one week; cost = cost_bps x turnover (sum of absolute weight changes)
- Record gross return, net return, turnover, holdings
- Benchmarks: TASI buy and hold, equal-weight universe
"""


def main():
    raise NotImplementedError("Step 3 not built yet")


if __name__ == "__main__":
    main()
