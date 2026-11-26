"""Cleaning and weekly panel (docs/STEPS.md, Step 1).

Tasks:
- Drop zero-volume days; flag daily returns beyond +/-10% outside a stock's first 3 days
- Keep a stock in a week only if it traded at least 3 days; require 60 days of history
- Define each week by its last trading day (weekend changed mid 2013)
- Build one row per (stock, week): close, return, traded value, days traded, sector, size
- Align Brent and VIX to the last US close strictly before the Saudi rebalance day
- Save daily.parquet, weekly_panel.parquet and results/tables/dataset_summary.csv
"""


def main():
    raise NotImplementedError("Step 1 not built yet")


if __name__ == "__main__":
    main()
