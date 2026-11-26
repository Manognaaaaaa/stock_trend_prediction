"""Data download (docs/STEPS.md, Step 1).

Tasks:
- Build data/raw/universe.csv from the Saudi Exchange main market list plus delisted/suspended names
- Download daily OHLC, adjusted close and volume for every .SR symbol from the earliest date
- Download ^TASI.SR, BZ=F and ^VIX
- Record the earliest date for each series
- Try each delisted symbol in yfinance and record the missing count
- Save raw files untouched in data/raw/; never overwrite them
"""


def main():
    raise NotImplementedError("Step 1 not built yet")


if __name__ == "__main__":
    main()
