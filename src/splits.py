"""Walk-forward folds (docs/STEPS.md, Step 2).

Tasks:
- Expanding window: train up to T-2, validate on T-1, test on T
- Drop the last 5 trading days of each block before the next block (embargo)
- Save the fold calendar to data/processed/folds.json
"""


def main():
    raise NotImplementedError("Step 2 not built yet")


if __name__ == "__main__":
    main()
