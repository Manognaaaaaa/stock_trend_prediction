"""Warning rule (docs/STEPS.md, Step 7).

Tasks:
- Bad-week probability from a second small LightGBM classifier
- Disagreement: 1 minus mean pairwise Spearman of model rankings, plus MLP seed spread
- Thresholds are percentiles of each signal on past weeks only
- Exposure 100% / 50% / 0% for zero / one / both signals above threshold
- Save results/tables/exposure.csv and results/tables/warning_sensitivity.csv (10%, 20%, 30%)
"""


def main():
    raise NotImplementedError("Step 7 not built yet")


if __name__ == "__main__":
    main()
