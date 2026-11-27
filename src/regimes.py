"""Regimes (docs/STEPS.md, Step 8), for reporting only.

A 3-component Gaussian mixture on standardised oil_ret_1m, oil_vol_20d,
tasi_vol_20d and vix_level over the test weeks. Components are named by their
means: the highest VIX is "panic", the highest oil volatility of the rest is
"oil stress", the remaining one is "calm".
"""
import logging

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.utils import get_path, load_config, setup_logging, write_table

logger = logging.getLogger(__name__)

REGIME_INPUTS = ["oil_ret_1m", "oil_vol_20d", "tasi_vol_20d", "vix_level"]


def name_components(means: pd.DataFrame) -> dict[int, str]:
    """Map component ids to calm / oil stress / panic using their mean inputs."""
    panic = int(means["vix_level"].idxmax())
    rest = means.drop(index=panic)
    oil = int(rest["oil_vol_20d"].idxmax())
    calm = int(rest.drop(index=oil).index[0])
    return {panic: "panic", oil: "oil stress", calm: "calm"}


def fit_regimes(market_weekly: pd.DataFrame, weeks: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Regime label per week, plus the component means in original units."""
    x = market_weekly.set_index("week_end").reindex(weeks)[REGIME_INPUTS].dropna()
    scaled = StandardScaler().fit_transform(x)
    gmm = GaussianMixture(n_components=cfg["regimes"]["n_components"], random_state=0, n_init=10).fit(scaled)
    comp = gmm.predict(scaled)
    means = x.groupby(comp).mean()
    names = name_components(means)
    return pd.DataFrame({"week_end": x.index, "regime": [names[c] for c in comp]})


def regime_summary(regimes: pd.DataFrame, market_weekly: pd.DataFrame) -> pd.DataFrame:
    """Weeks and mean inputs per regime."""
    d = regimes.merge(market_weekly, on="week_end")
    out = d.groupby("regime")[REGIME_INPUTS].mean()
    out.insert(0, "weeks", d.groupby("regime").size())
    return out.reset_index()


def main() -> None:
    """Fit the regimes on the test weeks and save the labels and their summary."""
    setup_logging()
    cfg = load_config()
    market_weekly = pd.read_parquet(get_path("processed") / "market_weekly.parquet")
    folds = pd.read_json(get_path("processed") / "folds.json")
    weeks = market_weekly.loc[market_weekly["week_end"] >= pd.Timestamp(folds["test_start"].min()), "week_end"]
    regimes = fit_regimes(market_weekly, pd.DatetimeIndex(weeks), cfg)
    write_table(regimes, "regimes")
    summary = regime_summary(regimes, market_weekly)
    write_table(summary, "regime_summary")
    logger.info("regimes:\n%s", summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
