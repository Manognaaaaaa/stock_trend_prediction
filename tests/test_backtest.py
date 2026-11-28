"""Backtester sanity checks (docs/STEPS.md, Step 3), on the synthetic data."""
import numpy as np
import pandas as pd
import pytest

from src.backtest import equal_weight_universe, run_backtest, sanity_scores, summarize


@pytest.fixture(scope="module")
def runs(synthetic: dict, cfg: dict) -> dict:
    """Backtests of the three sanity scores plus the equal-weight universe."""
    mt = synthetic["model_table"].loc[synthetic["model_table"]["target"].notna()]
    weeks = pd.DatetimeIndex(sorted(mt["week_end"].unique()))
    out = {name: run_backtest(s, synthetic["model_table"], cfg)[0].set_index("week_end")
           for name, s in sanity_scores(mt).items()}
    out["ew"] = equal_weight_universe(synthetic["model_table"], cfg, weeks)
    return out


def test_random_scores_match_universe_minus_costs(runs: dict) -> None:
    """Random scores: net excess return over the equal-weight universe is about zero minus costs."""
    r = runs["random"]
    gross_excess = r["gross"] - runs["ew"]
    se = gross_excess.std() / np.sqrt(len(gross_excess))
    assert abs(gross_excess.mean()) < 3 * se
    assert (r["net"] - runs["ew"]).mean() == pytest.approx(gross_excess.mean() - r["cost"].mean())


def test_perfect_foresight_gives_very_large_returns(runs: dict, cfg: dict) -> None:
    """Perfect foresight: the real next-week target as the score gives a very large Sharpe."""
    assert summarize(runs["perfect_foresight"]["net"], cfg)["sharpe"] > 5


def test_momentum_only_gives_believable_result(runs: dict, cfg: dict) -> None:
    """Momentum only: ret_12m as the score gives a small, believable result."""
    assert abs(summarize(runs["momentum_12m"]["net"], cfg)["sharpe"]) < 2


def test_exposure_cash_and_costs(synthetic: dict, cfg: dict) -> None:
    """Half exposure halves gross return; weights sum to exposure; costs follow turnover."""
    mt = synthetic["model_table"]
    scores = mt[["week_end", "symbol"]].assign(score=mt["ret_1m"].to_numpy())
    full, hold = run_backtest(scores, mt, cfg)
    half, _ = run_backtest(scores, mt, cfg, exposure=pd.Series(0.5, index=pd.DatetimeIndex(full["week_end"])))
    assert np.allclose(half["gross"], full["gross"] / 2)
    assert np.allclose(hold.groupby("week_end")["weight"].sum(), 1.0)
    assert np.allclose(full["cost"], full["turnover"] * cfg["portfolio"]["cost_bps_base"] / 1e4)
    n = hold.groupby("week_end").size()
    uni = mt.loc[mt["traded_on_week_end"] & ~mt["limit_up_on_week_end"]].groupby("week_end").size()
    assert (n == np.ceil(cfg["portfolio"]["top_pct"] * uni.reindex(n.index))).all()
