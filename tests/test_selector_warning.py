"""Timing checks for the selector (Step 6) and the warning rule (Step 7), on small made-up tables."""
import numpy as np
import pandas as pd
import pytest

from src.evaluate import dm_test, max_drawdown
from src.selector import blend, lgbm_classifier, fill_equal, hard_switch, rank01, recent_ic_weights, selector_table, walk_forward_proba
from src.warning import exposure_rule, past_threshold

MODELS = ["ridge", "lgbm", "mlp"]


@pytest.fixture
def toy() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Three years of weekly scores for 30 stocks, a target and market features."""
    rng = np.random.default_rng(0)
    weeks = pd.date_range("2014-01-02", "2016-12-29", freq="7D")
    rows = pd.MultiIndex.from_product([weeks, [f"S{i:02d}" for i in range(30)]], names=["week_end", "symbol"]).to_frame(index=False)
    for m in MODELS:
        rows[m] = rng.normal(size=len(rows))
    mt = rows[["week_end", "symbol"]].assign(target_rank=rng.uniform(-1, 1, len(rows)))
    mw = pd.DataFrame({"week_end": weeks, **{c: rng.normal(size=len(weeks)) for c in
                       ["oil_ret_1m", "oil_vol_20d", "tasi_vol_20d", "vix_level", "vix_chg_1w"]}})
    return rows, mt, mw


def test_lagged_ic_uses_only_earlier_weeks(toy, cfg):
    """ic4/ic12 at week t average realised ICs of weeks before t only."""
    scores, mt, mw = toy
    t = selector_table(scores, mt, mw, MODELS, cfg)
    assert np.isnan(t.loc[0, "ic4_ridge"])
    assert t.loc[1, "ic4_ridge"] == pytest.approx(t.loc[0, "ic_ridge"])
    assert t.loc[10, "ic4_lgbm"] == pytest.approx(t.loc[6:9, "ic_lgbm"].mean())


def test_walk_forward_trains_on_past_years_only(toy, cfg):
    """Changing a year's labels cannot change that year's probabilities (only later years)."""
    scores, mt, mw = toy
    t = selector_table(scores, mt, mw, MODELS, cfg)
    inputs = ["oil_ret_1m", "vix_level", "ic4_ridge"]
    p1 = walk_forward_proba(t, inputs, "best_model", 3, lgbm_classifier, cfg)
    t2 = t.copy()
    in_2015 = t2["week_end"].dt.year == 2015
    t2.loc[in_2015, "best_model"] = 0
    p2 = walk_forward_proba(t2, inputs, "best_model", 3, lgbm_classifier, cfg)
    upto_2015 = t["week_end"].dt.year <= 2015
    assert np.allclose(p1[upto_2015].to_numpy(), p2[upto_2015].to_numpy(), equal_nan=True)
    assert p1[t["week_end"].dt.year == 2014].isna().all().all()  # no past rows yet


def test_blend_weights(toy):
    """Equal weights give the mean rank; hard switch copies one model's rank; weights sum to one."""
    scores, _, _ = toy
    ranks = rank01(scores, MODELS)
    weeks = pd.Series(sorted(scores["week_end"].unique()))
    eq = pd.DataFrame(1 / 3, index=weeks.index, columns=range(3))
    assert np.allclose(blend(ranks, eq, weeks, MODELS), ranks[MODELS].mean(axis=1))
    p = fill_equal(pd.DataFrame(np.tile([0.2, 0.5, 0.3], (len(weeks), 1)), index=weeks.index))
    hs = hard_switch(p)
    assert (hs[1] == 1).all() and np.allclose(hs.sum(axis=1), 1)
    assert np.allclose(blend(ranks, hs, weeks, MODELS), ranks["lgbm"])


def test_recent_ic_weights_fall_back_to_equal(cfg):
    """Negative past ICs everywhere give equal weights; otherwise weights follow the positive ICs."""
    t = pd.DataFrame({"ic12_ridge": [-0.1, 0.02], "ic12_lgbm": [-0.2, 0.06], "ic12_mlp": [np.nan, -0.01]})
    w = recent_ic_weights(t, MODELS, cfg)
    assert np.allclose(w.iloc[0], 1 / 3)
    assert np.allclose(w.iloc[1], [0.25, 0.75, 0.0])


def test_threshold_uses_strictly_past_values():
    """A week's own value never enters its threshold."""
    s = pd.Series(np.arange(40, dtype=float))
    thr = past_threshold(s, 0.2, 26)
    assert thr.iloc[:26].isna().all()
    assert thr.iloc[30] == pytest.approx(np.quantile(np.arange(30), 0.8))
    s2 = s.copy()
    s2.iloc[30] = 1e9
    assert past_threshold(s2, 0.2, 26).iloc[30] == thr.iloc[30]


def test_exposure_levels(cfg):
    """No signal 100%, one signal 50%, both 0%."""
    n = 60
    sig = pd.DataFrame({"p_bad": np.r_[np.zeros(55), [1, 0, 1, 0, 0]], "disagreement": np.r_[np.zeros(55), [1, 1, 0, 0, 0]]})
    out = exposure_rule(sig, 0.2, cfg)
    assert out["exposure"].iloc[55:].tolist() == [0.0, 0.5, 0.5, 1.0, 1.0]
    assert len(out) == n


def test_dm_and_drawdown():
    """DM stat is positive when a has higher IC; max drawdown of a known path."""
    rng = np.random.default_rng(1)
    b = pd.Series(rng.normal(0, 0.1, 300))
    a = b + 0.05
    assert dm_test(a, b, 4)["dm_stat"] > 3
    assert max_drawdown(pd.Series([0.1, -0.5, 0.2])) == pytest.approx(-0.5)
