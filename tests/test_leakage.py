"""The five leakage tests (docs/SPEC_STEPS_0_2.md, Section 4.6).

Each test runs on the synthetic data and, with pytest -m data, on the real
files in data/processed/.
"""
import numpy as np
import pandas as pd
import pytest

from src.clean import week_end_list
from src.features import FEATURES, WEEKLY_KEYS, align_brent_to_saudi_days, build_features
from src.splits import embargo_gap, get_fold_rows
from src.target import compute_forward_returns

SOURCES = ["synthetic", pytest.param("real", marks=pytest.mark.data)]
N_ROWS = 20
PRICE_COLS = ["open", "high", "low", "close", "adj_close", "traded_value", "mktval"]


@pytest.fixture
def data(request: pytest.FixtureRequest, source: str) -> dict:
    """Synthetic outputs (with the synthetic config) or the real outputs."""
    if source == "real":
        return request.getfixturevalue("real")
    return {**request.getfixturevalue("synthetic"), "cfg": request.getfixturevalue("cfg")}


def sample_rows(df: pd.DataFrame, n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Pick n random rows reproducibly."""
    return df.sample(n=min(n, len(df)), random_state=seed)


def rows_of(df: pd.DataFrame, keys: pd.DataFrame) -> pd.DataFrame:
    """Rows of df for the given (week_end, symbol) keys, in key order."""
    return keys[WEEKLY_KEYS].merge(df, on=WEEKLY_KEYS, how="left")


def assert_same(a: pd.DataFrame, b: pd.DataFrame, cols: list[str]) -> None:
    """Two tables agree on cols within 1e-9, NaN equal to NaN."""
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), atol=1e-9, equal_nan=True)


def reprice(daily: pd.DataFrame, mask: pd.Series, factor: float) -> pd.DataFrame:
    """Multiply prices on masked rows and recompute the log return (flagged days stay NaN)."""
    out = daily.sort_values(["symbol", "date"]).copy()
    out.loc[mask, PRICE_COLS] = out.loc[mask, PRICE_COLS] * factor
    prev = out.groupby("symbol")["adj_close"].shift(1)
    out["ret"] = np.log(out["adj_close"] / prev).where(~out["ca_flag"])
    return out


# ---------------------------------------------------------------- 1. features


@pytest.mark.parametrize("source", SOURCES)
def test_shift_features_use_only_past_data(data: dict, source: str) -> None:
    """Features rebuilt with data up to week_end match the stored ones; later prices change nothing."""
    d, m, p, cfg = data["daily"], data["market_daily"], data["panel"], data["cfg"]
    picks = sample_rows(data["raw"])
    for week_end, rows in picks.groupby("week_end"):
        ranked, raw = build_features(d, m, p, cfg, as_of=week_end)
        assert_same(rows_of(raw, rows), rows_of(data["raw"], rows), FEATURES)
        assert_same(rows_of(ranked, rows), rows_of(data["ranked"], rows), FEATURES)

    week_end = picks["week_end"].sort_values().iloc[len(picks) // 2]
    later = d["date"] > week_end
    d2 = reprice(d, later, 2.0)
    d2.loc[d2["date"] > week_end, "volume"] *= 3
    m2 = m.copy()
    m_later = m2["date"] > week_end
    m2.loc[m_later, ["close", "ret"]] = m2.loc[m_later, ["close", "ret"]] * 2
    ranked2, raw2 = build_features(d2, m2, p, cfg)
    week = data["raw"].loc[data["raw"]["week_end"] == week_end]
    assert_same(rows_of(raw2, week), rows_of(data["raw"], week), FEATURES)
    assert_same(rows_of(ranked2, week), rows_of(data["ranked"], week), FEATURES)


# ---------------------------------------------------------------- 2. target


@pytest.mark.parametrize("source", SOURCES)
def test_target_uses_only_future_prices(data: dict, source: str) -> None:
    """fwd_ret sums daily returns in (week_end, target_end_date] and ignores later prices."""
    mt, d, cal = data["model_table"], data["daily"], data["calendar"]
    has = mt["target_end_date"].notna()
    assert (mt.loc[has, "target_end_date"] > mt.loc[has, "week_end"]).all()

    for _, r in sample_rows(mt.loc[mt["fwd_ret"].notna()]).iterrows():
        window = d.loc[(d["symbol"] == r["symbol"]) & (d["date"] > r["week_end"]) & (d["date"] <= r["target_end_date"])]
        assert r["fwd_ret"] == pytest.approx(window["ret"].sum(), abs=1e-12)

    # One row whose stock traded on target_end_date, so that day's price enters fwd_ret.
    traded = d[["symbol", "date"]].rename(columns={"date": "target_end_date"}).assign(t=True)
    cand = mt.loc[mt["fwd_ret"].notna()].merge(traded, on=["symbol", "target_end_date"])
    r = cand.sample(n=1, random_state=0).iloc[0]
    sym_daily = d.loc[d["symbol"] == r["symbol"]]
    sym_panel = data["panel"].loc[(data["panel"]["symbol"] == r["symbol"]) & (data["panel"]["week_end"] == r["week_end"])]

    def fwd(daily: pd.DataFrame) -> float:
        """fwd_ret of the chosen row computed from the given daily table."""
        return float(compute_forward_returns(daily, sym_panel, cal)["fwd_ret"].iloc[0])

    assert fwd(sym_daily) == pytest.approx(r["fwd_ret"], abs=1e-12)
    after = reprice(sym_daily, sym_daily["date"] > r["target_end_date"], 2.0)
    assert fwd(after) == pytest.approx(r["fwd_ret"], abs=1e-12)
    on_end = reprice(sym_daily, sym_daily["date"] == r["target_end_date"], 1.05)
    assert fwd(on_end) != pytest.approx(r["fwd_ret"], abs=1e-6)


# ---------------------------------------------------------------- 3. folds


@pytest.mark.parametrize("source", SOURCES)
def test_folds_do_not_overlap_and_respect_embargo(data: dict, source: str) -> None:
    """Blocks are disjoint, the embargo holds at both block ends, and targets end before the next block."""
    mt, cal, folds = data["model_table"], data["calendar"], data["folds"]
    embargo = data["cfg"]["splits"]["embargo_days"]
    assert folds, "no folds were built"
    all_weeks = week_end_list(cal)
    next_week = pd.Series(all_weeks[1:], index=all_weeks[:-1])
    for fold in folds:
        train, val, test = get_fold_rows(mt, fold)
        sets = [set(b["week_end"]) for b in (train, val, test)]
        assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
        assert embargo_gap(train["week_end"].max(), val["week_end"].min(), cal, next_week) >= embargo
        assert embargo_gap(val["week_end"].max(), test["week_end"].min(), cal, next_week) >= embargo
        assert train["target_end_date"].max() < val["week_end"].min()
        assert val["target_end_date"].max() < test["week_end"].min()
        assert len(train) == fold["n_train_rows"] and len(test) == fold["n_test_rows"]


# ---------------------------------------------------------------- 4. oil and VIX dates


@pytest.mark.parametrize("source", SOURCES)
def test_oil_and_vix_dated_before_rebalance_day(data: dict, source: str) -> None:
    """Brent and VIX values are dated strictly before the Saudi day they are used on."""
    mw = data["market_weekly"]
    assert (mw["brent_asof_date"].dropna() < mw.loc[mw["brent_asof_date"].notna(), "week_end"]).all()
    assert (mw["vix_asof_date"].dropna() < mw.loc[mw["vix_asof_date"].notna(), "week_end"]).all()
    aligned = align_brent_to_saudi_days(data["daily"][["symbol", "date"]], data["market_daily"])
    has = aligned["brent_date"].notna()
    assert has.mean() > 0.99
    assert (aligned.loc[has, "brent_date"] < aligned.loc[has, "date"]).all()


# ---------------------------------------------------------------- 5. random labels


def weekly_rank_ic(df: pd.DataFrame, pred: str, target: str) -> pd.Series:
    """Spearman correlation between pred and target inside each week."""
    ranks = df[["week_end"]].assign(p=df.groupby("week_end")[pred].rank(), t=df.groupby("week_end")[target].rank())
    return ranks.groupby("week_end").apply(lambda g: g["p"].corr(g["t"]), include_groups=False).dropna()


@pytest.mark.parametrize("source", ["synthetic", pytest.param("real", marks=[pytest.mark.data, pytest.mark.slow])])
def test_random_labels_give_zero_rank_ic(data: dict, source: str) -> None:
    """LightGBM trained on labels shuffled within each week has mean Rank IC near zero."""
    from lightgbm import LGBMRegressor

    mt = data["model_table"].loc[data["model_table"]["target_rank"].notna()].copy()
    rng = np.random.default_rng(0)
    mt["shuffled"] = mt.groupby("week_end")["target_rank"].transform(lambda s: rng.permutation(s.to_numpy()))
    train, _, test = get_fold_rows(mt, data["folds"][0])
    model = LGBMRegressor(n_estimators=200, num_leaves=15, learning_rate=0.05, verbose=-1, random_state=0)
    model.fit(train[FEATURES], train["shuffled"])
    test = test.assign(pred=model.predict(test[FEATURES]))
    ic = weekly_rank_ic(test, "pred", "shuffled")
    bound = max(0.01, 3 * ic.std() / np.sqrt(len(ic)))
    assert abs(ic.mean()) <= bound, f"mean IC {ic.mean():.4f} > {bound:.4f}"
