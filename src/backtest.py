"""Backtester (docs/STEPS.md, Step 3).

Each week: drop stocks at the +10% limit on the rebalance day, that did not
trade on it, or that traded fewer than 3 days; buy the top 10% by score, equal
weight, long only; invest `exposure` of capital and hold the rest in cash at 0%;
hold one week; cost = cost_bps x turnover. Benchmarks: TASI buy and hold and
the equal-weight universe.
"""
import logging
import math

import numpy as np
import pandas as pd

from src.utils import get_path, load_config, setup_logging, write_table

logger = logging.getLogger(__name__)

KEYS = ["week_end", "symbol"]


def tradable_universe(model_table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Rows that can be bought on week_end, with the realised next-week simple return."""
    min_days = cfg["universe"]["min_days_traded_per_week"]
    ok = model_table["traded_on_week_end"] & ~model_table["limit_up_on_week_end"] & (model_table["days_traded"] >= min_days)
    out = model_table.loc[ok, [*KEYS, "fwd_ret"]].copy()
    # fwd_ret is NaN when the next week has no clean price (suspension, flagged day, end of data).
    # The position is then counted at 0% for that week; the share of such holdings is reported.
    out["ret_missing"] = out["fwd_ret"].isna()
    out["ret"] = np.expm1(out["fwd_ret"]).fillna(0.0)
    return out.drop(columns="fwd_ret")


def select_top(scored: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    """Keep the top top_pct of rows by score inside each week (at least one stock)."""
    s = scored.loc[scored["score"].notna()].copy()
    s["n"] = s.groupby("week_end")["score"].transform("size")
    s["rank"] = s.groupby("week_end")["score"].rank(method="first", ascending=False)
    keep = s["rank"] <= np.ceil(top_pct * s["n"]).clip(lower=1)
    return s.loc[keep, [*KEYS, "ret", "ret_missing"]]


def turnover_by_week(holdings: pd.DataFrame) -> pd.Series:
    """Sum of absolute weight changes against the previous week's weights (first week = full buy)."""
    w = holdings.pivot_table(index="week_end", columns="symbol", values="weight", fill_value=0.0).sort_index()
    return w.diff().abs().sum(axis=1).fillna(w.iloc[0].abs().sum() if len(w) else 0.0)


def run_backtest(scores: pd.DataFrame, model_table: pd.DataFrame, cfg: dict,
                 exposure: pd.Series | None = None, cost_bps: float | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest weekly scores; return (weekly results, holdings)."""
    port = cfg["portfolio"]
    cost_bps = port["cost_bps_base"] if cost_bps is None else cost_bps
    scored = tradable_universe(model_table, cfg).merge(scores[[*KEYS, "score"]], on=KEYS)
    holdings = select_top(scored, port["top_pct"])
    weeks = pd.DatetimeIndex(sorted(scores["week_end"].unique()), name="week_end")
    exp = pd.Series(1.0, index=weeks) if exposure is None else exposure.reindex(weeks).fillna(1.0)
    holdings["weight"] = holdings["week_end"].map(exp) / holdings.groupby("week_end")["symbol"].transform("size")
    holdings["contrib"] = holdings["weight"] * holdings["ret"]
    weekly = pd.DataFrame(index=weeks)
    weekly["exposure"] = exp
    weekly["gross"] = holdings.groupby("week_end")["contrib"].sum().reindex(weeks).fillna(0.0)
    weekly["turnover"] = turnover_by_week(holdings).reindex(weeks).fillna(0.0)
    weekly["cost"] = cost_bps / 1e4 * weekly["turnover"]
    weekly["net"] = weekly["gross"] - weekly["cost"]
    weekly["n_holdings"] = holdings.groupby("week_end").size().reindex(weeks).fillna(0).astype(int)
    weekly["missing_ret_share"] = holdings.groupby("week_end")["ret_missing"].mean().reindex(weeks).fillna(0.0)
    return weekly.reset_index(), holdings.reset_index(drop=True)


def equal_weight_universe(model_table: pd.DataFrame, cfg: dict, weeks: pd.DatetimeIndex | None = None) -> pd.Series:
    """Weekly return of an equal-weight portfolio of every tradable stock (gross, before costs)."""
    uni = tradable_universe(model_table, cfg)
    ret = uni.groupby("week_end")["ret"].mean().rename("ew_universe")
    return ret if weeks is None else ret.reindex(weeks)


def tasi_weekly(market_daily: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.Series:
    """TASI buy-and-hold return from each week_end close to the next week_end close."""
    tasi = market_daily.loc[market_daily["series"] == "tasi"].set_index("date")["close"]
    close = tasi.reindex(weeks)
    return (close.shift(-1) / close - 1).rename("tasi")


def summarize(net: pd.Series, cfg: dict) -> dict:
    """Annualised return, volatility and Sharpe of a weekly return series."""
    r = net.dropna()
    n = cfg["portfolio"]["weeks_per_year"]
    ann_ret = (1 + r).prod() ** (n / len(r)) - 1 if len(r) else np.nan
    ann_vol = r.std() * math.sqrt(n)
    return {"weeks": len(r), "ann_return": ann_ret, "ann_vol": ann_vol,
            "sharpe": r.mean() / r.std() * math.sqrt(n) if r.std() > 0 else np.nan}


def sanity_scores(model_table: pd.DataFrame, seed: int = 0) -> dict[str, pd.DataFrame]:
    """Scores for the three sanity checks: random, perfect foresight and 12-month momentum."""
    base = model_table[KEYS].copy()
    rng = np.random.default_rng(seed)
    return {"random": base.assign(score=rng.random(len(base))),
            "perfect_foresight": base.assign(score=model_table["target"].to_numpy()),
            "momentum_12m": base.assign(score=model_table["ret_12m"].to_numpy())}


def sanity_table(model_table: pd.DataFrame, market_daily: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Run the three sanity checks and the two benchmarks over all weeks with a target."""
    mt = model_table.loc[model_table["target"].notna()]
    weeks = pd.DatetimeIndex(sorted(mt["week_end"].unique()))
    ew = equal_weight_universe(model_table, cfg, weeks)
    rows = []
    for name, scores in sanity_scores(mt).items():
        weekly, _ = run_backtest(scores, model_table, cfg)
        net = weekly.set_index("week_end")["net"]
        excess = net - ew
        rows.append({"method": name, **summarize(net, cfg),
                     "mean_weekly_excess_vs_ew": excess.mean(), "mean_weekly_cost": weekly["cost"].mean(),
                     "mean_turnover": weekly["turnover"].mean()})
    for name, series in [("ew_universe", ew), ("tasi", tasi_weekly(market_daily, weeks))]:
        rows.append({"method": name, **summarize(series, cfg)})
    return pd.DataFrame(rows)


def main() -> None:
    """Run the backtester sanity checks on the real model table."""
    setup_logging()
    cfg = load_config()
    processed = get_path("processed")
    model_table = pd.read_parquet(processed / "model_table.parquet")
    market_daily = pd.read_parquet(processed / "market_daily.parquet")
    table = sanity_table(model_table, market_daily, cfg)
    logger.info("backtest sanity checks:\n%s", table.round(4).to_string(index=False))
    write_table(table, "backtest_sanity")


if __name__ == "__main__":
    main()
