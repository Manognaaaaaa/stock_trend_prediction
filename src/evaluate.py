"""Evaluation (docs/STEPS.md, Step 8).

One script for every table and figure in the report:
- Table 1: Rank IC, ICIR, precision at top 10%, NDCG, annual return, Sharpe,
  Sortino, max drawdown and turnover per method (test weeks, net of costs)
- warning metrics (AUROC, precision, recall, F1, Brier)
- Diebold-Mariano tests (loss = -weekly Rank IC, Newey-West) and block-bootstrap
  intervals for Sharpe differences
- metrics per crisis window and per GMM regime, average selector weights per regime
- robustness: liquid stocks only, without the largest stocks, costs 10/25/50 bps
- figures in results/figures/
"""
import logging
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.api as sm  # noqa: E402
from arch.bootstrap import MovingBlockBootstrap  # noqa: E402
from sklearn.metrics import brier_score_loss, f1_score, ndcg_score, precision_score, recall_score, roc_auc_score  # noqa: E402

from src.backtest import equal_weight_universe, run_backtest, tasi_weekly  # noqa: E402
from src.models.common import BASE_MODELS, KEYS, TARGET, load_preds, weekly_rank_ic  # noqa: E402
from src.regimes import fit_regimes  # noqa: E402
from src.selector import VERSIONS  # noqa: E402
from src.utils import get_path, load_config, setup_logging, write_table  # noqa: E402

logger = logging.getLogger(__name__)

WARNING_METHOD = "selector_warning"


# ---------------------------------------------------------------- inputs


def method_scores() -> dict[str, pd.DataFrame]:
    """Weekly scores of every method with saved predictions (week_end, symbol, score)."""
    preds = get_path("preds")
    out = {m: load_preds(m)[[*KEYS, "score"]] for m in BASE_MODELS}
    if list(preds.glob("master_fold*.parquet")):
        out["master"] = load_preds("master")[[*KEYS, "score"]]
    blended = pd.concat([pd.read_parquet(p) for p in sorted(preds.glob("selector_fold*.parquet"))], ignore_index=True)
    for v in VERSIONS:
        if v in blended.columns and blended[v].notna().any():
            out[v] = blended[[*KEYS, v]].rename(columns={v: "score"})
    return out


# ---------------------------------------------------------------- metrics


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative value."""
    value = (1 + r.fillna(0)).cumprod()
    return float((value / value.cummax() - 1).min())


def perf_stats(r: pd.Series, cfg: dict) -> dict:
    """Annual return, volatility, Sharpe, Sortino and max drawdown of weekly returns."""
    r = r.dropna()
    n = cfg["portfolio"]["weeks_per_year"]
    if len(r) < 2:
        return {"weeks": len(r)}
    downside = r.clip(upper=0).pow(2).mean() ** 0.5
    return {"weeks": len(r), "ann_return": (1 + r).prod() ** (n / len(r)) - 1, "ann_vol": r.std() * math.sqrt(n),
            "sharpe": r.mean() / r.std() * math.sqrt(n), "sortino": r.mean() / downside * math.sqrt(n) if downside > 0 else np.nan,
            "max_drawdown": max_drawdown(r), "cum_return": (1 + r).prod() - 1}


def topk_stats(scored: pd.DataFrame, top_pct: float) -> dict:
    """Mean weekly precision at top_pct and NDCG at the same cut-off."""
    prec, ndcg = [], []
    for _, g in scored.loc[scored[TARGET].notna() & scored["score"].notna()].groupby("week_end"):
        k = max(1, math.ceil(top_pct * len(g)))
        if len(g) < 2:
            continue
        top_pred = set(g.nlargest(k, "score")["symbol"])
        top_true = set(g.nlargest(k, TARGET)["symbol"])
        prec.append(len(top_pred & top_true) / k)
        rel = ((g[TARGET] + 1) / 2).to_numpy()[None, :]
        ndcg.append(ndcg_score(rel, g["score"].to_numpy()[None, :], k=k))
    return {"precision_top": np.mean(prec), "ndcg_top": np.mean(ndcg)}


def ic_stats(ic: pd.Series) -> dict:
    """Mean Rank IC, ICIR and t-statistic."""
    return {"mean_ic": ic.mean(), "icir": ic.mean() / ic.std(), "ic_t": ic.mean() / ic.std() * math.sqrt(len(ic))}


def backtest_all(scores: dict[str, pd.DataFrame], model_table: pd.DataFrame, cfg: dict,
                 exposure: pd.Series | None, cost_bps: float | None = None) -> dict[str, pd.DataFrame]:
    """Weekly backtest of every method (plus the selector with the warning exposure)."""
    out = {name: run_backtest(s, model_table, cfg, cost_bps=cost_bps)[0].set_index("week_end") for name, s in scores.items()}
    if exposure is not None and "selector" in scores:
        out[WARNING_METHOD] = run_backtest(scores["selector"], model_table, cfg, exposure=exposure, cost_bps=cost_bps)[0].set_index("week_end")
    return out


def benchmark_returns(model_table: pd.DataFrame, market_daily: pd.DataFrame, weeks: pd.DatetimeIndex, cfg: dict) -> dict[str, pd.Series]:
    """Equal-weight universe and TASI buy-and-hold weekly returns."""
    return {"ew_universe": equal_weight_universe(model_table, cfg, weeks), "tasi": tasi_weekly(market_daily, weeks)}


def table1(scores: dict, runs: dict, bench: dict, model_table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Main results table: ranking quality and net portfolio performance per method."""
    rows = []
    for name, weekly in runs.items():
        s = scores.get("selector" if name == WARNING_METHOD else name)
        scored = s.merge(model_table[[*KEYS, TARGET]], on=KEYS, how="left")
        rank = {**ic_stats(weekly_rank_ic(scored)), **topk_stats(scored, cfg["portfolio"]["top_pct"])}
        rows.append({"method": name, **rank, **perf_stats(weekly["net"], cfg), "turnover": weekly["turnover"].mean(),
                     "mean_exposure": weekly["exposure"].mean()})
    for name, r in bench.items():
        rows.append({"method": name, **perf_stats(r, cfg)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- significance


def dm_test(ic_a: pd.Series, ic_b: pd.Series, lags: int) -> dict:
    """Diebold-Mariano on loss = -weekly Rank IC; Newey-West standard error. Positive stat: a has lower loss."""
    d = (-ic_b) - (-ic_a)
    d = d.dropna()
    res = sm.OLS(d.to_numpy(), np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return {"mean_ic_diff": d.mean(), "dm_stat": float(res.tvalues[0]), "p_value": float(res.pvalues[0]), "weeks": len(d)}


def sharpe(r: np.ndarray, n: int) -> float:
    """Annualised Sharpe ratio of weekly returns."""
    return r.mean() / r.std(ddof=1) * math.sqrt(n)


def bootstrap_sharpe_diff(a: pd.Series, b: pd.Series, cfg: dict) -> dict:
    """Moving-block bootstrap interval for Sharpe(a) - Sharpe(b)."""
    e = cfg["evaluation"]
    n = cfg["portfolio"]["weeks_per_year"]
    both = pd.concat([a, b], axis=1).dropna()
    bs = MovingBlockBootstrap(e["block_length"], both.iloc[:, 0].to_numpy(), both.iloc[:, 1].to_numpy(), seed=e["seed"])
    diffs = np.array([sharpe(x, n) - sharpe(y, n) for (x, y), _ in bs.bootstrap(e["bootstrap_draws"])])
    lo, hi = np.quantile(diffs, [(1 - e["ci"]) / 2, 1 - (1 - e["ci"]) / 2])
    point = sharpe(both.iloc[:, 0].to_numpy(), n) - sharpe(both.iloc[:, 1].to_numpy(), n)
    return {"sharpe_diff": point, "ci_low": lo, "ci_high": hi}


def significance(scores: dict, runs: dict, model_table: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """DM tests and bootstrap Sharpe intervals: selector vs simple average, then vs each single model."""
    ics = {name: weekly_rank_ic(s.merge(model_table[[*KEYS, TARGET]], on=KEYS, how="left")) for name, s in scores.items()}
    pairs = [("selector", "simple_avg"), *[("selector", m) for m in BASE_MODELS],
             ("selector", "selector_no_oil"), ("simple_avg", "ridge")]
    rows = []
    for a, b in pairs:
        if a in runs and b in runs:
            rows.append({"a": a, "b": b, **dm_test(ics[a], ics[b], cfg["evaluation"]["dm_lags"]),
                         **bootstrap_sharpe_diff(runs[a]["net"], runs[b]["net"], cfg)})
    if WARNING_METHOD in runs:
        rows.append({"a": WARNING_METHOD, "b": "selector", **bootstrap_sharpe_diff(runs[WARNING_METHOD]["net"], runs["selector"]["net"], cfg)})
    return pd.DataFrame(rows)


def warning_metrics(exposure: pd.DataFrame) -> pd.DataFrame:
    """AUROC, precision, recall, F1 and Brier of the bad-week classifier."""
    d = exposure.dropna(subset=["bad_week", "p_bad"])
    y, p, flag = d["bad_week"].astype(int), d["p_bad"], d["p_bad_flag"].astype(int)
    return pd.DataFrame([{"weeks": len(d), "bad_week_rate": y.mean(), "auroc": roc_auc_score(y, p),
                          "precision": precision_score(y, flag, zero_division=0), "recall": recall_score(y, flag, zero_division=0),
                          "f1": f1_score(y, flag, zero_division=0), "brier": brier_score_loss(y, p)}])


# ---------------------------------------------------------------- windows, regimes, robustness


def window_table(runs: dict, bench: dict, ics: dict, labels: pd.Series, cfg: dict) -> pd.DataFrame:
    """Metrics of every method inside each labelled group of weeks (crisis window or regime)."""
    rows = []
    for group in labels.dropna().unique():
        weeks = labels.index[labels == group]
        for name, weekly in runs.items():
            r = weekly["net"].reindex(weeks)
            ic = ics.get(name, pd.Series(dtype=float)).reindex(weeks)
            rows.append({"group": group, "method": name, "mean_ic": ic.mean(), **perf_stats(r, cfg),
                         "mean_exposure": weekly["exposure"].reindex(weeks).mean()})
        for name, r in bench.items():
            rows.append({"group": group, "method": name, **perf_stats(r.reindex(weeks), cfg)})
    return pd.DataFrame(rows)


def crisis_labels(weeks: pd.DatetimeIndex, cfg: dict) -> pd.Series:
    """Crisis window name per week (NaN outside every window)."""
    out = pd.Series(np.nan, index=weeks, dtype=object)
    for name, w in cfg["crisis_windows"].items():
        out[(weeks >= pd.Timestamp(w["start"])) & (weeks <= pd.Timestamp(w["end"]))] = name
    return out


def robustness(scores: dict, model_table: pd.DataFrame, panel: pd.DataFrame, exposure: pd.Series, cfg: dict) -> pd.DataFrame:
    """Net performance with costs 10/25/50 bps, liquid stocks only and without the largest stocks."""
    r = cfg["robustness"]
    info = panel[[*KEYS, "traded_value_week", "mktval"]]
    mt = model_table.merge(info, on=KEYS, how="left")
    liquid_cut = mt.groupby("week_end")["traded_value_week"].transform(lambda s: s.quantile(r["liquidity_drop_pct"]))
    largest = mt.groupby("week_end")["mktval"].rank(ascending=False, method="first") <= r["drop_largest_n"]
    universes = {"all": mt, "liquid_only": mt.loc[mt["traded_value_week"] > liquid_cut], "without_largest": mt.loc[~largest]}
    rows = []
    for uni_name, uni in universes.items():
        costs = cfg["portfolio"]["cost_bps_grid"] if uni_name == "all" else [cfg["portfolio"]["cost_bps_base"]]
        for cost in costs:
            runs = backtest_all(scores, uni, cfg, exposure, cost_bps=cost)
            for name, weekly in runs.items():
                rows.append({"universe": uni_name, "cost_bps": cost, "method": name, **perf_stats(weekly["net"], cfg)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- figures


def shade_crises(ax: plt.Axes, cfg: dict) -> None:
    """Grey bands for the crisis windows."""
    for w in cfg["crisis_windows"].values():
        ax.axvspan(pd.Timestamp(w["start"]), pd.Timestamp(w["end"]), color="grey", alpha=0.2, lw=0)


def save_figures(runs: dict, bench: dict, ics: dict, weights: pd.DataFrame, market_daily: pd.DataFrame, cfg: dict) -> None:
    """Cumulative value, selector weights with Brent, drawdowns with and without warning, Rank IC by year."""
    fig_dir = get_path("figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    show = [m for m in ["selector", WARNING_METHOD, "simple_avg", *BASE_MODELS] if m in runs]

    fig, ax = plt.subplots(figsize=(11, 5))
    for m in show:
        ax.plot((1 + runs[m]["net"].fillna(0)).cumprod(), label=m, lw=1.2)
    for m, r in bench.items():
        ax.plot((1 + r.fillna(0)).cumprod(), label=m, lw=1, ls="--")
    shade_crises(ax, cfg)
    ax.set(title="Cumulative net value (crisis windows shaded)", ylabel="value of 1 SAR")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "cumulative_net_value.png", dpi=150)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    cols = [f"selector_{m}" for m in BASE_MODELS]
    a1.stackplot(weights["week_end"], *[weights[c] for c in cols], labels=BASE_MODELS, alpha=0.8)
    a1.set(title="Selector weights", ylabel="weight", ylim=(0, 1))
    a1.legend(fontsize=8, loc="upper left")
    brent = market_daily.loc[(market_daily["series"] == "brent") & (market_daily["date"] >= weights["week_end"].min())]
    a2.plot(brent["date"], brent["close"], color="black", lw=1)
    a2.set(ylabel="Brent (USD)")
    shade_crises(a2, cfg)
    fig.tight_layout()
    fig.savefig(fig_dir / "selector_weights_brent.png", dpi=150)
    plt.close(fig)

    if WARNING_METHOD in runs:
        fig, ax = plt.subplots(figsize=(11, 4))
        for m in ["selector", WARNING_METHOD]:
            v = (1 + runs[m]["net"].fillna(0)).cumprod()
            ax.fill_between(v.index, v / v.cummax() - 1, 0, alpha=0.4, label=m)
        shade_crises(ax, cfg)
        ax.set(title="Drawdown with and without the warning rule", ylabel="drawdown")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(fig_dir / "drawdowns_warning.png", dpi=150)
        plt.close(fig)

    by_year = pd.DataFrame({m: ics[m].groupby(ics[m].index.year).mean() for m in show if m in ics})
    ax = by_year.plot.bar(figsize=(11, 4), width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set(title="Mean weekly Rank IC by test year", xlabel="", ylabel="Rank IC")
    ax.legend(fontsize=8, ncol=3)
    ax.figure.tight_layout()
    ax.figure.savefig(fig_dir / "rank_ic_by_year.png", dpi=150)
    plt.close(ax.figure)


# ---------------------------------------------------------------- main


def main() -> None:
    """Build every evaluation table and figure from the saved predictions."""
    setup_logging()
    cfg = load_config()
    processed, tables = get_path("processed"), get_path("tables")
    model_table = pd.read_parquet(processed / "model_table.parquet")
    market_daily = pd.read_parquet(processed / "market_daily.parquet")
    market_weekly = pd.read_parquet(processed / "market_weekly.parquet")
    panel = pd.read_parquet(processed / "weekly_panel.parquet")
    exp_table = pd.read_csv(tables / "exposure.csv", parse_dates=["week_end"])
    exposure = exp_table.set_index("week_end")["exposure"]
    weights = pd.read_csv(tables / "selector_weights.csv", parse_dates=["week_end"])

    scores = method_scores()
    runs = backtest_all(scores, model_table, cfg, exposure)
    weeks = pd.DatetimeIndex(runs["selector"].index)
    bench = benchmark_returns(model_table, market_daily, weeks, cfg)
    ics = {n: weekly_rank_ic(s.merge(model_table[[*KEYS, TARGET]], on=KEYS, how="left")) for n, s in scores.items()}
    ics[WARNING_METHOD] = ics["selector"]

    t1 = table1(scores, runs, bench, model_table, cfg)
    write_table(t1, "table1")
    logger.info("table 1:\n%s", t1.round(4).to_string(index=False))
    write_table(significance(scores, runs, model_table, cfg), "significance")
    write_table(warning_metrics(exp_table), "warning_metrics")
    write_table(window_table(runs, bench, ics, crisis_labels(weeks, cfg), cfg), "crisis_windows")
    regimes = fit_regimes(market_weekly, weeks, cfg).set_index("week_end")["regime"]
    write_table(window_table(runs, bench, ics, regimes.reindex(weeks), cfg), "regime_results")
    w = weights.set_index("week_end").join(regimes, how="inner")
    write_table(w.groupby("regime")[[f"selector_{m}" for m in BASE_MODELS]].mean().reset_index(), "regime_selector_weights")
    write_table(robustness(scores, model_table, panel, exposure, cfg), "robustness")
    save_figures(runs, bench, ics, weights, market_daily, cfg)
    logger.info("evaluation tables and figures written")


if __name__ == "__main__":
    main()
