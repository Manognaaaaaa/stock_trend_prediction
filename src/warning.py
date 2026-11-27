"""Warning rule (docs/STEPS.md, Step 7).

Two signals known at the start of each week:
1. Bad-week probability: a small LightGBM classifier with the selector's inputs
   and settings, trained walk-forward. bad_week = 1 when the selector portfolio's
   next-week net return is below the equal-weight universe.
2. Disagreement: 1 minus the mean pairwise Spearman correlation of the three
   models' rankings that week, plus the MLP seed spread (mean over stocks of the
   std across seeds of each stock's within-week rank).
A signal is raised when it is above the (1 - threshold_pct) quantile of its own
past values (weeks strictly before). Exposure: neither 100%, one 50%, both 0%.
"""
import logging

import numpy as np
import pandas as pd

from src.backtest import equal_weight_universe, run_backtest, summarize
from src.models.common import BASE_MODELS, KEYS, load_preds
from src.selector import lgbm_classifier, selector_inputs, stack_scores, walk_forward_proba
from src.utils import get_path, load_config, setup_logging, write_table

logger = logging.getLogger(__name__)

MAIN_VERSION = "selector"


def disagreement(scores: pd.DataFrame, seed_std: pd.DataFrame, models: list[str]) -> pd.Series:
    """1 - mean pairwise Spearman of the models' scores, plus the mean MLP seed spread, per week."""
    ranks = scores[["week_end"]].join(scores.groupby("week_end")[models].rank())
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    corr = ranks.groupby("week_end")[models].apply(lambda g: np.mean([g[a].corr(g[b]) for a, b in pairs]))
    spread = seed_std.groupby("week_end")["score_std"].mean()
    return ((1 - corr) + spread.reindex(corr.index)).rename("disagreement")


def past_threshold(signal: pd.Series, pct: float, min_weeks: int) -> pd.Series:
    """The (1 - pct) quantile of each week's strictly-earlier values (NaN until min_weeks exist)."""
    past = signal.shift(1).expanding(min_periods=min_weeks)
    return past.quantile(1 - pct)


def exposure_rule(signals: pd.DataFrame, pct: float, cfg: dict) -> pd.DataFrame:
    """Thresholds, raised flags and exposure for one threshold percentile."""
    w = cfg["warning"]
    out = signals.copy()
    flags = []
    for col in ["p_bad", "disagreement"]:
        out[f"{col}_thr"] = past_threshold(out[col], pct, w["min_history_weeks"])
        out[f"{col}_flag"] = (out[col] > out[f"{col}_thr"]).fillna(False)
        flags.append(out[f"{col}_flag"])
    out["n_flags"] = sum(f.astype(int) for f in flags)
    out["exposure"] = out["n_flags"].map(dict(enumerate(w["exposure_levels"])))
    return out


def bad_week_signal(table: pd.DataFrame, weekly: pd.DataFrame, ew: pd.Series, cfg: dict) -> pd.DataFrame:
    """Selector rows plus the bad_week label and its walk-forward probability."""
    t = table.merge(weekly[["week_end", "net"]], on="week_end", how="left")
    t["bad_week"] = (t["net"] < t["week_end"].map(ew)).astype(float).where(t["net"].notna() & t["week_end"].map(ew).notna())
    p = walk_forward_proba(t, selector_inputs(BASE_MODELS, cfg), "bad_week", 2, lgbm_classifier, cfg)
    t["p_bad"] = p[1]
    return t


def max_drawdown(r: pd.Series) -> float:
    """Largest peak-to-trough fall of the cumulative value."""
    value = (1 + r.fillna(0)).cumprod()
    return float((value / value.cummax() - 1).min())


def sensitivity_row(name: str, net: pd.Series, exposure: pd.Series, cfg: dict) -> dict:
    """Performance of one warning setting."""
    return {"setting": name, **summarize(net, cfg), "max_drawdown": max_drawdown(net),
            "mean_exposure": exposure.mean(), "share_weeks_50": (exposure == 0.5).mean(),
            "share_weeks_0": (exposure == 0.0).mean()}


def main() -> None:
    """Build the exposure series, the main warning backtest and the sensitivity table."""
    setup_logging()
    cfg = load_config()
    processed, tables = get_path("processed"), get_path("tables")
    model_table = pd.read_parquet(processed / "model_table.parquet")
    table = pd.read_csv(tables / "selector_table.csv", parse_dates=["week_end"])
    blended = pd.concat([pd.read_parquet(p) for p in sorted(get_path("preds").glob("selector_fold*.parquet"))])
    scores = blended[[*KEYS, MAIN_VERSION]].rename(columns={MAIN_VERSION: "score"})
    weekly, _ = run_backtest(scores, model_table, cfg)
    weeks = pd.DatetimeIndex(weekly["week_end"])
    ew = equal_weight_universe(model_table, cfg, weeks)

    sig = bad_week_signal(table, weekly, ew, cfg)
    base = stack_scores(BASE_MODELS)
    mlp = load_preds("mlp")
    sig["disagreement"] = sig["week_end"].map(disagreement(base, mlp, BASE_MODELS))
    signals = sig.set_index("week_end")[["bad_week", "p_bad", "disagreement"]]

    rows = [sensitivity_row("no_warning", weekly.set_index("week_end")["net"], pd.Series(1.0, index=weeks), cfg)]
    for pct in cfg["warning"]["threshold_grid"]:
        rule = exposure_rule(signals, pct, cfg)
        w_weekly, _ = run_backtest(scores, model_table, cfg, exposure=rule["exposure"])
        rows.append(sensitivity_row(f"threshold_{pct:.2f}", w_weekly.set_index("week_end")["net"], rule["exposure"], cfg))
        if pct == cfg["warning"]["threshold_pct"]:
            write_table(rule.reset_index(), "exposure")
    table_out = pd.DataFrame(rows)
    write_table(table_out, "warning_sensitivity")
    logger.info("warning sensitivity:\n%s", table_out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
