"""Precomputed files for the demo app (docs/STEPS.md, Step 9).

Writes small files to app/data/ so the Streamlit app never trains or reads the
full data: weekly rankings, selector weights with Brent, warning signals,
weekly net returns per method and cost level, and the crisis-window table.
Run after python -m src.evaluate.
"""
import logging

import pandas as pd

from src.evaluate import WARNING_METHOD, backtest_all, benchmark_returns, method_scores
from src.models.common import BASE_MODELS, KEYS
from src.utils import ROOT, get_path, load_config, setup_logging

logger = logging.getLogger(__name__)

APP_DATA = ROOT / "app" / "data"


def rankings(scores: dict[str, pd.DataFrame], model_table: pd.DataFrame, names: pd.DataFrame) -> pd.DataFrame:
    """Per week and stock: sector, each base model's percentile rank and the selector score."""
    out = model_table[[*KEYS, "industry_group", "sector_group"]].copy()
    for m in [*BASE_MODELS, "selector"]:
        s = scores[m].copy()
        s[m] = s.groupby("week_end")["score"].rank(pct=True)
        out = out.merge(s[[*KEYS, m]], on=KEYS, how="inner" if m == "selector" else "left")
    out = out.rename(columns={"selector": "selector_score"})
    return out.merge(names, on="symbol", how="left")


def main() -> None:
    """Write every app input file."""
    setup_logging()
    cfg = load_config()
    processed, tables = get_path("processed"), get_path("tables")
    APP_DATA.mkdir(parents=True, exist_ok=True)
    model_table = pd.read_parquet(processed / "model_table.parquet")
    market_daily = pd.read_parquet(processed / "market_daily.parquet")
    exposure_tbl = pd.read_csv(tables / "exposure.csv", parse_dates=["week_end"])
    exposure = exposure_tbl.set_index("week_end")["exposure"]
    names = pd.read_csv(get_path("reference") / "universe.csv", dtype={"symbol": "string"})[["symbol", "name"]]
    names["symbol"] = names["symbol"].str.zfill(4)
    names = names.drop_duplicates("symbol")

    scores = method_scores()
    rankings(scores, model_table, names).to_parquet(APP_DATA / "rankings.parquet", index=False)
    weights = pd.read_csv(tables / "selector_weights.csv", parse_dates=["week_end"])
    weights["week_end"] = weights["week_end"].astype("datetime64[ns]")
    brent = market_daily.loc[market_daily["series"] == "brent", ["date", "close"]].rename(columns={"date": "brent_date", "close": "brent"})
    brent["brent_date"] = brent["brent_date"].astype("datetime64[ns]")
    weights = pd.merge_asof(weights.sort_values("week_end"), brent, left_on="week_end", right_on="brent_date",
                            direction="backward", allow_exact_matches=False)
    weights.to_csv(APP_DATA / "selector_weights.csv", index=False)
    exposure_tbl.to_csv(APP_DATA / "warning.csv", index=False)

    parts = []
    for cost in cfg["portfolio"]["cost_bps_grid"]:
        runs = backtest_all(scores, model_table, cfg, exposure, cost_bps=cost)
        for name, weekly in runs.items():
            parts.append(weekly[["net"]].assign(method=name, cost_bps=cost).reset_index())
    weeks = pd.DatetimeIndex(sorted(parts[0]["week_end"].unique()))
    for name, r in benchmark_returns(model_table, market_daily, weeks, cfg).items():
        for cost in cfg["portfolio"]["cost_bps_grid"]:
            parts.append(pd.DataFrame({"week_end": weeks, "net": r.to_numpy(), "method": name, "cost_bps": cost}))
    pd.concat(parts, ignore_index=True).to_parquet(APP_DATA / "backtest.parquet", index=False)
    pd.read_csv(tables / "crisis_windows.csv").to_csv(APP_DATA / "crisis_windows.csv", index=False)
    pd.DataFrame([{"name": k, **v} for k, v in cfg["crisis_windows"].items()]).to_csv(APP_DATA / "crisis_dates.csv", index=False)
    logger.info("app data written to %s (warning method: %s)", APP_DATA, WARNING_METHOD)


if __name__ == "__main__":
    main()
