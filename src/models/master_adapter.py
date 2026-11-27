"""MASTER benchmark adapter (docs/STEPS.md, Step 5).

Our own data loader for the official MASTER code (https://github.com/SJTU-DMTai/MASTER).
The model class and training loop come from that repo, cloned next to this
project (path in config `master.repo_dir`); training needs a GPU (Colab/Kaggle,
see notebooks/master_colab.ipynb).

Inputs per sample (one stock, one day): the last `lookback` days of
[13 stock features ranked per day | ~30 market features], with the label in the
last column, as the official loop expects (feature = x[:, :, :-1], label = x[:, -1, -1]).
- Stock features: features_daily_raw.parquet, ranked to [-1, 1] across stocks each day, NaN -> 0
- Market vector: mean and std over 5, 10, 20, 30, 60 days of the TASI return, the log
  change of total market traded value, and the Brent return (Brent dated strictly
  before the Saudi day, D10)
- Label: next-`label_days`-day return minus the sector mean, z-scored across stocks each day
The prediction on each week's last trading day becomes the weekly score.
"""
import argparse
import json
import logging
import sys

import numpy as np
import pandas as pd

from src.clean import week_end_map
from src.features import FEATURES, align_brent_to_saudi_days, calendar_from_market, rank_scale
from src.utils import get_path, load_config, setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------- inputs


def market_vector(daily: pd.DataFrame, market_daily: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Daily market features: rolling mean and std of TASI, market traded value and Brent."""
    cal = pd.DataFrame({"date": calendar})
    tasi = market_daily.loc[market_daily["series"] == "tasi", ["date", "ret"]].rename(columns={"ret": "tasi"})
    value = daily.groupby("date")["traded_value"].sum()
    value = np.log(value).diff().rename("value").reset_index()
    brent = align_brent_to_saudi_days(cal.assign(symbol="m"), market_daily)[["date", "brent_ret"]]
    base = cal.merge(tasi, on="date", how="left").merge(value, on="date", how="left").merge(
        brent.rename(columns={"brent_ret": "brent"}), on="date", how="left").sort_values("date")
    out = base[["date"]].copy()
    for col in ["tasi", "value", "brent"]:
        for w in cfg["master"]["market_windows"]:
            out[f"{col}_mean_{w}"] = base[col].rolling(w, min_periods=1).mean()
            out[f"{col}_std_{w}"] = base[col].rolling(w, min_periods=2).std()
    return out.fillna(0.0).reset_index(drop=True)


def daily_labels(daily: pd.DataFrame, mapping: pd.DataFrame, calendar: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Next-n-day sector-relative return, z-scored across stocks each day (NaN when a flagged day is in the window)."""
    n = cfg["master"]["label_days"]
    d = daily[["symbol", "date", "ret", "industry_group"]].sort_values(["symbol", "date"]).copy()
    grp = d.groupby("symbol")["ret"]
    fwd = sum(grp.shift(-k) for k in range(1, n + 1))  # NaN if any day is missing or flagged
    d["fwd"] = fwd
    wk = week_end_map(calendar)[["date", "week_end"]]
    d = d.merge(wk, on="date").merge(mapping[["week_end", "industry_group", "sector_group"]],
                                     on=["week_end", "industry_group"], how="left")
    d["sector_group"] = d["sector_group"].fillna(d["industry_group"])
    rel = d["fwd"] - d.groupby(["date", "sector_group"])["fwd"].transform("mean")
    by_day = rel.groupby(d["date"])
    d["label"] = (rel - by_day.transform("mean")) / by_day.transform("std")
    return d[["symbol", "date", "label"]]


def stock_inputs(feats_daily: pd.DataFrame) -> pd.DataFrame:
    """Daily stock features ranked across stocks each day, NaN filled with 0."""
    return rank_scale(feats_daily, FEATURES, by="date")


class MasterDataset:
    """Samples (stock, day) -> [lookback, n_features + 1] arrays, label in the last column.

    get_index() returns the (datetime, instrument) MultiIndex sorted by date, which the
    official DailyBatchSamplerRandom groups into one batch per day.
    """

    def __init__(self, table: pd.DataFrame, feature_cols: list[str], lookback: int, keep: pd.Series) -> None:
        """table: one row per (symbol, date) sorted by symbol then date; keep marks the samples to serve."""
        self.values = table[[*feature_cols, "label"]].to_numpy(np.float32, copy=True)
        self.values[:, -1] = np.nan_to_num(self.values[:, -1])
        row_num = table.groupby("symbol").cumcount().to_numpy()
        pos = np.flatnonzero(keep.to_numpy() & (row_num >= lookback - 1))
        order = np.lexsort((table["symbol"].to_numpy()[pos], table["date"].to_numpy()[pos]))
        self.pos = pos[order]
        self.lookback = lookback
        self.index = pd.MultiIndex.from_arrays(
            [table["date"].to_numpy()[self.pos], table["symbol"].to_numpy()[self.pos]], names=["datetime", "instrument"])

    def get_index(self) -> pd.MultiIndex:
        """(datetime, instrument) of every sample, sorted by date."""
        return self.index

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.pos)

    def __getitem__(self, i: int) -> np.ndarray:
        """The last `lookback` rows of one stock ending at sample i."""
        end = self.pos[i]
        return self.values[end - self.lookback + 1:end + 1]


def build_table(cfg: dict) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Join ranked stock features, the market vector and labels; return (table, feature columns, panel)."""
    processed = get_path("processed")
    daily = pd.read_parquet(processed / "daily.parquet")
    market_daily = pd.read_parquet(processed / "market_daily.parquet")
    panel = pd.read_parquet(processed / "weekly_panel.parquet")
    feats = stock_inputs(pd.read_parquet(processed / "features_daily_raw.parquet"))
    mapping = pd.read_csv(get_path("tables") / "sector_mapping.csv", parse_dates=["week_end"])
    calendar = calendar_from_market(market_daily)
    market = market_vector(daily, market_daily, calendar, cfg)
    labels = daily_labels(daily, mapping, calendar, cfg)
    table = feats.merge(market, on="date", how="left").merge(labels, on=["symbol", "date"], how="left")
    table = table.sort_values(["symbol", "date"]).reset_index(drop=True)
    market_cols = [c for c in market.columns if c != "date"]
    return table, [*FEATURES, *market_cols], panel


def fold_datasets(table: pd.DataFrame, cols: list[str], fold: dict, cfg: dict) -> dict:
    """Train, validation and test datasets for one fold (train/val only on rows with a label)."""
    lb = cfg["master"]["lookback"]

    def between(start: str, end: str) -> pd.Series:
        """Rows dated inside a block."""
        return table["date"].between(pd.Timestamp(start), pd.Timestamp(end))

    labelled = table["label"].notna()
    return {"train": MasterDataset(table, cols, lb, between(fold["train_start"], fold["train_end"]) & labelled),
            "valid": MasterDataset(table, cols, lb, between(fold["val_start"], fold["val_end"]) & labelled),
            "test": MasterDataset(table, cols, lb, between(fold["test_start"], fold["test_end"]))}


def weekly_scores(pred: pd.Series, panel: pd.DataFrame) -> pd.DataFrame:
    """Take each stock's prediction on its last trading day of the week."""
    p = pred.rename("score").reset_index().rename(columns={"datetime": "last_date", "instrument": "symbol"})
    rows = panel.loc[panel["eligible"], ["week_end", "symbol", "last_date"]]
    return rows.merge(p, on=["symbol", "last_date"])[["week_end", "symbol", "score"]]


# ---------------------------------------------------------------- training (needs the official repo)


def train_fold(datasets: dict, n_stock: int, n_market: int, cfg: dict, seed: int) -> pd.Series:
    """Train the official MASTERModel on one fold and seed; return test predictions."""
    from master import MASTERModel  # from the cloned official repo

    m = cfg["master"]
    model = MASTERModel(
        d_feat=n_stock, d_model=m["d_model"], t_nhead=m["t_nhead"], s_nhead=m["s_nhead"],
        T_dropout_rate=m["dropout"], S_dropout_rate=m["dropout"], beta=m["beta"],
        gate_input_start_index=n_stock, gate_input_end_index=n_stock + n_market,
        n_epochs=m["n_epochs"], lr=m["lr"], GPU=0, seed=seed,
        train_stop_loss_thred=m["train_stop_loss_thred"], save_path=str(get_path("preds") / "master_models"),
        save_prefix=f"tadawul_{seed}",
    )
    model.fit(datasets["train"], datasets["valid"])
    pred, _ = model.predict(datasets["test"])
    return pred


def main(argv: list[str] | None = None) -> None:
    """Train MASTER on every fold (seed average) and save results/preds/master_fold{k}.parquet."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Train the official MASTER model on Tadawul data.")
    parser.add_argument("--folds", nargs="*", type=int, help="only these fold numbers")
    args = parser.parse_args(argv)
    cfg = load_config()
    repo = (get_path("raw").parents[1] / cfg["master"]["repo_dir"]).resolve()
    if not (repo / "master.py").exists():
        raise FileNotFoundError(f"official MASTER code not found in {repo}: clone https://github.com/SJTU-DMTai/MASTER there")
    sys.path.insert(0, str(repo))
    table, cols, panel = build_table(cfg)
    n_stock, n_market = len(FEATURES), len(cols) - len(FEATURES)
    folds = json.loads((get_path("processed") / "folds.json").read_text())
    get_path("preds").mkdir(parents=True, exist_ok=True)
    for fold in folds:
        if args.folds and fold["fold"] not in args.folds:
            continue
        ds = fold_datasets(table, cols, fold, cfg)
        preds = [train_fold(ds, n_stock, n_market, cfg, seed) for seed in cfg["models"]["seeds"]]
        pred = pd.concat(preds, axis=1).mean(axis=1)
        out = weekly_scores(pred, panel)
        out.to_parquet(get_path("preds") / f"master_fold{fold['fold']}.parquet", index=False)
        logger.info("master fold %d: %d weekly scores", fold["fold"], len(out))


if __name__ == "__main__":
    main()
