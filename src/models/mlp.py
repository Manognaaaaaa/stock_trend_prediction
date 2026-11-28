"""MLP base model (docs/STEPS.md, Step 4).

PyTorch 13 -> 64 -> 32 -> 1, ReLU, dropout, Adam with weight decay, MSE loss on
the ranked target. Early stopping on the validation year's Rank IC; 5 seeds
averaged. Saves results/preds/mlp_fold{k}.parquet with the seed-average score
and score_std, the std across seeds of each stock's within-week rank (Step 7).
"""
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.features import FEATURES
from src.models.common import KEYS, TARGET, mean_ic, run_folds
from src.utils import setup_logging


def build_net(cfg: dict) -> nn.Sequential:
    """The MLP from the config: hidden sizes, ReLU and dropout."""
    layers, width = [], len(FEATURES)
    for hidden in cfg["mlp"]["hidden"]:
        layers += [nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(cfg["mlp"]["dropout"])]
        width = hidden
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)


def tensor(df: pd.DataFrame) -> torch.Tensor:
    """Feature matrix as a float32 tensor."""
    return torch.tensor(df[FEATURES].to_numpy(np.float32))


def predict(net: nn.Module, x: torch.Tensor) -> np.ndarray:
    """Predictions in eval mode (no dropout)."""
    net.eval()
    with torch.no_grad():
        return net(x).squeeze(1).numpy()


def train_seed(train: pd.DataFrame, val: pd.DataFrame, cfg: dict, seed: int) -> tuple[nn.Module, int, float]:
    """Train one seed; keep the epoch with the best validation Rank IC."""
    m = cfg["mlp"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = build_net(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
    x, y = tensor(train), torch.tensor(train[TARGET].to_numpy(np.float32))
    xv = tensor(val)
    best_ic, best_state, best_epoch, wait = -np.inf, None, 0, 0
    for epoch in range(1, m["max_epochs"] + 1):
        net.train()
        for idx in torch.randperm(len(x), generator=gen).split(m["batch_size"]):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(x[idx]).squeeze(1), y[idx])
            loss.backward()
            opt.step()
        ic = mean_ic(val.assign(score=predict(net, xv)))
        if ic > best_ic:
            best_ic, best_state, best_epoch, wait = ic, copy.deepcopy(net.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= m["patience"]:
                break
    net.load_state_dict(best_state)
    return net, best_epoch, best_ic


def train_fixed(train: pd.DataFrame, cfg: dict, seed: int, epochs: int) -> nn.Module:
    """Train one seed for a fixed number of epochs (used by the quarterly refits)."""
    m = cfg["mlp"]
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    net = build_net(cfg)
    opt = torch.optim.Adam(net.parameters(), lr=m["lr"], weight_decay=m["weight_decay"])
    x, y = tensor(train), torch.tensor(train[TARGET].to_numpy(np.float32))
    for _ in range(max(1, epochs)):
        net.train()
        for idx in torch.randperm(len(x), generator=gen).split(m["batch_size"]):
            opt.zero_grad()
            nn.functional.mse_loss(net(x[idx]).squeeze(1), y[idx]).backward()
            opt.step()
    return net


def seed_rank_std(test: pd.DataFrame, seed_scores: np.ndarray) -> np.ndarray:
    """Std across seeds of each row's within-week percentile rank."""
    ranks = [test[["week_end"]].assign(s=s).groupby("week_end")["s"].rank(pct=True).to_numpy() for s in seed_scores]
    return np.std(np.vstack(ranks), axis=0)


def fit_predict(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, cfg: dict, fold: int) -> tuple[pd.DataFrame, dict]:
    """Train every seed, average the test predictions and record the seed spread."""
    torch.set_num_threads(max(1, torch.get_num_threads()))
    xt = tensor(test)
    seed_scores, epochs, val_ics = [], [], []
    for seed in cfg["models"]["seeds"]:
        net, epoch, ic = train_seed(train, val, cfg, seed)
        seed_scores.append(predict(net, xt))
        epochs.append(epoch)
        val_ics.append(ic)
    seed_scores = np.vstack(seed_scores)
    preds = test[KEYS].assign(score=seed_scores.mean(axis=0), score_std=seed_rank_std(test, seed_scores))
    return preds, {"epochs": epochs, "val_ic_mean": float(np.mean(val_ics))}


def main() -> None:
    """Train the MLP on every fold and save predictions."""
    setup_logging()
    run_folds("mlp", fit_predict)


if __name__ == "__main__":
    main()
