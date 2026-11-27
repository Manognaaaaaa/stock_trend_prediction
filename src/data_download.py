"""Data download (docs/SPEC_STEPS_0_2.md, Section 3.1).

Downloads daily prices and share counts for every stock in
data/reference/universe.csv, plus TASI, Brent and VIX. Raw output is saved
untouched in data/raw/ and is never overwritten.

CLI:
    python -m src.data_download                      # all symbols + market series
    python -m src.data_download --symbols 2222 1120  # a few symbols + market series
    python -m src.data_download --market-only        # only TASI, Brent, VIX
"""
import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.utils import get_path, load_config, setup_logging, write_raw

logger = logging.getLogger(__name__)

UNIVERSE_COLUMNS = ["symbol", "name", "industry_group", "status", "listed_date", "delisted_date"]
LOG_COLUMNS = [
    "symbol", "yahoo_ticker", "status", "first_date", "last_date", "n_rows",
    "has_shares_full", "has_snapshot", "fetched_at", "yfinance_version",
]


def load_universe(path: Path | str, cfg: dict | None = None, keep_excluded: bool = False) -> pd.DataFrame:
    """Read universe.csv, keep symbols as 4-digit strings and drop excluded groups (D3)."""
    cfg = cfg or load_config()
    uni = pd.read_csv(path, dtype={"symbol": "string"})
    if uni.empty:
        raise ValueError(
            f"{path} has no rows. Fill it by hand from saudiexchange.sa "
            "(docs/SPEC_STEPS_0_2.md, Section 5.1)."
        )
    missing = set(UNIVERSE_COLUMNS) - set(uni.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    uni["symbol"] = uni["symbol"].str.strip().str.zfill(4)
    for col in ["listed_date", "delisted_date"]:
        uni[col] = pd.to_datetime(uni[col], format="%Y-%m-%d").astype("datetime64[ns]")
    if keep_excluded:
        return uni
    keep = ~uni["industry_group"].isin(cfg["universe"]["exclude_groups"])
    return uni.loc[keep].reset_index(drop=True)


def yahoo_symbol(symbol: str, cfg: dict | None = None) -> str:
    """Return the Yahoo ticker for a Tadawul code, e.g. '2222' -> '2222.SR'."""
    cfg = cfg or load_config()
    return f"{symbol}{cfg['tickers']['stock_suffix']}"


def download_end(cfg: dict) -> str:
    """Return the exclusive end date for Yahoo (config end_date plus one day)."""
    # yfinance treats `end` as exclusive, so add one day to include end_date itself.
    end = pd.Timestamp(cfg["data"]["end_date"]) + pd.Timedelta(days=1)
    return end.strftime("%Y-%m-%d")


def download_prices(ticker: str, start: str, end: str, cfg: dict | None = None) -> pd.DataFrame | None:
    """Download raw daily prices (auto_adjust=False, D4) with retries; None if empty."""
    cfg = cfg or load_config()
    retries, sleep_sec = cfg["download"]["retries"], cfg["download"]["sleep_sec"]
    for attempt in range(retries):
        try:
            hist = yf.Ticker(ticker).history(start=start, end=end, auto_adjust=False, actions=True)
            if hist is not None and not hist.empty:
                return hist.reset_index()
            logger.info("%s empty on attempt %d", ticker, attempt + 1)
        except Exception as exc:  # Yahoo raises many different errors
            logger.warning("%s error on attempt %d: %s", ticker, attempt + 1, exc)
        time.sleep(sleep_sec * 2**attempt)
    return None


def download_shares(ticker: str, start: str) -> tuple[pd.DataFrame | None, float | None]:
    """Return dated share counts from get_shares_full and the sharesOutstanding snapshot (D5)."""
    tk = yf.Ticker(ticker)
    full = None
    try:
        series = tk.get_shares_full(start=start)
        if series is not None and len(series) > 0:
            full = pd.DataFrame({"date": series.index, "shares": series.to_numpy()})
    except Exception as exc:
        logger.info("%s get_shares_full failed: %s", ticker, exc)
    snapshot = None
    try:
        snapshot = tk.info.get("sharesOutstanding")
    except Exception as exc:
        logger.info("%s info failed: %s", ticker, exc)
    return full, snapshot


def log_row(symbol: str, ticker: str, status: str, prices: pd.DataFrame | None,
            has_full: bool, has_snapshot: bool) -> dict:
    """Build one row of download_log.csv."""
    dates = None if prices is None else pd.to_datetime(prices["Date"])
    return {
        "symbol": symbol,
        "yahoo_ticker": ticker,
        "status": status,
        "first_date": None if dates is None else dates.min().date().isoformat(),
        "last_date": None if dates is None else dates.max().date().isoformat(),
        "n_rows": 0 if prices is None else len(prices),
        "has_shares_full": has_full,
        "has_snapshot": has_snapshot,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "yfinance_version": yf.__version__,
    }


def append_csv(rows: list[dict], path: Path, columns: list[str]) -> None:
    """Append rows to a CSV, writing the header only when the file is new."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="a", header=not path.exists(), index=False)


def known_snapshots(path: Path) -> set[str]:
    """Return the symbols already stored in shares_snapshot.csv."""
    if not path.exists():
        return set()
    return set(pd.read_csv(path, dtype={"symbol": "string"})["symbol"])


def fetch_symbol(symbol: str, cfg: dict, raw: Path, have_snapshot: set[str]) -> tuple[dict, dict | None]:
    """Download and save one stock; return its log row and a new snapshot row (or None)."""
    ticker = yahoo_symbol(symbol, cfg)
    start, end = cfg["data"]["start_date"], download_end(cfg)
    price_path = raw / "prices" / f"{symbol}.parquet"
    share_path = raw / "shares" / f"{symbol}.parquet"
    need_snapshot = symbol not in have_snapshot
    if price_path.exists() and share_path.exists() and not need_snapshot:
        logger.info("%s skipped, exists", symbol)
        return log_row(symbol, ticker, "skipped_exists", None, True, True), None

    status, prices = "skipped_exists", None
    if not price_path.exists():
        prices = download_prices(ticker, start, end, cfg)
        status = "empty" if prices is None else "ok"
        if prices is not None:
            write_raw(prices, price_path)
    else:
        logger.info("%s prices skipped, exists", symbol)

    full, snapshot = (None, None)
    if not share_path.exists() or need_snapshot:
        full, snapshot = download_shares(ticker, start)
    if full is not None and not share_path.exists():
        write_raw(full, share_path)
    snap_row = None
    if need_snapshot and snapshot is not None:
        snap_row = {"symbol": symbol, "shares_outstanding": snapshot,
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    has_full = share_path.exists()
    has_snap = (not need_snapshot) or snap_row is not None
    return log_row(symbol, ticker, status, prices, has_full, has_snap), snap_row


def fetch_market(cfg: dict, raw: Path) -> list[dict]:
    """Download TASI, Brent and VIX into data/raw/market/; return their log rows."""
    rows = []
    start, end = cfg["data"]["start_date"], download_end(cfg)
    for name in ["tasi", "brent", "vix"]:
        ticker = cfg["tickers"][name]
        path = raw / "market" / f"{name}.parquet"
        if path.exists():
            logger.info("%s skipped, exists", path)
            rows.append(log_row(name, ticker, "skipped_exists", None, False, False))
            continue
        prices = download_prices(ticker, start, end, cfg)
        if prices is not None and name == "tasi" and len(prices) < 2:
            prices = None  # Yahoo returns only today's quote for TASI, no history
        if prices is not None:
            write_raw(prices, path)
        elif name == "tasi":
            logger.warning("TASI empty on Yahoo; clean.py will read data/reference/%s instead",
                           cfg["tasi_csv"]["pattern"])
        rows.append(log_row(name, ticker, "empty" if prices is None else "ok", prices, False, False))
        time.sleep(cfg["download"]["sleep_sec"])
    return rows


def log_summary(universe: pd.DataFrame, log_rows: list[dict]) -> None:
    """Log how many listed and delisted symbols were found and how many were empty."""
    log = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    log = log.merge(universe[["symbol", "status"]].rename(columns={"status": "listing"}), on="symbol")
    for listing, grp in log.groupby("listing"):
        found = grp["status"].isin(["ok", "skipped_exists"]).sum()
        empty = (grp["status"] == "empty").sum()
        logger.info("%s: %d symbols, %d found, %d empty", listing, len(grp), found, empty)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line options."""
    parser = argparse.ArgumentParser(description="Download raw Yahoo data (never overwrites).")
    parser.add_argument("--symbols", nargs="+", help="only these Tadawul codes, e.g. 2222 1120")
    parser.add_argument("--market-only", action="store_true", help="only TASI, Brent and VIX")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Download every universe stock and the market series into data/raw/."""
    setup_logging()
    args = parse_args(argv)
    cfg = load_config()
    raw = get_path("raw")
    log_rows = fetch_market(cfg, raw)
    if not args.market_only:
        universe = load_universe(get_path("reference") / "universe.csv", cfg)
        if args.symbols:
            wanted = {s.zfill(4) for s in args.symbols}
            unknown = wanted - set(universe["symbol"])
            if unknown:
                raise ValueError(f"symbols not in universe.csv: {sorted(unknown)}")
            universe = universe.loc[universe["symbol"].isin(wanted)]
        logger.info("universe: %d symbols to fetch", len(universe))
        snap_path = raw / "shares_snapshot.csv"
        have_snapshot = known_snapshots(snap_path)
        stock_rows, snap_rows = [], []
        for symbol in universe["symbol"]:
            row, snap = fetch_symbol(symbol, cfg, raw, have_snapshot)
            stock_rows.append(row)
            if snap is not None:
                snap_rows.append(snap)
            time.sleep(cfg["download"]["sleep_sec"])
        # Only adds symbols that are not in the file yet; existing rows are never changed.
        append_csv(snap_rows, snap_path, ["symbol", "shares_outstanding", "fetched_at"])
        log_rows += stock_rows
        log_summary(universe, stock_rows)
    append_csv(log_rows, raw / "download_log.csv", LOG_COLUMNS)
    logger.info("download log: %d rows appended", len(log_rows))


if __name__ == "__main__":
    main()
