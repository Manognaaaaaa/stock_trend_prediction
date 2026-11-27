"""Shared helpers: config, project paths, logging and safe file writing."""
import logging
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config.yaml"
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"

logger = logging.getLogger(__name__)


def load_config() -> dict:
    """Read config.yaml and return it as a dict."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(name: str) -> Path:
    """Return the absolute path for a key in the config 'paths' section."""
    return ROOT / load_config()["paths"][name]


def setup_logging() -> None:
    """Configure root logging at INFO level with the project format."""
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def write_raw(df: pd.DataFrame, path: Path | str) -> bool:
    """Write df to Parquet only if path does not exist; never overwrite raw data."""
    path = Path(path)
    if path.exists():
        logger.info("%s skipped, exists", path)
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", index=False)
    logger.info("%s written, %d rows", path, len(df))
    return True


def write_table(df: pd.DataFrame, name: str) -> Path:
    """Write df as CSV to results/tables/{name}.csv, overwriting any old copy."""
    path = get_path("tables") / f"{name}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("%s written, %d rows", path, len(df))
    return path
