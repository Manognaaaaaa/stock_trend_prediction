"""Shared helpers: read config.yaml and resolve project paths."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = ROOT / "config.yaml"


def load_config() -> dict:
    """Read config.yaml and return it as a dict."""
    with open(CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(name: str) -> Path:
    """Return the absolute path for a key in the config 'paths' section."""
    return ROOT / load_config()["paths"][name]
