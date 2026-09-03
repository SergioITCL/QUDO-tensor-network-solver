"""Load the shared experiment configuration."""

import json
from pathlib import Path

EXPERIMENTATION_DIR = Path(__file__).resolve().parent


def load_experiment(name: str) -> dict:
    """Return one experiment section from experiments.json."""
    config_path = EXPERIMENTATION_DIR / "experiments.json"
    return json.loads(config_path.read_text(encoding="utf-8"))[name]


def experiment_path(path: str) -> Path:
    """Resolve a configuration path relative to experimentation/."""
    return EXPERIMENTATION_DIR / path
