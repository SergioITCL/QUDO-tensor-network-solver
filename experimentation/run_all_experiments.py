"""Run all configured experiments and their result processors."""

import json
import subprocess
import sys
from pathlib import Path

EXPERIMENTATION_DIR = Path(__file__).resolve().parent
REPOSITORY_DIR = EXPERIMENTATION_DIR.parent


def main() -> None:
    config = json.loads(
        (EXPERIMENTATION_DIR / "experiments.json").read_text(encoding="utf-8")
    )

    for name, experiment in config.items():
        print(f"\n=== {name} ===", flush=True)
        for key in ("script", "processor"):
            script = EXPERIMENTATION_DIR / experiment[key]
            subprocess.run([sys.executable, str(script)], cwd=REPOSITORY_DIR, check=True)


if __name__ == "__main__":
    main()
