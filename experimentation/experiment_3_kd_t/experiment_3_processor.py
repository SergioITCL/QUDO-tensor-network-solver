"""Generate the k/d runtime plot from experiment 3 results."""

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENT_DIR / "results" / "experiment_3_kd_t.json"
OUTPUT_PATH = EXPERIMENT_DIR / "processed_results" / "experiment_3_kd_t.png"

METHODS = {
    "tensor_method": ("tensor method", "#e76f7a"),
    "matrix_method": ("matrix method", "#168aad"),
    "simulated_annealing": ("simulated annealing", "#f4a261"),
    "dynamic_programming": ("dynamic programming", "#4a9d45"),
}


def get_median_times(results: list[dict], series: str, method: str):
    series_results = [result for result in results if result["series"] == series]
    values = sorted({result["value"] for result in series_results})
    times = [
        median(
            result[method]["execution_time"]
            for result in series_results
            if result["value"] == value
        )
        for value in values
    ]
    return values, times


def main() -> None:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results = payload["results"]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    for axis, series in zip(axes, ("k", "d")):
        for method, (label, color) in METHODS.items():
            # Old result files created before SA was added remain processable.
            if not any(method in result for result in results):
                continue
            values, times = get_median_times(results, series, method)
            axis.plot(values, times, marker="o", markersize=3, color=color, label=label)

        axis.set_xlabel(series)
        axis.set_ylabel("t(s)")
        axis.legend()

    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Gráfica guardada en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
