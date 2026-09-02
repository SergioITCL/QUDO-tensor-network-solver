"""Generate the runtime plots from experiment 5 results."""

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENT_DIR / "results" / "experiment_5.json"
OUTPUT_PATH = EXPERIMENT_DIR / "processed_results" / "experiment_5.png"


def get_median_times(results: list[dict], n_neighbors: int, method: str):
    k_results = [
        result for result in results if result["n_neighbors"] == n_neighbors
    ]
    n_values = sorted({result["n_variables"] for result in k_results})
    times = [
        median(
            result[method]["execution_time"]
            for result in k_results
            if result["n_variables"] == n_variables
        )
        for n_variables in n_values
    ]
    return n_values, times


def main() -> None:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results = payload["results"]
    k_values = payload["summary"]["k_values"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 4))

    for axis, method, title in zip(
        axes,
        ("matrix_method", "tensor_method", "dynamic_programming"),
        (
            "Matrix method comparison",
            "Tensor method comparison",
            "Dynamic programming comparison",
        ),
    ):
        for n_neighbors in k_values:
            n_values, times = get_median_times(results, n_neighbors, method)
            axis.plot(
                n_values,
                times,
                marker="o",
                markersize=3,
                label=f"k={n_neighbors}",
            )

        axis.set_xlabel("n")
        axis.set_ylabel("t(s)")
        axis.set_title(title)
        axis.legend()

    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"Gráfica guardada en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
