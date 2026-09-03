"""Generate the runtime plots and LaTeX table from experiment 5 results."""

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experimentation.experiment_config import experiment_path, load_experiment

CONFIG = load_experiment("experiment_5")


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
    results_path = experiment_path(CONFIG["results_dir"]) / CONFIG["output_file"]
    output_dir = experiment_path(CONFIG["processed_results_dir"])
    output_path = output_dir / "experiment_5.png"
    latex_path = output_dir / "experiment_5.tex"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
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
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    table_rows = []
    for n_neighbors in k_values:
        n_values, matrix_times = get_median_times(
            results, n_neighbors, "matrix_method"
        )
        _, tensor_times = get_median_times(results, n_neighbors, "tensor_method")
        _, exact_times = get_median_times(
            results, n_neighbors, "dynamic_programming"
        )
        for n, matrix, tensor, exact in zip(
            n_values, matrix_times, tensor_times, exact_times
        ):
            table_rows.append(
                f"{n_neighbors} & {n} & {matrix:.6f} & {tensor:.6f} & "
                f"{exact:.6f} \\\\\n"
            )

    latex = (
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n"
        "\\caption{Median execution time over three random instances as a "
        "function of $n$ and $k$, with fixed $d=2$.}\n"
        "\\label{tab:experiment-5-runtime}\n"
        "\\begin{tabular}{rrrrr}\n\\toprule\n"
        "$k$ & $n$ & SMVC (s) & STC (s) & Exact DP (s) \\\\\n"
        "\\midrule\n" + "".join(table_rows) +
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    latex_path.write_text(latex, encoding="utf-8")
    print(f"Plot saved to: {output_path}")
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
