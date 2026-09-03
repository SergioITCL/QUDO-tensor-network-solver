"""Generate the runtime plot and LaTeX table from experiment 3 results."""

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from experimentation.experiment_config import experiment_path, load_experiment

CONFIG = load_experiment("experiment_3")

METHODS = {
    "tensor_method": ("tensor method", "#e76f7a"),
    "matrix_method": ("matrix method", "#168aad"),
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
    results_path = experiment_path(CONFIG["results_dir"]) / CONFIG["output_file"]
    output_dir = experiment_path(CONFIG["processed_results_dir"])
    output_path = output_dir / "experiment_3_kd_t.png"
    latex_path = output_dir / "experiment_3_kd_t.tex"
    payload = json.loads(results_path.read_text(encoding="utf-8"))
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
    output_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    table_rows = []
    for series in ("k", "d"):
        series_results = [result for result in results if result["series"] == series]
        for value in sorted({result["value"] for result in series_results}):
            selected = [result for result in series_results if result["value"] == value]
            sample = selected[0]
            times = [
                median(result[method]["execution_time"] for result in selected)
                for method in ("tensor_method", "matrix_method")
            ]
            table_rows.append(
                f"{series} & {value} & {sample['dits']} & {sample['n_neighbors']} & "
                f"{times[0]:.6f} & {times[1]:.6f} \\\\\n"
            )

    latex = (
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n"
        "\\caption{Median execution time over three random instances when "
        "varying $k$ or $d$ at fixed $n=100$.}\n"
        "\\label{tab:experiment-3-runtime}\n"
        "\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Varied parameter & Value & $d$ & $k$ & STC (s) & SMVC (s) \\\\\n"
        "\\midrule\n" + "".join(table_rows) +
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    latex_path.write_text(latex, encoding="utf-8")
    print(f"Plot saved to: {output_path}")
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == "__main__":
    main()
