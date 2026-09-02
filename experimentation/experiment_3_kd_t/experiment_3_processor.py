"""Generate the runtime plot and LaTeX table from experiment 3 results."""

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENT_DIR / "results" / "experiment_3_kd_t.json"
OUTPUT_PATH = EXPERIMENT_DIR / "processed_results" / "experiment_3_kd_t.png"
LATEX_PATH = EXPERIMENT_DIR / "processed_results" / "experiment_3_kd_t.tex"

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

    table_rows = []
    for series in ("k", "d"):
        series_results = [result for result in results if result["series"] == series]
        for value in sorted({result["value"] for result in series_results}):
            selected = [result for result in series_results if result["value"] == value]
            sample = selected[0]
            times = [
                median(result[method]["execution_time"] for result in selected)
                for method in ("tensor_method", "matrix_method", "dynamic_programming")
            ]
            table_rows.append(
                f"{series} & {value} & {sample['dits']} & {sample['n_neighbors']} & "
                f"{times[0]:.6f} & {times[1]:.6f} & {times[2]:.6f} \\\\\n"
            )

    latex = (
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n"
        "\\caption{Median execution time over three random instances when "
        "varying $k$ or $d$ at fixed $n=100$.}\n"
        "\\label{tab:experiment-3-runtime}\n"
        "\\begin{tabular}{lrrrrrr}\n\\toprule\n"
        "Varied parameter & Value & $d$ & $k$ & STC (s) & SMVC (s) & Exact DP (s) \\\\\n"
        "\\midrule\n" + "".join(table_rows) +
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    LATEX_PATH.write_text(latex, encoding="utf-8")
    print(f"Gráfica guardada en: {OUTPUT_PATH}")
    print(f"Tabla LaTeX guardada en: {LATEX_PATH}")


if __name__ == "__main__":
    main()
