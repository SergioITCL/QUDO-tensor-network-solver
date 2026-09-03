"""Generate the runtime plots and LaTeX table from experiment 4 results."""

import json
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENT_DIR / "results" / "experiment_4.json"
OUTPUT_PATH = EXPERIMENT_DIR / "processed_results" / "experiment_4.png"
LATEX_PATH = EXPERIMENT_DIR / "processed_results" / "experiment_4.tex"


def get_median_times(results: list[dict], dits: int, method: str):
    dits_results = [result for result in results if result["dits"] == dits]
    n_values = sorted({result["n_variables"] for result in dits_results})
    times = [
        median(
            result[method]["execution_time"]
            for result in dits_results
            if result["n_variables"] == n_variables
        )
        for n_variables in n_values
    ]
    return n_values, times


def main() -> None:
    payload = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    results = payload["results"]
    dits_values = payload["summary"]["dits_values"]
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
        for dits in dits_values:
            n_values, times = get_median_times(results, dits, method)
            axis.plot(n_values, times, marker="o", markersize=3, label=f"d={dits}")

        axis.set_xlabel("n")
        axis.set_ylabel("t(s)")
        axis.set_title(title)
        axis.legend()

    figure.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)

    table_rows = []
    for dits in dits_values:
        n_values, matrix_times = get_median_times(results, dits, "matrix_method")
        _, tensor_times = get_median_times(results, dits, "tensor_method")
        _, exact_times = get_median_times(results, dits, "dynamic_programming")
        for n, matrix, tensor, exact in zip(
            n_values, matrix_times, tensor_times, exact_times
        ):
            table_rows.append(
                f"{dits} & {n} & {matrix:.6f} & {tensor:.6f} & {exact:.6f} \\\\\n"
            )

    latex = (
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n"
        "\\caption{Median execution time over three random instances as a "
        "function of $n$ and $d$, with fixed $k=2$.}\n"
        "\\label{tab:experiment-4-runtime}\n"
        "\\begin{tabular}{rrrrr}\n\\toprule\n"
        "$d$ & $n$ & SMVC (s) & STC (s) & Exact DP (s) \\\\\n"
        "\\midrule\n" + "".join(table_rows) +
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    LATEX_PATH.write_text(latex, encoding="utf-8")
    print(f"Gráfica guardada en: {OUTPUT_PATH}")
    print(f"Tabla LaTeX guardada en: {LATEX_PATH}")


if __name__ == "__main__":
    main()
