"""Generate the CSV and LaTeX tables for Experiment 1."""

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENT_DIR / "results" / "experiment_1_accuracy.json"
OUTPUT_DIR = EXPERIMENT_DIR / "processed_results"


def latex_number(value: float) -> str:
    if value == 0:
        return "0"
    if abs(value) < 1e-3:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    return f"{value:.3g}"


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["results"]
    groups = defaultdict(list)
    for result in results:
        key = (result["dits"], result["n_neighbors"], result["n_variables"])
        groups[key].append(result)

    rows = []
    for (dits, n_neighbors, n_variables), instances in sorted(groups.items()):
        relative_errors = []
        optimal_count = 0
        for instance in instances:
            optimum = instance["exact_dp"]["cost"]
            cost = instance["smvc"]["cost"]
            relative_errors.append(abs(cost - optimum) / max(abs(optimum), 1e-9))
            optimal_count += math.isclose(cost, optimum, rel_tol=1e-8, abs_tol=1e-9)

        rows.append(
            {
                "d": dits,
                "k": n_neighbors,
                "n": n_variables,
                "instances": len(instances),
                "optimal_pct": 100 * optimal_count / len(instances),
                "mean_error_pct": 100 * mean(relative_errors),
                "max_error_pct": 100 * max(relative_errors),
                "time_s": mean(
                    instance["smvc"]["execution_time"] for instance in instances
                ),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "experiment_1_accuracy.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    latex_rows = [
        f"{row['d']} & {row['k']} & {row['n']} & {row['instances']} & "
        f"{row['optimal_pct']:.0f} & {latex_number(row['mean_error_pct'])} & "
        f"{latex_number(row['max_error_pct'])} & {row['time_s']:.6f} \\\\\n"
        for row in rows
    ]
    latex = (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\caption{Accuracy and execution time of SMVC relative to Exact DP on "
        "random $k$-neighbor QUDO instances. Each row aggregates "
        "50 independently generated instances.}\n"
        "\\label{tab:experiment-1-accuracy}\n"
        "\\begin{tabular}{rrrrrrrr}\n"
        "\\toprule\n"
        "$d$ & $k$ & $n$ & Instances & Optimal (\\%) & Mean error (\\%) & "
        "Max. error (\\%) & Time (s) \\\\\n"
        "\\midrule\n"
        + "".join(latex_rows)
        + "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    (OUTPUT_DIR / "experiment_1_accuracy.tex").write_text(latex, encoding="utf-8")
    print(f"Processed results saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
