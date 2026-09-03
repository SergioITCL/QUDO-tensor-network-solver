"""Generate the plot, CSV, and LaTeX table for Experiment 7."""

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_PATH = EXPERIMENT_DIR / "results" / "experiment_7_memory.json"
OUTPUT_DIR = EXPERIMENT_DIR / "processed_results"
METHODS = {
    "exact_dp": "Exact DP",
    "smvc": "SMVC",
    "stc": "STC",
    "beam_dp": "Beam DP",
    "tabu": "Tabu",
}


def collect_rows(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        key = (
            result["series"], result["value"], result["n_variables"],
            result["n_neighbors"], result["dits"], result["method"],
        )
        groups[key].append(result)

    rows = []
    for (series, value, n, k, d, method), measurements in sorted(groups.items()):
        row = {
            "series": series,
            "value": value,
            "n": n,
            "k": k,
            "d": d,
            "method": method,
        }
        for metric in (
            "baseline_rss_mib",
            "peak_rss_mib",
            "incremental_peak_rss_mib",
        ):
            values = [measurement[metric] for measurement in measurements]
            row[f"median_{metric}"] = median(values)
            row[f"min_{metric}"] = min(values)
            row[f"max_{metric}"] = max(values)
        rows.append(row)
    return rows


def write_latex(rows: list[dict]) -> Path:
    values = {
        (row["series"], row["value"], row["method"]): row for row in rows
    }
    configurations = {
        (row["series"], row["value"]): (row["n"], row["k"], row["d"])
        for row in rows
    }
    lines = []
    for series, value in sorted(configurations, key=lambda item: ("nkd".index(item[0]), item[1])):
        n, k, d = configurations[(series, value)]
        memory = [values[(series, value, method)] for method in METHODS]
        lines.append(
            f"{series} & {value} & {n} & {k} & {d} & "
            + " & ".join(
                f"{item['median_peak_rss_mib']:.1f} / "
                f"{item['median_incremental_peak_rss_mib']:.1f}"
                for item in memory
            )
            + " \\\\\n"
        )

    path = OUTPUT_DIR / "experiment_7_memory.tex"
    latex = (
        "\\begin{table*}[t]\n\\centering\n\\scriptsize\n"
        "\\caption{Median absolute/incremental peak RSS in MiB over three random instances.}\n"
        "\\label{tab:experiment-7-memory}\n"
        "\\begin{tabular}{lrrrrrrrrr}\n\\toprule\n"
        "Varied & Value & $n$ & $k$ & $d$ & Exact DP & SMVC & STC & Beam DP & Tabu \\\\\n"
        "\\midrule\n" + "".join(lines) +
        "\\bottomrule\n\\end{tabular}\n\\end{table*}\n"
    )
    path.write_text(latex, encoding="utf-8")
    return path


def make_plot(rows: list[dict]) -> Path:
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))

    metrics = (
        ("peak_rss_mib", "Absolute peak RSS (MiB)"),
        ("incremental_peak_rss_mib", "Incremental peak RSS (MiB)"),
    )

    for row_axes, (metric, y_label) in zip(axes, metrics):
        for axis, series in zip(row_axes, ("n", "k", "d")):
            selected = [row for row in rows if row["series"] == series]

            for method, label in METHODS.items():
                points = sorted(
                    (row for row in selected if row["method"] == method),
                    key=lambda row: row["value"],
                )

                centers = [row[f"median_{metric}"] for row in points]

                axis.errorbar(
                    [row["value"] for row in points],
                    centers,
                    yerr=[
                        [
                            center - row[f"min_{metric}"]
                            for center, row in zip(centers, points)
                        ],
                        [
                            row[f"max_{metric}"] - center
                            for center, row in zip(centers, points)
                        ],
                    ],
                    marker="o",
                    capsize=2,
                    label=label,
                )

            axis.set_xlabel(series)
            axis.set_ylabel(y_label)

            # Logarithmic scale for memory
            axis.set_yscale("log")

            axis.grid(
                True,
                which="both",
                alpha=0.3,
            )

    axes[0, 0].legend(fontsize=8)

    figure.tight_layout()

    path = OUTPUT_DIR / "experiment_7_memory.png"
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return path


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))["results"]
    rows = collect_rows(results)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "experiment_7_memory.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    outputs = (csv_path, write_latex(rows), make_plot(rows))
    print("Processed results:")
    for output in outputs:
        print(f"  - {output}")


if __name__ == "__main__":
    main()
