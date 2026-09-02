"""Create instance and configuration tables for Experiment 6."""

from __future__ import annotations

import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
OUTPUT_DIR = HERE / "processed_results"

SUMMARY_FIELDS = (
    "n",
    "dits",
    "k",
    "n_instances",
    "n_target_reached",
    "target_success_rate",
    "n_target_not_reached",
    "mean_matrix_time",
    "median_matrix_time",
    "mean_time_to_target",
    "median_time_to_target",
    "min_time_to_target",
    "max_time_to_target",
    "mean_time_ratio",
    "median_time_ratio",
    "min_time_ratio",
    "max_time_ratio",
)

INSTANCE_FIELDS = (
    "n",
    "dits",
    "k",
    "instance_index",
    "seed",
    "matrix_cost",
    "matrix_time",
    "tau",
    "target_reached",
    "time_to_target",
    "final_cost",
    "target_difference",
    "time_ratio",
    "ratio_lower_bound",
    "status",
    "nodes",
    "best_bound",
    "gap",
)


def collect_tables() -> tuple[list[dict], list[dict]]:
    files = sorted(RESULTS_DIR.glob("experiment_6_params_n*_d*_k*.json"))
    if not files:
        raise FileNotFoundError(f"No Experiment 6 results found in {RESULTS_DIR}")

    summaries: list[dict] = []
    instances: list[dict] = []
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        parameters = data["parameters"]
        summary = data["summary"]
        matrix = summary["matrix_time"]
        target = summary["scip_time_to_target"]
        ratio = summary["time_ratio"]
        summaries.append(
            {
                "n": parameters["n_variables"],
                "dits": parameters["dits"],
                "k": parameters["n_neighbors"],
                "n_instances": summary["n_instances"],
                "n_target_reached": summary["n_target_reached"],
                "target_success_rate": summary["target_success_rate"],
                "n_target_not_reached": summary["n_target_not_reached"],
                "mean_matrix_time": matrix["mean"],
                "median_matrix_time": matrix["median"],
                "mean_time_to_target": target["mean"],
                "median_time_to_target": target["median"],
                "min_time_to_target": target["min"],
                "max_time_to_target": target["max"],
                "mean_time_ratio": ratio["mean"],
                "median_time_ratio": ratio["median"],
                "min_time_ratio": ratio["min"],
                "max_time_ratio": ratio["max"],
            }
        )
        for result in data["results"]:
            instances.append(
                {field: result.get(field) for field in INSTANCE_FIELDS}
            )
    return summaries, instances


def _write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(summaries: list[dict], instances: list[dict]) -> tuple[Path, ...]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_csv = OUTPUT_DIR / "configuration_summary.csv"
    instance_csv = OUTPUT_DIR / "instance_results.csv"
    markdown = OUTPUT_DIR / "configuration_summary.md"
    _write_csv(summary_csv, SUMMARY_FIELDS, summaries)
    _write_csv(instance_csv, INSTANCE_FIELDS, instances)

    lines = [
        "# Experiment 6: SCIP time-to-Matrix-TN target\n\n",
        "Timeouts are right-censored and are excluded from time-to-target and ratio statistics.\n\n",
        "| n | d | k | Reached | Success | Matrix median (s) | SCIP median (s) | Ratio median | Not reached |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in summaries:
        target_median = row["median_time_to_target"]
        ratio_median = row["median_time_ratio"]
        target_text = (
            f"{target_median:.6f}" if target_median is not None else "n/a"
        )
        ratio_text = (
            f"{ratio_median:.3f}x" if ratio_median is not None else "n/a"
        )
        lines.append(
            f"| {row['n']} | {row['dits']} | {row['k']} | "
            f"{row['n_target_reached']}/{row['n_instances']} | "
            f"{100 * row['target_success_rate']:.1f}% | "
            f"{row['median_matrix_time']:.6f} | "
            f"{target_text} | {ratio_text} | "
            f"{row['n_target_not_reached']} |\n"
        )
    markdown.write_text("".join(lines), encoding="utf-8")
    return summary_csv, instance_csv, markdown


def main() -> None:
    outputs = write_outputs(*collect_tables())
    print("Processed Experiment 6 results:")
    for output in outputs:
        print(f"  - {output}")


if __name__ == "__main__":
    main()
