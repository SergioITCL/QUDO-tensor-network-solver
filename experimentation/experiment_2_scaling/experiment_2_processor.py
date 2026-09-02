"""Create individual and comparative scaling plots from experiment 2 JSON files."""

import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
PROCESSED_RESULTS_DIR = EXPERIMENT_DIR / "processed_results"


def build_processed_rows(payload: dict) -> list[dict]:
    """Aggregate raw execution records with median times by problem size."""
    results = payload["results"]
    rows = []

    for n_variables in sorted({record["n_variables"] for record in results}):
        records = [record for record in results if record["n_variables"] == n_variables]
        exact_times = [record["exact_dynamic_programming"]["execution_time"] for record in records]
        matrix_times = [record["matrix_method"]["execution_time"] for record in records]
        matches = sum(
            math.isclose(
                record["exact_dynamic_programming"]["cost"],
                record["matrix_method"]["cost"],
                abs_tol=1e-9,
                rel_tol=1e-8,
            )
            for record in records
        )
        rows.append(
            {
                "n_variables": n_variables,
                "n_instances": len(records),
                "median_exact_time": median(exact_times),
                "median_matrix_time": median(matrix_times),
                "matrix_matches_exact": matches,
                "matrix_match_rate": matches / len(records),
            }
        )

    return rows


def save_summary_table(rows: list[dict], output_path: Path) -> None:
    fieldnames = [
        "n_variables",
        "n_instances",
        "median_exact_time",
        "median_matrix_time",
        "matrix_matches_exact",
        "matrix_match_rate",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_scaling_plot(rows: list[dict], summary: dict, output_path: Path) -> None:
    """Create one PNG per experiment configuration."""
    n_values = [row["n_variables"] for row in rows]
    exact_times = [row["median_exact_time"] for row in rows]
    matrix_times = [row["median_matrix_time"] for row in rows]

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.plot(n_values, exact_times, marker="o", linewidth=2, label="DP exacto")
    axis.plot(n_values, matrix_times, marker="o", linewidth=2, label="M?todo matricial")
    axis.set_xlabel("N?mero de variables (n)")
    axis.set_ylabel("Tiempo mediano de ejecuci?n (s)")
    axis.set_title(
        f"Escalado temporal: dits={summary['dits']}, k={summary['n_neighbors']}"
    )
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_comparison_plot(
    series: list[tuple[str, list[dict]]],
    title: str,
    output_path: Path,
) -> None:
    """Create a multi-configuration plot for the matrix method."""
    figure, axis = plt.subplots(figsize=(8, 5))

    for label, rows in series:
        axis.plot(
            [row["n_variables"] for row in rows],
            [row["median_matrix_time"] for row in rows],
            marker="o",
            linewidth=2,
            label=label,
        )

    axis.set_xlabel("N?mero de variables (n)")
    axis.set_ylabel("Tiempo mediano del m?todo matricial (s)")
    axis.set_title(title)
    axis.grid(True, linestyle="--", alpha=0.4)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def save_comparison_plots(processed_experiments: list[tuple[dict, list[dict]]]) -> None:
    """Compare dits at fixed k and k at fixed dits, like the paper figures."""
    by_neighbors: dict[int, list[tuple[dict, list[dict]]]] = {}
    by_dits: dict[int, list[tuple[dict, list[dict]]]] = {}

    for summary, rows in processed_experiments:
        by_neighbors.setdefault(summary["n_neighbors"], []).append((summary, rows))
        by_dits.setdefault(summary["dits"], []).append((summary, rows))

    for k_neighbors, experiments in by_neighbors.items():
        if len(experiments) < 2:
            continue
        series = [
            (f"d={summary['dits']}", rows)
            for summary, rows in sorted(experiments, key=lambda item: item[0]["dits"])
        ]
        save_comparison_plot(
            series,
            f"M?todo matricial: comparaci?n de dits (k={k_neighbors})",
            PROCESSED_RESULTS_DIR / f"matrix_comparison_by_dits_k{k_neighbors}.png",
        )

    for dits, experiments in by_dits.items():
        if len(experiments) < 2:
            continue
        series = [
            (f"k={summary['n_neighbors']}", rows)
            for summary, rows in sorted(experiments, key=lambda item: item[0]["n_neighbors"])
        ]
        save_comparison_plot(
            series,
            f"M?todo matricial: comparaci?n de vecinos (d={dits})",
            PROCESSED_RESULTS_DIR / f"matrix_comparison_by_neighbors_d{dits}.png",
        )


def process_file(json_path: Path) -> tuple[dict, list[dict]]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    rows = build_processed_rows(payload)
    stem = json_path.stem

    PROCESSED_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = PROCESSED_RESULTS_DIR / f"{stem}_summary.csv"
    png_path = PROCESSED_RESULTS_DIR / f"{stem}_scaling.png"
    save_summary_table(rows, csv_path)
    save_scaling_plot(rows, summary, png_path)

    print(f"Tabla: {csv_path}")
    print(f"Gr?fico: {png_path}")
    return summary, rows


def main() -> None:
    json_files = sorted(RESULTS_DIR.glob("experiment_2_d*_k*.json"))
    if not json_files:
        print(f"No se encontraron resultados JSON en: {RESULTS_DIR}")
        return

    processed_experiments = [process_file(json_path) for json_path in json_files]
    save_comparison_plots(processed_experiments)
    print(f"Gr?ficos recopilatorios: {PROCESSED_RESULTS_DIR}")


if __name__ == "__main__":
    main()
