"""Post-process Experiment 2 raw JSON files into tables and PNG figures."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from statistics import fmean
from typing import TypedDict, cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from experimentation.experiment_config import experiment_path, load_experiment

HERE = Path(__file__).resolve().parent
CONFIG = load_experiment("experiment_2")
RESULTS_DIR = experiment_path(CONFIG["results_dir"])
LEGACY_RESULTS_DIR = HERE.parent / "experiment_1_correctness_vs_optimal" / "results"
OUTPUT_DIR = experiment_path(CONFIG["processed_results_dir"])
METHODS = (
    ("matrix_method", "Matrix method"),
    ("beam_dynamic_programming", "Beam DP"),
    ("simulated_annealing", "Simulated annealing"),
    ("tabu_search", "Tabu search"),
    ("scip", "SCIP"),
)
FIELDS: tuple[str, ...] = (
    "d", "k", "n_variables", "instance_type", "method", "n_instances",
    "feasible_solution_count", "no_incumbent_count",
    "optimal_found_count", "optimal_found_pct",
    "beam_width",
    "mean_absolute_gap", "max_absolute_gap",
    "mean_relative_error_pct", "max_relative_error_pct",
    "mean_time_s", "std_time_s", "mean_dynamic_programming_time_s",
)


class AggregationGroup(TypedDict):
    optimal: int
    absolute_gap: list[float]
    relative_error: list[float]
    time: list[float]
    dp_time: list[float]
    beam_width: list[int]
    no_incumbent: int


class SummaryRow(TypedDict):
    d: int
    k: int
    n_variables: int
    instance_type: str
    method: str
    n_instances: int
    feasible_solution_count: int
    no_incumbent_count: int
    optimal_found_count: int
    optimal_found_pct: float
    beam_width: int | None
    mean_absolute_gap: float
    max_absolute_gap: float
    mean_relative_error_pct: float
    max_relative_error_pct: float
    mean_time_s: float
    std_time_s: float
    mean_dynamic_programming_time_s: float


def _as_mapping(value: object, context: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a JSON object")
    return cast(dict[str, object], value)


def _as_list(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{context} must be a JSON array")
    return cast(list[object], value)


def _number(mapping: dict[str, object], key: str, context: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{context}.{key} must be numeric")
    return float(value)


def _integer(mapping: dict[str, object], key: str, context: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context}.{key} must be an integer")
    return value


def average(values: Sequence[float]) -> float:
    return fmean(values) if values else math.nan


def sample_std(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    value_mean = average(values)
    return math.sqrt(sum((value - value_mean) ** 2 for value in values) / (len(values) - 1))


def latex_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    if value == 0:
        return "0"
    if abs(value) < 1e-2 or abs(value) >= 1e3:
        mantissa, exponent = f"{value:.2e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    return f"{value:.3g}"


def latex_error(value: float | None, bold: bool = False) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    if value == 0:
        return "\\textbf{0}" if bold else "0"
    if abs(value) < 1e-2:
        mantissa, exponent = f"{value:.2e}".split("e")
        number = f"{mantissa}\\times10^{{{int(exponent)}}}"
        return f"$\\mathbf{{{number}}}$" if bold else f"${number}$"
    number = f"{value:.3g}"
    return f"\\textbf{{{number}}}" if bold else number


def collect_rows() -> list[SummaryRow]:
    """Aggregate the independent seeds by (d, k, n, instance type, method)."""
    # Accept both names so that results generated before/after the experiment
    # directory rename are processed from the current results directory.
    preferred_files = sorted(RESULTS_DIR.glob("experiment_2_params_d*_k*.json"))
    old_named_files = sorted(RESULTS_DIR.glob("experiment_1_params_d*_k*.json"))
    files_by_configuration = {
        path.name.split("_params_", 1)[1]: path for path in old_named_files
    }
    files_by_configuration.update(
        {path.name.split("_params_", 1)[1]: path for path in preferred_files}
    )
    files = [files_by_configuration[key] for key in sorted(files_by_configuration)]
    if not files:
        files = sorted(LEGACY_RESULTS_DIR.glob("experiment_1_params_d*_k*.json"))
    if not files:
        raise FileNotFoundError(f"No experiment result files found in {RESULTS_DIR}")

    groups: defaultdict[
        tuple[int, int, int, str, str], AggregationGroup
    ] = defaultdict(
        lambda: AggregationGroup(
            optimal=0,
            absolute_gap=[],
            relative_error=[],
            time=[],
            dp_time=[],
            beam_width=[],
            no_incumbent=0,
        )
    )
    for path in files:
        experiment = _as_mapping(
            json.loads(path.read_text(encoding="utf-8")), str(path)
        )
        parameters = _as_mapping(
            experiment.get("parameters"), f"{path}.parameters"
        )
        d = _integer(parameters, "dits", f"{path}.parameters")
        k = _integer(parameters, "n_neighbors", f"{path}.parameters")
        raw_results = _as_list(experiment.get("results"), f"{path}.results")
        for index, raw_result in enumerate(raw_results):
            result_context = f"{path}.results[{index}]"
            result = _as_mapping(raw_result, result_context)
            raw_instance_type = result.get("instance_type", "unspecified")
            instance_type = (
                raw_instance_type
                if isinstance(raw_instance_type, str)
                else "unspecified"
            )
            n_variables = _integer(result, "n_variables", result_context)
            dynamic_programming = _as_mapping(
                result.get("dynamic_programming"),
                f"{result_context}.dynamic_programming",
            )
            for name, label in METHODS:
                # Keep result files produced before a method was added usable.
                if name not in result:
                    continue
                method = _as_mapping(result[name], f"{result_context}.{name}")
                group = groups[(d, k, n_variables, instance_type, label)]
                group["time"].append(_number(method, "time", name))
                group["dp_time"].append(
                    _number(dynamic_programming, "time", "dynamic_programming")
                )
                raw_beam_width = method.get("beam_width")
                if raw_beam_width is not None:
                    if isinstance(raw_beam_width, bool) or not isinstance(
                        raw_beam_width, int
                    ):
                        raise TypeError(f"{result_context}.{name}.beam_width must be an integer")
                    group["beam_width"].append(raw_beam_width)
                # A very small matched budget can expire while SCIP is still
                # building its model, before any feasible incumbent exists.
                if method.get("cost") is None:
                    group["no_incumbent"] += 1
                    continue
                # SMVC metrics are at root; Beam DP metrics are nested.
                metrics = result if name == "matrix_method" else method
                reached_optimal = metrics.get("reached_optimal")
                if not isinstance(reached_optimal, bool):
                    raise TypeError(
                        f"{result_context}.{name}.reached_optimal must be boolean"
                    )
                group["optimal"] += int(reached_optimal)
                group["absolute_gap"].append(
                    abs(_number(metrics, "cost_difference", name))
                )
                group["relative_error"].append(
                    abs(_number(metrics, "relative_gap", name))
                )

    rows: list[SummaryRow] = []
    for (d, k, n, instance_type, method), group in sorted(groups.items()):
        count = len(group["time"])
        rows.append({
            "d": d, "k": k, "n_variables": n,
            "instance_type": instance_type, "method": method,
            "n_instances": count, "optimal_found_count": group["optimal"],
            "feasible_solution_count": len(group["absolute_gap"]),
            "no_incumbent_count": group["no_incumbent"],
            "optimal_found_pct": 100 * group["optimal"] / count,
            "beam_width": (
                group["beam_width"][0] if group["beam_width"] else None
            ),
            "mean_absolute_gap": average(group["absolute_gap"]),
            "max_absolute_gap": max(group["absolute_gap"], default=math.nan),
            "mean_relative_error_pct": 100 * average(group["relative_error"]),
            "max_relative_error_pct": 100 * max(
                group["relative_error"], default=math.nan
            ),
            "mean_time_s": average(group["time"]),
            "std_time_s": sample_std(group["time"]),
            "mean_dynamic_programming_time_s": average(group["dp_time"]),
        })
    return rows


def write_tables(rows: Sequence[SummaryRow]) -> tuple[Path, Path, Path]:
    csv_path = OUTPUT_DIR / "summary_by_d_k_n.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    markdown_path = OUTPUT_DIR / "summary_by_d_k_n.md"
    lines = [
        "# Experiment 1: correctness versus optimum\n\n",
        "Each row aggregates independent seeds for one fixed $(d,k,n)$ configuration and instance type. Dynamic programming is the reference optimum.\n\n",
        "| $d$ | $k$ | $n$ | Instance type | Method | Instances | Feasible | Optimal found | Beam width | Mean abs. gap | Max abs. gap | Mean relative error | Max relative error | Mean time (s) |\n",
        "|---:|---:|---:|:---|:---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for row in rows:
        lines.append(
            f"| {row['d']} | {row['k']} | {row['n_variables']} | "
            f"{row['instance_type']} | {row['method']} | "
            f"{row['n_instances']} | {row['feasible_solution_count']}/{row['n_instances']} | "
            f"{row['optimal_found_count']}/{row['n_instances']} ({row['optimal_found_pct']:.1f}%) | "
            f"{row['beam_width'] if row['beam_width'] is not None else '-'} | "
            f"{row['mean_absolute_gap']:.3g} | {row['max_absolute_gap']:.3g} | "
            f"{row['mean_relative_error_pct']:.3g}% | {row['max_relative_error_pct']:.3g}% | {row['mean_time_s']:.6f} |\n"
        )
    markdown_path.write_text("".join(lines), encoding="utf-8")

    selected = {
        (row["d"], row["k"], row["method"]): row
        for row in rows
        if row["n_variables"] == 500 and row["instance_type"] == "random"
    }
    latex_rows: list[str] = []
    configurations = sorted({(d, k) for d, k, _ in selected})
    previous_d: int | None = None
    for d, k in configurations:
        smvc = selected[(d, k, "Matrix method")]
        beam = selected[(d, k, "Beam DP")]
        tabu = selected[(d, k, "Tabu search")]
        scip = selected[(d, k, "SCIP")]
        smvc_time = smvc["mean_time_s"]
        beam_width = beam["beam_width"]
        if beam_width is None:
            raise ValueError(f"Missing Beam DP width for d={d}, k={k}, n=500")
        errors = [
            smvc["mean_relative_error_pct"],
            beam["mean_relative_error_pct"],
            tabu["mean_relative_error_pct"],
            scip["mean_relative_error_pct"],
        ]
        best_error = min(errors)
        separator = "        \\midrule\n" if previous_d is not None and d != previous_d else ""
        latex_rows.append(
            separator
            + f"        {d} & {k} & "
            f"{latex_error(smvc['mean_relative_error_pct'], smvc['mean_relative_error_pct'] == best_error)} & "
            f"{latex_error(beam['mean_relative_error_pct'], beam['mean_relative_error_pct'] == best_error)} & "
            f"{beam_width / d**k:.3f} & "
            f"{beam['mean_time_s'] / smvc_time:.3f} & "
            f"{latex_error(tabu['mean_relative_error_pct'], tabu['mean_relative_error_pct'] == best_error)} & "
            f"{tabu['mean_time_s'] / smvc_time:.3f} & "
            f"{latex_error(scip['mean_relative_error_pct'], scip['mean_relative_error_pct'] == best_error)} & "
            f"{scip['mean_time_s'] / smvc_time:.3f} \\\\\n"
        )
        previous_d = d

    latex_path = OUTPUT_DIR / "experiment_2_comparison.tex"
    latex = (
        "\\begin{table*}[t]\n"
        "    \\centering\n"
        "    \\caption{\n"
        "    Comparison of SMVC, Beam DP, tabu search, and SCIP on random\n"
        "    $k$-neighbor QUDO instances with $n=500$. Each configuration\n"
        "    contains 50 instances. Errors are mean relative errors, with the\n"
        "    lowest value in each row highlighted in bold. Runtime ratios are\n"
        "    computed relative to SMVC.\n"
        "    }\n"
        "    \\label{tab:n500-method-comparison}\n"
        "    \\small\n"
        "    \\setlength{\\tabcolsep}{4pt}\n"
        "    \\renewcommand{\\arraystretch}{1.15}\n\n"
        "    \\begin{tabular*}{\\textwidth}{\n"
        "        @{\\extracolsep{\\fill}}ccrrrrrrrr@{}\n"
        "    }\n"
        "        \\toprule\n"
        "        & &\n"
        "        \\multicolumn{1}{c}{SMVC} &\n"
        "        \\multicolumn{3}{c}{Beam DP} &\n"
        "        \\multicolumn{2}{c}{Tabu search} &\n"
        "        \\multicolumn{2}{c}{SCIP} \\\\\n"
        "        \\cmidrule(lr){3-3}\n"
        "        \\cmidrule(lr){4-6}\n"
        "        \\cmidrule(lr){7-8}\n"
        "        \\cmidrule(l){9-10}\n\n"
        "        $d$ & $k$ &\n"
        "        Error (\\%) & Error (\\%) & $b/d^k$ & Beam/SMVC &\n"
        "        Error (\\%) & Tabu/SMVC & Error (\\%) & SCIP/SMVC \\\\\n"
        "        \\midrule\n"
        + "".join(latex_rows)
        + "        \\bottomrule\n"
        "    \\end{tabular*}\n"
        "\\end{table*}\n"
    )
    latex_path.write_text(latex, encoding="utf-8")
    return csv_path, markdown_path, latex_path


def plot_method_label(method: str, points: Sequence[SummaryRow]) -> str:
    """Include calibrated beam widths in Beam DP plot legends."""
    if method != "Beam DP":
        return method
    beam_by_n = [
        (point["n_variables"], point["beam_width"])
        for point in points
        if point["beam_width"] is not None
    ]
    if not beam_by_n:
        return method
    unique_widths = {beam_width for _, beam_width in beam_by_n}
    if len(unique_widths) == 1:
        return f"Beam DP (beam={beam_by_n[0][1]})"
    mapping = ", ".join(f"n={n}:{beam}" for n, beam in beam_by_n)
    return f"Beam DP (beam {mapping})"


def make_plots(rows: Sequence[SummaryRow]) -> tuple[Path, Path]:
    series: defaultdict[
        tuple[int, int, str, str], list[SummaryRow]
    ] = defaultdict(list)
    for row in rows:
        series[(row["d"], row["k"], row["instance_type"], row["method"])].append(row)
    for points in series.values():
        points.sort(key=lambda row: row["n_variables"])

    plt.style.use("seaborn-v0_8-whitegrid")
    colours: dict[str, str] = {
        "Matrix method": "#1f77b4",
        "Beam DP": "#d62728",
        "Simulated annealing": "#f4a261",
        "Tabu search": "#2a9d8f",
        "SCIP": "#9467bd",
    }
    styles: dict[str, str | tuple[int, tuple[int, ...]]] = {
        "Matrix method": "-",
        "Beam DP": "--",
        "Simulated annealing": ":",
        "Tabu search": "-.",
        "SCIP": (0, (3, 1, 1, 1)),
    }
    markers: dict[str, str] = {
        "random": "o",
        "fixed": "s",
        "unspecified": "^",
    }
    figure, raw_axes = plt.subplots(
        1, 2, figsize=(13, 4.6), constrained_layout=True
    )
    axes = cast(tuple[Axes, Axes], raw_axes)
    for (d, k, instance_type, method), points in series.items():
        x = [point["n_variables"] for point in points]
        method_label = plot_method_label(method, points)
        label = f"d={d}, k={k}, {instance_type} - {method_label}"
        marker = markers.get(instance_type, "D")
        axes[0].plot(
            x,
            [point["optimal_found_pct"] for point in points],
            marker=marker,
            color=colours[method],
            linestyle=styles[method],
            label=label,
        )
        axes[1].plot(
            x,
            [point["mean_relative_error_pct"] for point in points],
            marker=marker,
            color=colours[method],
            linestyle=styles[method],
            label=label,
        )
    axes[0].set(title="Optimum recovery", xlabel="Number of variables ($n$)",
                ylabel="Optimal found (%)", ylim=(-2, 102))
    axes[1].set(title="Solution error", xlabel="Number of variables ($n$)",
                ylabel="Mean relative error (%)")
    axes[1].set_yscale("symlog", linthresh=1e-12)
    axes[1].axhline(0, color="black", linewidth=0.7)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=8,
                  bbox_to_anchor=(0.5, -0.16))
    quality_path = OUTPUT_DIR / "correctness_by_d_k_n.png"
    figure.savefig(quality_path, dpi=300, bbox_inches="tight")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    for (d, k, instance_type, method), points in series.items():
        method_label = plot_method_label(method, points)
        axis.plot([point["n_variables"] for point in points],
                  [point["mean_time_s"] for point in points],
                  marker=markers.get(instance_type, "D"), color=colours[method],
                  linestyle=styles[method],
                  label=f"d={d}, k={k}, {instance_type} - {method_label}")
    axis.set(title="Mean execution time", xlabel="Number of variables ($n$)",
             ylabel="Mean time per instance (s)", yscale="log")
    axis.legend(fontsize=8, ncol=2)
    time_path = OUTPUT_DIR / "runtime_by_d_k_n.png"
    figure.savefig(time_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return quality_path, time_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    outputs = (*write_tables(rows), *make_plots(rows))
    print("Processed Experiment 2 results:")
    for output in outputs:
        print(f"  - {output}")


if __name__ == "__main__":
    main()
