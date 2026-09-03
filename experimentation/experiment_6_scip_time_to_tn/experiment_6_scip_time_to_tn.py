"""Experiment 6: SCIP wall time needed to reach Matrix TN solution quality."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import fmean, median

from qudo_solver.solvers.smvc.smvc import solver_smvc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qudo_solver.auxiliar_functions import estimate_tau_max, qudo_value
from qudo_solver.data_generator.qudo_problem_generator import (
    qudo_problem_generation,
)
from qudo_solver.solvers.scip import solver_scip_time_to_target
from experimentation.experiment_config import experiment_path, load_experiment

CONFIG = load_experiment("experiment_6")


def _distribution(values: list[float]) -> dict[str, float | None]:
    """Return descriptive statistics without inventing values for no data."""
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": fmean(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_results(results: list[dict]) -> dict:
    """Summarize successes separately from right-censored observations."""
    total = len(results)
    successful = [result for result in results if result["target_reached"]]
    matrix_times = [float(result["matrix_time"]) for result in results]
    time_to_target = [
        float(result["time_to_target"]) for result in successful
    ]
    time_ratios = [float(result["time_ratio"]) for result in successful]
    reached = len(successful)

    return {
        "n_instances": total,
        "n_target_reached": reached,
        "n_target_not_reached": total - reached,
        "target_success_rate": reached / total if total else 0.0,
        "matrix_time": _distribution(matrix_times),
        "scip_time_to_target": _distribution(time_to_target),
        "time_ratio": _distribution(time_ratios),
    }


def _write_checkpoint(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _format_stat(value: float | None, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.3f}{suffix}"


def print_summary(n: int, dits: int, n_neighbors: int, summary: dict) -> None:
    reached = summary["n_target_reached"]
    total = summary["n_instances"]
    percentage = 100.0 * summary["target_success_rate"]
    matrix_stats = summary["matrix_time"]
    target_stats = summary["scip_time_to_target"]
    ratio_stats = summary["time_ratio"]

    print("=" * 60)
    print(f"n={n}, d={dits}, k={n_neighbors}")
    print(f"Instances: {total}")
    print(f"SCIP reached TN target: {reached}/{total} ({percentage:.1f}%)")
    print("\nMatrix TN:")
    print(f"  median time: {_format_stat(matrix_stats['median'], ' s')}")
    print(f"  mean time:   {_format_stat(matrix_stats['mean'], ' s')}")
    print("\nSCIP time-to-TN (reached instances only):")
    print(f"  median: {_format_stat(target_stats['median'], ' s')}")
    print(f"  mean:   {_format_stat(target_stats['mean'], ' s')}")
    print("\nSCIP/TN time ratio (reached instances only):")
    print(f"  median: {_format_stat(ratio_stats['median'], 'x')}")
    print(f"  mean:   {_format_stat(ratio_stats['mean'], 'x')}")
    print(
        f"\nNot reached within {CONFIG['scip_max_time']:g} s: "
        f"{summary['n_target_not_reached']}/{total}"
    )
    print("=" * 60)


def run_configuration(n: int, dits: int, n_neighbors: int) -> Path:
    """Run and checkpoint one reproducible ``(n, d, k)`` configuration."""
    tau = estimate_tau_max(
        n_variables=n,
        dits=dits,
        n_neighbors=n_neighbors,
    )
    instances = qudo_problem_generation(
        n_variables=n,
        n_neighbors=n_neighbors,
        n_random_instances=len(CONFIG["seeds"]),
        n_fixed_instances=0,
        random_seeds=CONFIG["seeds"],
    )
    output_path = experiment_path(CONFIG["results_dir"]) / (
        f"experiment_6_params_n{n}_d{dits}_k{n_neighbors}.json"
    )
    experiment_data = {
        "experiment": "SCIP time-to-Matrix-TN target",
        "parameters": {
            "n_variables": n,
            "dits": dits,
            "n_neighbors": n_neighbors,
            "n_random_instances": len(CONFIG["seeds"]),
            "seeds": CONFIG["seeds"],
            "scip_max_time": CONFIG["scip_max_time"],
            "target_tolerance": CONFIG["target_tolerance"],
            "require_nonzero": CONFIG["require_nonzero"],
            "tau_policy": "estimate_tau_max",
            "tau": tau,
            "censoring": (
                "time_to_target is null when the target is not reached; "
                "ratio_lower_bound is not treated as an observed ratio"
            ),
        },
        "summary": summarize_results([]),
        "results": [],
    }

    for instance_index, instance in enumerate(instances):
        q_matrix = instance["q_matrix"]
        q_row = instance["q_row"]

        matrix_result = solver_smvc(
            Q_list=q_matrix,
            Q_row=q_row,
            dits=dits,
            n_neighbors=n_neighbors,
            tau=tau,
        )
        matrix_solution = [int(value) for value in matrix_result.solution_list]
        matrix_cost = qudo_value(matrix_solution, q_matrix, q_row)

        target_result = solver_scip_time_to_target(
            q_matrix=q_matrix,
            q_row=q_row,
            dits=dits,
            n_neighbors=n_neighbors,
            target_cost=matrix_cost,
            max_time=CONFIG["scip_max_time"],
            require_nonzero=CONFIG["require_nonzero"],
            seed=instance["seed"],
            target_tolerance=CONFIG["target_tolerance"],
        )

        if target_result.solution is None:
            final_solution = None
            final_cost = None
        else:
            final_solution = [
                int(value) for value in target_result.solution.solution_list
            ]
            final_cost = qudo_value(final_solution, q_matrix, q_row)

        reached = target_result.reached
        time_to_target = target_result.time_to_target if reached else None
        time_ratio = (
            time_to_target / matrix_result.execution_time
            if time_to_target is not None
            else None
        )
        ratio_lower_bound = (
            None
            if reached
            else CONFIG["scip_max_time"] / matrix_result.execution_time
        )

        experiment_data["results"].append(
            {
                "instance_index": instance_index,
                "seed": instance["seed"],
                "n": n,
                "dits": dits,
                "k": n_neighbors,
                "matrix_cost": matrix_cost,
                "matrix_time": matrix_result.execution_time,
                "tau": tau,
                "matrix_solution": matrix_solution,
                "target_cost": matrix_cost,
                "target_reached": reached,
                "target_not_reached": not reached,
                "time_to_target": time_to_target,
                "final_cost": final_cost,
                "final_solution": final_solution,
                "target_difference": (
                    final_cost - matrix_cost if final_cost is not None else None
                ),
                "time_ratio": time_ratio,
                "ratio_lower_bound": ratio_lower_bound,
                "scip_total_execution_time": (
                    target_result.total_execution_time
                ),
                "status": target_result.metadata.status,
                "solving_time": target_result.metadata.solving_time,
                "nodes": target_result.metadata.nodes,
                "incumbent_objective": target_result.metadata.objective,
                "best_bound": target_result.metadata.best_bound,
                "gap": target_result.metadata.gap,
                "incumbent_history": target_result.incumbent_history,
            }
        )
        experiment_data["summary"] = summarize_results(
            experiment_data["results"]
        )
        _write_checkpoint(output_path, experiment_data)
        print(
            f"[{instance_index + 1:02d}/{len(instances)}] "
            f"n={n}, d={dits}, k={n_neighbors}, seed={instance['seed']}: "
            f"target_reached={reached}, time_to_target={time_to_target}"
        )

    print_summary(n, dits, n_neighbors, experiment_data["summary"])
    return output_path


def main() -> None:
    for n in CONFIG["n_variables"]:
        for dits, n_neighbors in CONFIG["configurations"]:
            run_configuration(n, dits, n_neighbors)


if __name__ == "__main__":
    main()
