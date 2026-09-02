"""Experiment 2: scaling of the exact DP solver and the matrix method."""

import json
import math
import sys
from pathlib import Path

from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver3 import solver_dynamic_programming3

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qudo_solver.data_generator.qudo_problem_generator import qudo_problem_generation
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver2 import (
    solver_dynamic_programming2,
)
from qudo_solver.solvers.matrix_method.matrix_method_solver import solver_matrix_method

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_VARIABLES = [100,200,400,600,800,1000]
DITS_VALUES = [2,4,6,8]
K_VALUES = [2, 4,6,8]
N_RANDOM_INSTANCES = 5
N_FIXED_INSTANCES = 5


def build_summary(
    results: list[dict],
    dits: int,
    n_neighbors: int,
) -> dict:
    """Create compact per-size results to place before the raw records."""
    by_n_variables = []

    for n_variables in N_VARIABLES:
        records = [record for record in results if record["n_variables"] == n_variables]
        exact_times = [record["exact_dynamic_programming"]["execution_time"] for record in records]
        matrix_times = [record["matrix_method"]["execution_time"] for record in records]
        matching_costs = sum(
            math.isclose(
                record["exact_dynamic_programming"]["cost"],
                record["matrix_method"]["cost"],
                abs_tol=1e-9,
                rel_tol=1e-8,
            )
            for record in records
        )

        by_n_variables.append(
            {
                "n_variables": n_variables,
                "n_instances": len(records),
                "mean_exact_time": sum(exact_times) / len(exact_times),
                "mean_matrix_time": sum(matrix_times) / len(matrix_times),
                "matrix_matches_exact": matching_costs,
                "matrix_match_rate": matching_costs / len(records),
            }
        )

    return {
        "dits": dits,
        "n_neighbors": n_neighbors,
        "n_random_instances": N_RANDOM_INSTANCES,
        "n_fixed_instances": N_FIXED_INSTANCES,
        "total_instances": len(results),
        "by_n_variables": by_n_variables,
    }


def run_experiment(dits: int, n_neighbors: int) -> None:
    results = []

    for n_variables in N_VARIABLES:
        print(f"\nPrueba con dits={dits}, k={n_neighbors}, n={n_variables}")

        instances = qudo_problem_generation(
            n_variables=n_variables,
            n_neighbors=n_neighbors,
            n_random_instances=N_RANDOM_INSTANCES,
            n_fixed_instances=N_FIXED_INSTANCES,
        )

        for index, instance in enumerate(instances):
            q_matrix = instance["q_matrix"]
            q_row = instance["q_row"]

            exact_solution = solver_dynamic_programming3(
                q_matrix=q_matrix,
                q_row=q_row,
                dits=dits,
                n_neighbors=n_neighbors,
            )
            matrix_solution = solver_matrix_method(
                Q_list=q_matrix,
                Q_row=q_row,
                dits=dits,
                n_neighbors=n_neighbors,
            )

            results.append(
                {
                    "n_variables": n_variables,
                    "dits": dits,
                    "n_neighbors": n_neighbors,
                    "instance_index": index,
                    "instance_type": instance["instance_type"],
                    "seed": instance["seed"],
                    "exact_dynamic_programming": {
                        "execution_time": exact_solution.execution_time,
                        "cost": exact_solution.cost,
                    },
                    "matrix_method": {
                        "execution_time": matrix_solution.execution_time,
                        "cost": matrix_solution.cost,
                    },
                }
            )

            print(
                f"  instancia={index} ({instance['instance_type']}, seed={instance['seed']}): "
                f"exacto={exact_solution.execution_time:.6f}s, "
                f"matriz={matrix_solution.execution_time:.6f}s, "
                f"costes=({exact_solution.cost:.6f}, {matrix_solution.cost:.6f})"
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / f"experiment_2_d{dits}_k{n_neighbors}.json"
    payload = {
        "summary": build_summary(results, dits, n_neighbors),
        "results": results,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nResultados guardados en: {output_path}")


def main() -> None:
    for dits in DITS_VALUES:
        for n_neighbors in K_VALUES:
            run_experiment(dits=dits, n_neighbors=n_neighbors)


if __name__ == "__main__":
    main()
