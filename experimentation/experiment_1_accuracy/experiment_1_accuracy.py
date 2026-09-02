"""Experiment 1: SMVC accuracy relative to exact dynamic programming."""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qudo_solver.data_generator.qudo_problem_generator import qudo_problem_generation
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver3 import (
    solver_dynamic_programming3,
)
from qudo_solver.solvers.matrix_method.matrix_method_solver import solver_matrix_method

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_VARIABLES = (500, 1000)
CONFIGURATIONS = ((2, 2), (2, 4), (4, 2), (4, 4))
N_RANDOM_INSTANCES = 50


def run_configuration(n_variables: int, dits: int, n_neighbors: int) -> list[dict]:
    results = []
    instances = qudo_problem_generation(
        n_variables=n_variables,
        n_neighbors=n_neighbors,
        n_random_instances=N_RANDOM_INSTANCES,
        n_fixed_instances=0,
    )

    for instance in instances:
        q_matrix = instance["q_matrix"]
        q_row = instance["q_row"]
        exact_dp = solver_dynamic_programming3(q_matrix, q_row, dits, n_neighbors)
        smvc = solver_matrix_method(q_matrix, q_row, dits, n_neighbors)

        results.append(
            {
                "n_variables": n_variables,
                "dits": dits,
                "n_neighbors": n_neighbors,
                "seed": instance["seed"],
                "exact_dp": {
                    "cost": exact_dp.cost,
                    "execution_time": exact_dp.execution_time,
                },
                "smvc": {
                    "cost": smvc.cost,
                    "execution_time": smvc.execution_time,
                },
            }
        )
        print(f"n={n_variables}, d={dits}, k={n_neighbors}, seed={instance['seed']}")

    return results


def main() -> None:
    results = []
    for dits, n_neighbors in CONFIGURATIONS:
        for n_variables in N_VARIABLES:
            results.extend(run_configuration(n_variables, dits, n_neighbors))

    payload = {
        "parameters": {
            "n_variables": N_VARIABLES,
            "configurations": CONFIGURATIONS,
            "n_random_instances": N_RANDOM_INSTANCES,
            "linear_coefficients": 0,
        },
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_1_accuracy.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
