"""Experiment 5: execution time as a function of n and k."""

import json
import sys
from pathlib import Path

from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import solver_dynamic_programming
from qudo_solver.solvers.smvc.smvc import solver_smvc
from qudo_solver.solvers.stc.stc_solver import solver_stc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qudo_solver.data_generator.qudo_problem_generator import (
    qudo_problem_generation,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_VARIABLES = list(range(200, 1001, 200))
K_VALUES = list(range(2, 11))
DITS = 2
N_RANDOM_INSTANCES = 3


def run_configuration(n_variables: int, n_neighbors: int) -> list[dict]:
    results = []
    instances = qudo_problem_generation(
        n_variables=n_variables,
        n_neighbors=n_neighbors,
        n_random_instances=N_RANDOM_INSTANCES,
        n_fixed_instances=0,
    )

    for index, instance in enumerate(instances):
        q_matrix = instance["q_matrix"]
        q_row = instance["q_row"]
        tau = None

        matrix_solution = solver_smvc(
            q_matrix, q_row, DITS, n_neighbors, tau
        )
        tensor_solution = solver_stc(
            q_matrix, q_row, tau, DITS, n_neighbors
        )
        dynamic_solution = solver_dynamic_programming(
            q_matrix,
            q_row,
            DITS,
            n_neighbors,
        )

        results.append(
            {
                "n_variables": n_variables,
                "dits": DITS,
                "n_neighbors": n_neighbors,
                "instance_index": index,
                "instance_type": instance["instance_type"],
                "seed": instance["seed"],
                "matrix_method": {
                    "execution_time": matrix_solution.execution_time,
                    "cost": matrix_solution.cost,
                },
                "tensor_method": {
                    "execution_time": tensor_solution.execution_time,
                    "cost": tensor_solution.cost,
                },
                "dynamic_programming": {
                    "execution_time": dynamic_solution.execution_time,
                    "cost": dynamic_solution.cost,
                },
            }
        )

        print(
            f"n={n_variables}, k={n_neighbors}, instance={index}: "
            f"matrix={matrix_solution.execution_time:.4f}s, "
            f"tensor={tensor_solution.execution_time:.4f}s, "
            f"dynamic={dynamic_solution.execution_time:.4f}s"
        )

    return results


def main() -> None:
    results = []

    for n_neighbors in K_VALUES:
        for n_variables in N_VARIABLES:
            results.extend(run_configuration(n_variables, n_neighbors))

    payload = {
        "summary": {
            "n_variables": N_VARIABLES,
            "k_values": K_VALUES,
            "dits": DITS,
            "n_random_instances": N_RANDOM_INSTANCES,
        },
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_5.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
