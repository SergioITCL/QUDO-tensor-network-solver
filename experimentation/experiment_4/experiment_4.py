"""Experiment 4: matrix and tensor execution time as a function of n and d."""

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qudo_solver.auxiliar_functions import estimate_tau_max
from qudo_solver.data_generator.qudo_problem_generator import (
    normalize_list_of_lists,
    qudo_problem_generation,
)
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver3 import (
    solver_dynamic_programming3,
)
from qudo_solver.solvers.matrix_method.matrix_method_solver import solver_matrix_method
from qudo_solver.solvers.tensorkrowch_tn.tensorkrowch_solver import (
    solver_tensorkrowch,
)


RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_VARIABLES = list(range(200, 1001, 200))
DITS_VALUES = list(range(2, 9))
N_NEIGHBORS = 2
N_RANDOM_INSTANCES = 3


def run_configuration(n_variables: int, dits: int) -> list[dict]:
    results = []
    instances = qudo_problem_generation(
        n_variables=n_variables,
        n_neighbors=N_NEIGHBORS,
        n_random_instances=N_RANDOM_INSTANCES,
        n_fixed_instances=0,
    )

    for index, instance in enumerate(instances):
        q_matrix = instance["q_matrix"]
        q_row = instance["q_row"]
        tau = None

        matrix_solution = solver_matrix_method(
            q_matrix,
            q_row,
            dits,
            N_NEIGHBORS,
            tau,
        )
        tensor_solution = solver_tensorkrowch(
            q_matrix,
            q_row,
            tau,
            dits,
            N_NEIGHBORS,
        )
        dynamic_solution = solver_dynamic_programming3(
            q_matrix,
            q_row,
            dits,
            N_NEIGHBORS,
        )

        results.append(
            {
                "n_variables": n_variables,
                "dits": dits,
                "n_neighbors": N_NEIGHBORS,
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
            f"n={n_variables}, d={dits}, instancia={index}: "
            f"matriz={matrix_solution.execution_time:.4f}s, "
            f"tensor={tensor_solution.execution_time:.4f}s, "
            f"dinamica={dynamic_solution.execution_time:.4f}s"
        )

    return results


def main() -> None:
    results = []

    for dits in DITS_VALUES:
        for n_variables in N_VARIABLES:
            results.extend(run_configuration(n_variables, dits))

    payload = {
        "summary": {
            "n_variables": N_VARIABLES,
            "dits_values": DITS_VALUES,
            "n_neighbors": N_NEIGHBORS,
            "n_random_instances": N_RANDOM_INSTANCES,
        },
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_4.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
