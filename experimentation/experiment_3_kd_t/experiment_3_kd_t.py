"""Experiment 3: execution time as a function of k and d."""

import json
import sys
from pathlib import Path

from qudo_solver.solvers.smvc.smvc import solver_smvc
from qudo_solver.solvers.stc.stc_solver import solver_stc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qudo_solver.data_generator.qudo_problem_generator import (
    qudo_problem_generation,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
N_VARIABLES = 100
K_VALUES = list(range(1, 15))
DITS_VALUES = list(range(2, 30))
FIXED_DITS = 2
FIXED_K = 2
N_RANDOM_INSTANCES = 3


def run_configuration(series: str, value: int, dits: int, n_neighbors: int) -> list[dict]:
    results = []
    instances = qudo_problem_generation(
        n_variables=N_VARIABLES,
        n_neighbors=n_neighbors,
        n_random_instances=N_RANDOM_INSTANCES,
        n_fixed_instances=0,
    )

    for index, instance in enumerate(instances):
        q_matrix = instance["q_matrix"]
        q_row = instance["q_row"]
        tensor_solution = solver_stc(
            Q_matrix=q_matrix,
            Q_row=q_row,
            dits=dits,
            n_neighbors=n_neighbors,
            tau=None,
        )
        matrix_solution = solver_smvc(
            Q_list=q_matrix,
            Q_row=q_row,
            dits=dits,
            n_neighbors=n_neighbors,
        )

        results.append(
            {
                "series": series,
                "value": value,
                "n_variables": N_VARIABLES,
                "dits": dits,
                "n_neighbors": n_neighbors,
                "instance_index": index,
                "instance_type": instance["instance_type"],
                "seed": instance["seed"],
                "tensor_method": {
                    "execution_time": tensor_solution.execution_time,
                    "cost": tensor_solution.cost,
                },
                "matrix_method": {
                    "execution_time": matrix_solution.execution_time,
                    "cost": matrix_solution.cost,
                },

            }
        )

        print(
            f"{series}={value}, instancia={index}: "
            f"tensor={tensor_solution.execution_time:.4f}s, "
            f"matriz={matrix_solution.execution_time:.4f}s, "
        )

    return results


def main() -> None:
    results = []

    for n_neighbors in K_VALUES:
        results.extend(
            run_configuration("k", n_neighbors, FIXED_DITS, n_neighbors)
        )

    for dits in DITS_VALUES:
        results.extend(run_configuration("d", dits, dits, FIXED_K))

    payload = {
        "summary": {
            "n_variables": N_VARIABLES,
            "k_values": K_VALUES,
            "dits_values": DITS_VALUES,
            "fixed_dits": FIXED_DITS,
            "fixed_k": FIXED_K,
            "n_random_instances": N_RANDOM_INSTANCES,
        },
        "results": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_3_kd_t.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
    main()
