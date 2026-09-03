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
from experimentation.experiment_config import experiment_path, load_experiment

CONFIG = load_experiment("experiment_3")


def run_configuration(series: str, value: int, dits: int, n_neighbors: int) -> list[dict]:
    results = []
    instances = qudo_problem_generation(
        n_variables=CONFIG["n_variables"],
        n_neighbors=n_neighbors,
        n_random_instances=len(CONFIG["seeds"]),
        random_seeds=CONFIG["seeds"],
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
                "n_variables": CONFIG["n_variables"],
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
            f"{series}={value}, instance={index}: "
            f"tensor={tensor_solution.execution_time:.4f}s, "
            f"matrix={matrix_solution.execution_time:.4f}s, "
        )

    return results


def main() -> None:
    results = []

    for n_neighbors in CONFIG["k_values"]:
        results.extend(
            run_configuration("k", n_neighbors, CONFIG["fixed_dits"], n_neighbors)
        )

    for dits in CONFIG["dits_values"]:
        results.extend(run_configuration("d", dits, dits, CONFIG["fixed_k"]))

    payload = {
        "summary": {
            "n_variables": CONFIG["n_variables"],
            "k_values": CONFIG["k_values"],
            "dits_values": CONFIG["dits_values"],
            "fixed_dits": CONFIG["fixed_dits"],
            "fixed_k": CONFIG["fixed_k"],
            "n_random_instances": len(CONFIG["seeds"]),
            "seeds": CONFIG["seeds"],
        },
        "results": results,
    }

    results_dir = experiment_path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / CONFIG["output_file"]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
