"""Experiment 4: matrix and tensor execution time as a function of n and d."""

import json
import sys
from pathlib import Path

from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (
    solver_dynamic_programming,
)
from qudo_solver.solvers.dynamic_programming.vectorized_dynamic_programin import (
    solver_dynamic_programming_vectorized,
)
from qudo_solver.solvers.smvc.smvc import solver_smvc
from qudo_solver.solvers.stc.stc_solver import solver_stc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimentation.experiment_config import experiment_path, load_experiment
from qudo_solver.data_generator.qudo_problem_generator import (
    qudo_problem_generation,
)

CONFIG = load_experiment("experiment_4")


def run_configuration(n_variables: int, dits: int) -> list[dict]:
    results = []
    instances = qudo_problem_generation(
        n_variables=n_variables,
        n_neighbors=CONFIG["n_neighbors"],
        n_random_instances=len(CONFIG["seeds"]),
        random_seeds=CONFIG["seeds"],
        n_fixed_instances=0,
    )

    for index, instance in enumerate(instances):
        q_matrix = instance["q_matrix"]
        q_row = instance["q_row"]
        tau = None

        matrix_solution = solver_smvc(
            q_matrix,
            q_row,
            dits,
            CONFIG["n_neighbors"],
            tau,
        )
        tensor_solution = solver_stc(
            q_matrix,
            q_row,
            tau,
            dits,
            CONFIG["n_neighbors"],
        )
        dynamic_solution = solver_dynamic_programming_vectorized(
            q_matrix,
            q_row,
            dits,
            CONFIG["n_neighbors"],
        )

        results.append(
            {
                "n_variables": n_variables,
                "dits": dits,
                "n_neighbors": CONFIG["n_neighbors"],
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
            f"n={n_variables}, d={dits}, instance={index}: "
            f"matrix={matrix_solution.execution_time:.4f}s, "
            f"tensor={tensor_solution.execution_time:.4f}s, "
            f"dynamic={dynamic_solution.execution_time:.4f}s, "
            f"cost_tensor={tensor_solution.cost:.4f}s, "
            f"cost_dynamic={dynamic_solution.cost:.4f}s, "
        )

    return results


def main() -> None:
    results = []

    for dits in CONFIG["dits_values"]:
        for n_variables in CONFIG["n_variables"]:
            results.extend(run_configuration(n_variables, dits))

    payload = {
        "summary": {
            "n_variables": CONFIG["n_variables"],
            "dits_values": CONFIG["dits_values"],
            "n_neighbors": CONFIG["n_neighbors"],
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
