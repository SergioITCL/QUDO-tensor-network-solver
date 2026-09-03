"""Experiment 1: SMVC accuracy relative to exact dynamic programming."""

import json
import sys
from pathlib import Path

from qudo_solver.solvers.smvc.smvc import solver_smvc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimentation.experiment_config import experiment_path, load_experiment
from qudo_solver.data_generator.qudo_problem_generator import qudo_problem_generation
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (
    solver_dynamic_programming,
)

CONFIG = load_experiment("experiment_1")
print(CONFIG)

def run_configuration(n_variables: int, dits: int, n_neighbors: int) -> list[dict]:
    results = []
    instances = qudo_problem_generation(
        n_variables=n_variables,
        n_neighbors=n_neighbors,
        n_random_instances=len(CONFIG["seeds"]),
        random_seeds=CONFIG["seeds"],
        n_fixed_instances=0,
    )

    for instance in instances:
        q_matrix = instance["q_matrix"]
        q_row = instance["q_row"]
        exact_dp = solver_dynamic_programming(q_matrix, q_row, dits, n_neighbors)
        smvc = solver_smvc(q_matrix, q_row, dits, n_neighbors)

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
    for dits, n_neighbors in CONFIG["configurations"]:
        for n_variables in CONFIG["n_variables"]:
            results.extend(run_configuration(n_variables, dits, n_neighbors))

    payload = {
        "parameters": {
            "n_variables": CONFIG["n_variables"],
            "configurations": CONFIG["configurations"],
            "n_random_instances": len(CONFIG["seeds"]),
            "seeds": CONFIG["seeds"],
            "linear_coefficients": "random",
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
