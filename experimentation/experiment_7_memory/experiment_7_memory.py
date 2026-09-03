"""Experiment 7: peak memory as a function of n, k, and d."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimentation.experiment_config import experiment_path, load_experiment
from qudo_solver.data_generator.qudo_problem_generator import (
    qudo_problem_generation,
)

CONFIG = load_experiment("experiment_7")


def current_rss_mib(pid: int) -> float | None:
    try:
        resident_pages = int(Path(f"/proc/{pid}/statm").read_text().split()[1])
    except FileNotFoundError:
        return None
    return resident_pages * os.sysconf("SC_PAGE_SIZE") / 1024**2


def get_solver(method: str, dits: int, n_neighbors: int, seed: int):
    if method == "exact_dp":
        from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (
            solver_dynamic_programming,
        )

        return lambda q_matrix, q_row: solver_dynamic_programming(
            q_matrix, q_row, dits, n_neighbors
        )
    if method == "smvc":
        from qudo_solver.solvers.smvc.smvc import (
            solver_smvc,
        )

        return lambda q_matrix, q_row: solver_smvc(
            q_matrix, q_row, dits, n_neighbors
        )
    if method == "stc":
        from qudo_solver.solvers.stc.stc_solver import (
            solver_stc,
        )

        return lambda q_matrix, q_row: solver_stc(
            q_matrix, q_row, None, dits, n_neighbors
        )
    if method == "beam_dp":
        from qudo_solver.solvers.dynamic_programming.beam_dynamic_programming_solver import (
            solver_beam_dynamic_programming,
        )

        return lambda q_matrix, q_row: solver_beam_dynamic_programming(
            q_matrix, q_row, dits, n_neighbors, beam_width=CONFIG["beam_width"]
        )
    if method == "tabu":
        from qudo_solver.solvers.tabu_search import solver_tabu_search

        return lambda q_matrix, q_row: solver_tabu_search(
            q_matrix,
            q_row,
            dits,
            n_neighbors,
            time_limit=CONFIG["tabu_time_limit"],
            max_iterations=CONFIG["tabu_iterations"],
            seed=seed,
        )
    raise ValueError(f"Unknown method: {method}")


def run_worker(method: str, n: int, k: int, d: int, seed: int) -> None:
    solver = get_solver(method, d, k, seed)
    instance = qudo_problem_generation(
        n_variables=n,
        n_neighbors=k,
        n_random_instances=1,
        n_fixed_instances=0,
        random_seeds=[seed],
    )[0]
    q_matrix = instance["q_matrix"]
    q_row = instance["q_row"]

    baseline = current_rss_mib(os.getpid())
    print(json.dumps({"baseline_rss_mib": baseline}), flush=True)
    sys.stdin.readline()
    solution = solver(q_matrix, q_row)
    print(
        json.dumps(
            {
                "execution_time": solution.execution_time,
                "cost": solution.cost,
            }
        ),
        flush=True,
    )


def measure(method: str, n: int, k: int, d: int, seed: int) -> dict:
    command = [sys.executable, __file__, "worker", method, str(n), str(k), str(d), str(seed)]
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None

    baseline = json.loads(process.stdout.readline())["baseline_rss_mib"]
    peak = baseline
    process.stdin.write("start\n")
    process.stdin.flush()

    while process.poll() is None:
        memory = current_rss_mib(process.pid)
        if memory is not None:
            peak = max(peak, memory)
        time.sleep(CONFIG["sample_interval"])

    output = process.stdout.read().strip().splitlines()
    error = process.stderr.read()
    if process.returncode != 0:
        raise RuntimeError(error)

    result = json.loads(output[-1])
    return {
        "baseline_rss_mib": baseline,
        "peak_rss_mib": peak,
        "incremental_peak_rss_mib": peak - baseline,
        **result,
    }


def configurations():
    for n in CONFIG["n_values"]:
        yield "n", n, n, CONFIG["fixed_k"], CONFIG["fixed_d"]
    for k in CONFIG["k_values"]:
        yield "k", k, CONFIG["fixed_n"], k, CONFIG["fixed_d"]
    for d in CONFIG["d_values"]:
        yield "d", d, CONFIG["fixed_n"], CONFIG["fixed_k"], d


def main() -> None:
    results = []
    for series, value, n, k, d in configurations():
        for seed in CONFIG["seeds"]:
            for method in CONFIG["methods"]:
                measurement = measure(method, n, k, d, seed)
                results.append(
                    {
                        "series": series,
                        "value": value,
                        "n_variables": n,
                        "n_neighbors": k,
                        "dits": d,
                        "seed": seed,
                        "method": method,
                        **measurement,
                    }
                )
                print(
                    f"{series}={value}, seed={seed}, {method}: "
                    f"peak={measurement['peak_rss_mib']:.3f} MiB, "
                    f"increment={measurement['incremental_peak_rss_mib']:.3f} MiB"
                )

    payload = {
        "parameters": {
            "n_values": CONFIG["n_values"],
            "k_values": CONFIG["k_values"],
            "d_values": CONFIG["d_values"],
            "fixed_n": CONFIG["fixed_n"],
            "fixed_k": CONFIG["fixed_k"],
            "fixed_d": CONFIG["fixed_d"],
            "n_random_instances": len(CONFIG["seeds"]),
            "seeds": CONFIG["seeds"],
            "beam_width": CONFIG["beam_width"],
            "tabu_iterations": CONFIG["tabu_iterations"],
            "sample_interval_s": CONFIG["sample_interval"],
            "memory_metric": "Linux process RSS in MiB",
        },
        "results": results,
    }
    results_dir = experiment_path(CONFIG["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / CONFIG["output_file"]
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "worker":
        run_worker(sys.argv[2], *(int(value) for value in sys.argv[3:]))
    else:
        main()
