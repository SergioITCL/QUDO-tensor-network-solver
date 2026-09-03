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

from qudo_solver.data_generator.qudo_problem_generator import generate_k_random_qudo

RESULTS_DIR = Path(__file__).resolve().parent / "results"
METHODS = ("exact_dp", "smvc", "stc", "beam_dp", "tabu")
N_VALUES = (100, 200,300,400,500,600,700,800,900,1000)
K_VALUES = (2, 4, 6, 8, 10)
D_VALUES = (2, 4, 6, 8, 10)
FIXED_N = 50
FIXED_K = 3
FIXED_D = 3
N_RANDOM_INSTANCES = 3
BEAM_WIDTH = 256
TABU_ITERATIONS = 100
SAMPLE_INTERVAL = 0.002


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
            q_matrix, q_row, dits, n_neighbors, beam_width=BEAM_WIDTH
        )
    if method == "tabu":
        from qudo_solver.solvers.tabu_search import solver_tabu_search

        return lambda q_matrix, q_row: solver_tabu_search(
            q_matrix,
            q_row,
            dits,
            n_neighbors,
            time_limit=60.0,
            max_iterations=TABU_ITERATIONS,
            seed=seed,
        )
    raise ValueError(f"Unknown method: {method}")


def run_worker(method: str, n: int, k: int, d: int, seed: int) -> None:
    solver = get_solver(method, d, k, seed)
    q_matrix = generate_k_random_qudo(n, k, seed)
    q_row = [0.0] * n
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
        time.sleep(SAMPLE_INTERVAL)

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
    for n in N_VALUES:
        yield "n", n, n, FIXED_K, FIXED_D
    for k in K_VALUES:
        yield "k", k, FIXED_N, k, FIXED_D
    for d in D_VALUES:
        yield "d", d, FIXED_N, FIXED_K, d


def main() -> None:
    results = []
    for series, value, n, k, d in configurations():
        for seed in range(N_RANDOM_INSTANCES):
            for method in METHODS:
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
            "n_values": N_VALUES,
            "k_values": K_VALUES,
            "d_values": D_VALUES,
            "fixed_n": FIXED_N,
            "fixed_k": FIXED_K,
            "fixed_d": FIXED_D,
            "n_random_instances": N_RANDOM_INSTANCES,
            "beam_width": BEAM_WIDTH,
            "tabu_iterations": TABU_ITERATIONS,
            "sample_interval_s": SAMPLE_INTERVAL,
            "memory_metric": "Linux process RSS in MiB",
        },
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "experiment_7_memory.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Resultados guardados en: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "worker":
        run_worker(sys.argv[2], *(int(value) for value in sys.argv[3:]))
    else:
        main()
