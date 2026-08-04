import json
import sys
from pathlib import Path


from qudo_solver.auxiliar_functions import estimate_tau_max
from qudo_solver.data_generator.qudo_problem_generator import generate_frustrated_k_qubo, generate_k_qubo, normalize_list_of_lists

from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver2 import solver_dynamic_programming2
from qudo_solver.solvers.dynamic_programming.heuristic_dynamic_programming_solver import solver_dynamic_programming_heuristic
from qudo_solver.solvers.matrix_method.matrix_method_solver import solver_matrix_method


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULTS_DIR = PROJECT_ROOT / "experimentation" / "experiment_1_correctness_vs_optimal" / "results"

N_VARIABLES = list(range(10, 1001, 100))
DITS_VALUES = [2]
K_VALUES = [4,5,6]
SEED = list(range(1, 101, 20))
OPTIMAL_TOLERANCE = 1e-9
HEURISTIC_LOOKAHEAD_DEPTH = 0
HEURISTIC_LOCAL_SEARCH_PASSES = 0
HEURISTIC_MIN_BEAM_WIDTH = 1
HEURISTIC_MAX_BEAM_WIDTH = 4096
HEURISTIC_TIME_MATCH_TOLERANCE = 0.25
HEURISTIC_BEAM_WIDTH_MAX_PROBES = 10


def solve_heuristic_with_matching_time(
    q_matrix: list[list[float]],
    dits: int,
    n_neighbors: int,
    target_time: float,
    lookahead_depth: int = HEURISTIC_LOOKAHEAD_DEPTH,
    local_search_passes: int = HEURISTIC_LOCAL_SEARCH_PASSES,
    min_beam_width: int = HEURISTIC_MIN_BEAM_WIDTH,
    max_beam_width: int = HEURISTIC_MAX_BEAM_WIDTH,
    max_probes: int = HEURISTIC_BEAM_WIDTH_MAX_PROBES,
) -> tuple:
    """Pick beam_width so heuristic runtime is close to target_time."""

    def run_heuristic(beam_width: int):
        return solver_dynamic_programming_heuristic(
            q_matrix=q_matrix,
            dits=dits,
            n_neighbors=n_neighbors,
            beam_width=beam_width,
            lookahead_depth=lookahead_depth,
            local_search_passes=local_search_passes,
        )

    best_beam_width = min_beam_width
    best_solution = None
    best_time_diff = float("inf")
    probes = 0

    below_beam_width = None
    above_beam_width = None
    beam_width = min_beam_width

    while beam_width <= max_beam_width and probes < max_probes:
        solution = run_heuristic(beam_width)
        probes += 1
        time_diff = abs(solution.execution_time - target_time)

        if time_diff < best_time_diff:
            best_time_diff = time_diff
            best_beam_width = beam_width
            best_solution = solution

        if solution.execution_time < target_time:
            below_beam_width = beam_width
            next_beam_width = beam_width * 2
            if next_beam_width > max_beam_width:
                break
            beam_width = next_beam_width
        else:
            above_beam_width = beam_width
            break

    if best_solution is None:
        raise RuntimeError("beam_width calibration failed to evaluate any candidate")

    if (
        below_beam_width is not None
        and above_beam_width is not None
        and below_beam_width < above_beam_width - 1
        and probes < max_probes
    ):
        low = below_beam_width
        high = above_beam_width

        while low + 1 < high and probes < max_probes:
            mid_beam_width = (low + high) // 2
            solution = run_heuristic(mid_beam_width)
            probes += 1
            time_diff = abs(solution.execution_time - target_time)

            if time_diff < best_time_diff:
                best_time_diff = time_diff
                best_beam_width = mid_beam_width
                best_solution = solution

            if solution.execution_time < target_time:
                low = mid_beam_width
            else:
                high = mid_beam_width

    time_match_ratio = best_solution.execution_time / target_time
    return best_solution, best_beam_width, time_match_ratio, probes


def compute_method_metrics(
    method_cost: float,
    optimal_cost: float,
) -> dict:
    optimality_gap = method_cost - optimal_cost
    relative_gap = optimality_gap / (abs(optimal_cost) + OPTIMAL_TOLERANCE)
    cost_difference = abs(optimality_gap)
    reached_optimal = cost_difference < OPTIMAL_TOLERANCE
    return {
        "optimality_gap": optimality_gap,
        "relative_gap": relative_gap,
        "cost_difference": cost_difference,
        "reached_optimal": reached_optimal,
    }


def compute_stability_summary(
    gaps: list[float],
    relative_gaps: list[float],
    optimal_count: int,
    total_trials: int,
) -> dict:
    return {
        "optimal_count": optimal_count,
        "total_trials": total_trials,
        "success_rate": optimal_count / total_trials,
        "mean_gap": sum(gaps) / total_trials,
        "mean_relative_gap": sum(relative_gaps) / total_trials,
        "max_relative_gap": max(relative_gaps),
    }


def run_experiment(dits: int, n_neighbors: int):
    dynamic_programming_times = []
    matrix_method_times = []
    results = []
    optimal_summary = []
    heuristic_summary = []
    json_path = RESULTS_DIR / f"experiment_1_params_d{dits}_k{n_neighbors}.json"

    for n_variables in N_VARIABLES:
        optimal_count = 0
        gaps: list[float] = []
        relative_gaps: list[float] = []
        heuristic_optimal_count = 0
        heuristic_gaps: list[float] = []
        heuristic_relative_gaps: list[float] = []
        heuristic_time_match_ratios: list[float] = []

        for seed in SEED:
            qubo_problem = generate_k_qubo(
                n_variables=n_variables,
                k_neighbor=n_neighbors,
                seed=seed,
            )
            qubo_problem = generate_frustrated_k_qubo(
                n_variables=n_variables,
                k_neighbor=n_neighbors,
                seed=seed,
                frustration_probability=1.0,
            )
            #qubo_problem = normalize_list_of_lists(qubo_problem)

            dynamic_programming_solution = solver_dynamic_programming2(
                q_matrix=qubo_problem,
                dits=dits,
                n_neighbors=n_neighbors,
            )
            tau = estimate_tau_max(
                n_variables=n_variables,
                n_neighbors=n_neighbors,
                dits=dits,
            )
            matrix_method_solution = solver_matrix_method(
                Q_list=qubo_problem,
                dits=dits,
                n_neighbors=n_neighbors,
            )
            heuristic_solution, heuristic_beam_width, heuristic_time_match_ratio, heuristic_probes = (
                solve_heuristic_with_matching_time(
                    q_matrix=qubo_problem,
                    dits=dits,
                    n_neighbors=n_neighbors,
                    target_time=matrix_method_solution.execution_time,
                )
            )

            matrix_metrics = compute_method_metrics(
                matrix_method_solution.cost,
                dynamic_programming_solution.cost,
            )
            heuristic_metrics = compute_method_metrics(
                heuristic_solution.cost,
                dynamic_programming_solution.cost,
            )

            gaps.append(matrix_metrics["optimality_gap"])
            relative_gaps.append(matrix_metrics["relative_gap"])
            if matrix_metrics["reached_optimal"]:
                optimal_count += 1

            heuristic_gaps.append(heuristic_metrics["optimality_gap"])
            heuristic_relative_gaps.append(heuristic_metrics["relative_gap"])
            heuristic_time_match_ratios.append(heuristic_time_match_ratio)
            if heuristic_metrics["reached_optimal"]:
                heuristic_optimal_count += 1

            dynamic_programming_times.append(dynamic_programming_solution.execution_time)
            matrix_method_times.append(matrix_method_solution.execution_time)
            results.append(
                {
                    "n_variables": n_variables,
                    "seed": seed,
                    "reached_optimal": matrix_metrics["reached_optimal"],
                    "optimality_gap": matrix_metrics["optimality_gap"],
                    "relative_gap": matrix_metrics["relative_gap"],
                    "cost_difference": matrix_metrics["cost_difference"],
                    "dynamic_programming": {
                        "time": dynamic_programming_solution.execution_time,
                        "cost": dynamic_programming_solution.cost,
                    },
                    "matrix_method": {
                        "tau": tau,
                        "time": matrix_method_solution.execution_time,
                        "cost": matrix_method_solution.cost,
                    },
                    "heuristic": {
                        "beam_width": heuristic_beam_width,
                        "beam_width_calibration_probes": heuristic_probes,
                        "time_match_ratio": heuristic_time_match_ratio,
                        "lookahead_depth": HEURISTIC_LOOKAHEAD_DEPTH,
                        "local_search_passes": HEURISTIC_LOCAL_SEARCH_PASSES,
                        "time": heuristic_solution.execution_time,
                        "cost": heuristic_solution.cost,
                        "reached_optimal": heuristic_metrics["reached_optimal"],
                        "optimality_gap": heuristic_metrics["optimality_gap"],
                        "relative_gap": heuristic_metrics["relative_gap"],
                        "cost_difference": heuristic_metrics["cost_difference"],
                    },
                }
            )

        print("\\")
        print(f"Prueba con dits={dits}, k={n_neighbors}, n={n_variables}")

        stability_summary = compute_stability_summary(
            gaps=gaps,
            relative_gaps=relative_gaps,
            optimal_count=optimal_count,
            total_trials=len(SEED),
        )
        heuristic_stability_summary = compute_stability_summary(
            gaps=heuristic_gaps,
            relative_gaps=heuristic_relative_gaps,
            optimal_count=heuristic_optimal_count,
            total_trials=len(SEED),
        )
        heuristic_stability_summary["mean_time_match_ratio"] = (
            sum(heuristic_time_match_ratios) / len(SEED)
        )
        optimal_summary.append(
            {
                "n_variables": n_variables,
                **stability_summary,
            }
        )
        heuristic_summary.append(
            {
                "n_variables": n_variables,
                **heuristic_stability_summary,
            }
        )
        print(
            f"n={n_variables}: matrix optimal {optimal_count}/{len(SEED)}, "
            f"mean_relative_gap={stability_summary['mean_relative_gap']:.6f}, "
            f"max_relative_gap={stability_summary['max_relative_gap']:.6f}"
        )
        print(
            f"n={n_variables}: heuristic optimal {heuristic_optimal_count}/{len(SEED)}, "
            f"mean_relative_gap={heuristic_stability_summary['mean_relative_gap']:.6f}, "
            f"max_relative_gap={heuristic_stability_summary['max_relative_gap']:.6f}, "
            f"mean_time_match_ratio={heuristic_stability_summary['mean_time_match_ratio']:.3f}"
        )

    experiment_data = {
        "parameters": {
            "seed": SEED,
            "dits": dits,
            "n_neighbors": n_neighbors,
            "n_variables": N_VARIABLES,
            "optimal_tolerance": OPTIMAL_TOLERANCE,
            "heuristic": {
                "beam_width_calibration": {
                    "min_beam_width": HEURISTIC_MIN_BEAM_WIDTH,
                    "max_beam_width": HEURISTIC_MAX_BEAM_WIDTH,
                    "time_match_tolerance": HEURISTIC_TIME_MATCH_TOLERANCE,
                    "max_probes": HEURISTIC_BEAM_WIDTH_MAX_PROBES,
                },
                "lookahead_depth": HEURISTIC_LOOKAHEAD_DEPTH,
                "local_search_passes": HEURISTIC_LOCAL_SEARCH_PASSES,
            },
        },
        "matrix_method_summary": optimal_summary,
        "heuristic_summary": heuristic_summary,
        "results": results,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(experiment_data, indent=2))
    print(f"JSON saved in: {json_path}")


def main():
    for dits in DITS_VALUES:
        for n_neighbors in K_VALUES:
            print(f"Starting experiment for dits={dits}, k={n_neighbors}")
            run_experiment(dits=dits, n_neighbors=n_neighbors)


if __name__ == "__main__":
    main()
    # n_variables = 110
    # dits = 3
    # n_neighbors = 3
    # seed = 7
    # qubo_problem = generate_k_qubo(
    #     n_variables=n_variables,
    #     k_neighbor=n_neighbors,
    #     seed=seed,
    # )
    # # qubo_problem = normalize_list_of_lists(qubo_problem)

    # dynamic_programming_solution = solver_dynamic_programming2(
    #     q_matrix=qubo_problem,
    #     dits=dits,
    #     n_neighbors=n_neighbors,
    # )

    # tau = estimate_tau_max(
    #    n_neighbors=n_neighbors,
    #    n_variables=n_variables,
    #     dits=dits,
    # )

    # print(tau)
    # matrix_method_solution = solver_matrix_method(
    #     Q_list=qubo_problem,
    #     dits=dits,
    #     tau = 200,
    #     n_neighbors=n_neighbors,
    # )

    # heuristic_solution = solver_dynamic_programming_heuristic(
    #     q_matrix=qubo_problem,
    #     dits=dits,
    #     n_neighbors=n_neighbors,
    #     beam_width=20,
    #     lookahead_depth=0,
    #     local_search_passes=0,
    # )
    # print("dynamic cost", dynamic_programming_solution.cost, dynamic_programming_solution.execution_time)
    # print("matrix cost", matrix_method_solution.cost, matrix_method_solution.execution_time)
    # print("heuristic cost", heuristic_solution.cost, heuristic_solution.execution_time)