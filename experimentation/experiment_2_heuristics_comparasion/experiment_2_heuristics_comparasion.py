import json
import math
import sys
from pathlib import Path
from time import perf_counter

from qudo_solver.solvers.dynamic_programming.beam_dynamic_programming_solver import (
    solver_beam_dynamic_programming,
)
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (
    solver_dynamic_programming,
)
from qudo_solver.solvers.smvc.smvc import solver_smvc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from experimentation.experiment_config import experiment_path, load_experiment
from qudo_solver.auxiliar_functions import estimate_tau_max
from qudo_solver.data_generator.qudo_problem_generator import (
    qudo_problem_generation,
)
from qudo_solver.qudo_solver_core.solution import SolutionClass
from qudo_solver.solvers.scip import solver_scip_with_metadata
from qudo_solver.solvers.tabu_search import solver_tabu_search

CONFIG = load_experiment("experiment_2")
BEAM = CONFIG["beam"]
TABU = CONFIG["tabu"]


def solve_beam_dynamic_programming_with_matching_time(
    q_matrix: list[list[float]],
    q_row: list[float],
    dits: int,
    n_neighbors: int,
    target_time: float,
    lookahead_depth: int = BEAM["lookahead_depth"],
    local_search_passes: int = BEAM["local_search_passes"],
    min_beam_width: int = BEAM["min_width"],
    max_beam_width: int = BEAM["max_width"],
    max_probes: int = BEAM["max_probes"],
) -> tuple[SolutionClass, int, float, int]:
    """Return the largest measured beam width within Matrix's time budget."""
    if target_time <= 0.0:
        raise ValueError("target_time must be positive")

    def run(beam_width: int):
        return solver_beam_dynamic_programming(
            q_matrix=q_matrix,
            q_row=q_row,
            dits=dits,
            n_neighbors=n_neighbors,
            beam_width=beam_width,
            lookahead_depth=lookahead_depth,
            local_search_passes=local_search_passes,
        )

    selected_solution = None
    selected_beam_width = None
    minimum_beam_solution = None
    probes = 0
    largest_feasible = None
    smallest_infeasible = None
    beam_width = min_beam_width
    maximum_allowed_time = (
        1.0 + BEAM["time_match_tolerance"]
    ) * target_time

    while beam_width <= max_beam_width and probes < max_probes:
        solution = run(beam_width)
        probes += 1
        if minimum_beam_solution is None:
            minimum_beam_solution = solution

        if solution.execution_time <= maximum_allowed_time:
            selected_solution = solution
            selected_beam_width = beam_width
            largest_feasible = beam_width
            next_beam_width = min(beam_width * 2, max_beam_width)
            if next_beam_width == beam_width:
                break
            beam_width = next_beam_width
        else:
            smallest_infeasible = beam_width
            break

    if minimum_beam_solution is None:
        raise RuntimeError("beam_width calibration evaluated no candidates")

    if largest_feasible is not None and smallest_infeasible is not None:
        low = largest_feasible
        high = smallest_infeasible
        while low + 1 < high and probes < max_probes:
            beam_width = (low + high) // 2
            solution = run(beam_width)
            probes += 1
            if solution.execution_time <= maximum_allowed_time:
                selected_solution = solution
                selected_beam_width = beam_width
                low = beam_width
            else:
                high = beam_width

    # If even the minimum beam exceeds the allowance, there is no beam width
    # satisfying the requested budget. Return beam 1 and expose that fact via
    # its time-match ratio instead of selecting a still larger overrun.
    if selected_solution is None or selected_beam_width is None:
        selected_solution = minimum_beam_solution
        selected_beam_width = min_beam_width

    return (
        selected_solution,
        selected_beam_width,
        selected_solution.execution_time / target_time,
        probes,
    )


def compute_method_metrics(
    method_cost: float,
    optimal_cost: float,
    *,
    abs_tol: float = 1e-9,
    rel_tol: float = 1e-8,
) -> dict:
    optimality_gap = method_cost - optimal_cost
    cost_difference = abs(optimality_gap)

    scale = max(abs(optimal_cost), abs_tol)
    relative_gap = optimality_gap / scale

    reached_optimal = math.isclose(
        method_cost,
        optimal_cost,
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )

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


def compute_partial_stability_summary(
    gaps: list[float],
    relative_gaps: list[float],
    optimal_count: int,
    total_trials: int,
) -> dict:
    """Summarize a method that may finish without a feasible incumbent."""
    feasible_count = len(gaps)
    return {
        "optimal_count": optimal_count,
        "feasible_count": feasible_count,
        "no_incumbent_count": total_trials - feasible_count,
        "total_trials": total_trials,
        "success_rate": optimal_count / total_trials,
        "feasible_rate": feasible_count / total_trials,
        "mean_gap": sum(gaps) / feasible_count if feasible_count else None,
        "mean_relative_gap": (
            sum(relative_gaps) / feasible_count if feasible_count else None
        ),
        "max_relative_gap": max(relative_gaps) if feasible_count else None,
    }


def run_experiment(dits: int, n_neighbors: int):
    results = []
    optimal_summary = []
    beam_dynamic_programming_summary = []
    tabu_search_summary = []
    scip_summary = []
    results_dir = experiment_path(CONFIG["results_dir"])
    json_path = results_dir / f"experiment_2_params_d{dits}_k{n_neighbors}.json"

    for n_variables in CONFIG["n_variables"]:
        optimal_count = 0
        gaps: list[float] = []
        relative_gaps: list[float] = []
        beam_optimal_count = 0
        beam_gaps: list[float] = []
        beam_relative_gaps: list[float] = []
        beam_time_match_ratios: list[float] = []
        tabu_search_optimal_count = 0
        tabu_search_gaps: list[float] = []
        tabu_search_relative_gaps: list[float] = []
        tabu_search_time_match_ratios: list[float] = []
        scip_optimal_count = 0
        scip_gaps: list[float] = []
        scip_relative_gaps: list[float] = []
        scip_time_match_ratios: list[float] = []
        calibrated_beam_width: int | None = None
        calibration_probes = 0
        calibration_wall_time = 0.0
        calibration_time_match_ratio = 0.0
        calibration_within_budget = False
        calibration_seed: int | None = None

        tau = estimate_tau_max(
            n_variables=n_variables,
            dits=dits,
            n_neighbors=n_neighbors,
        )

        qudo_instances = qudo_problem_generation(
            n_variables=n_variables,
            n_neighbors=n_neighbors,
            n_random_instances=len(CONFIG["seeds"]),
            n_fixed_instances=CONFIG["n_fixed_instances"],
            random_seeds=CONFIG["seeds"],
        )
        for index, instance in enumerate(qudo_instances):
            qudo_problem_matrix = instance["q_matrix"]
            qudo_problem_row = instance["q_row"]

            dynamic_programming_solution = solver_dynamic_programming(
                q_matrix=qudo_problem_matrix,
                q_row=qudo_problem_row,
                dits=dits,
                n_neighbors=n_neighbors,
                require_nonzero=False,
            )

            matrix_method_solution = solver_smvc(
                Q_list=qudo_problem_matrix,
                Q_row=qudo_problem_row,
                dits=dits,
                n_neighbors=n_neighbors,
                tau=tau,
            )
            matrix_method_time_limit = matrix_method_solution.execution_time
            tabu_search_solution = solver_tabu_search(
                q_matrix=qudo_problem_matrix,
                q_row=qudo_problem_row,
                dits=dits,
                n_neighbors=n_neighbors,
                time_limit=matrix_method_time_limit,
                tabu_tenure=TABU["tenure"],
                candidate_list_size=TABU["candidate_list_size"],
                diversification_interval=TABU["diversification_interval"],
                seed=instance["seed"],
                greedy_initialization=TABU["greedy_initialization"],
                require_nonzero=False,
            )
            tabu_search_time_match_ratio = (
                tabu_search_solution.execution_time / matrix_method_time_limit
            )

            is_calibration_instance = calibrated_beam_width is None
            if is_calibration_instance:
                calibration_started_at = perf_counter()
                (
                    beam_solution,
                    calibrated_beam_width,
                    beam_time_match_ratio,
                    calibration_probes,
                ) = solve_beam_dynamic_programming_with_matching_time(
                    q_matrix=qudo_problem_matrix,
                    q_row=qudo_problem_row,
                    dits=dits,
                    n_neighbors=n_neighbors,
                    target_time=matrix_method_time_limit,
                    max_beam_width=min(
                        BEAM["max_width"],
                        dits**n_neighbors,
                    ),
                )
                calibration_wall_time = perf_counter() - calibration_started_at
                calibration_time_match_ratio = beam_time_match_ratio
                calibration_within_budget = (
                    beam_time_match_ratio
                    <= 1.0 + BEAM["time_match_tolerance"]
                )
                calibration_seed = instance["seed"]
                beam_calibration_probes = calibration_probes
            else:
                # Calibration is completed on the first instance, so every
                # subsequent call must have a concrete beam width.
                assert calibrated_beam_width is not None
                beam_solution = solver_beam_dynamic_programming(
                    q_matrix=qudo_problem_matrix,
                    q_row=qudo_problem_row,
                    dits=dits,
                    n_neighbors=n_neighbors,
                    beam_width=calibrated_beam_width,
                    lookahead_depth=BEAM["lookahead_depth"],
                    local_search_passes=BEAM["local_search_passes"],
                )
                beam_time_match_ratio = (
                    beam_solution.execution_time
                    / matrix_method_time_limit
                )
                beam_calibration_probes = 0
            beam_width_for_instance = calibrated_beam_width
            scip_started_at = perf_counter()
            try:
                scip_solution, scip_metadata = solver_scip_with_metadata(
                    q_matrix=qudo_problem_matrix,
                    q_row=qudo_problem_row,
                    dits=dits,
                    n_neighbors=n_neighbors,
                    time_limit=matrix_method_time_limit,
                    seed=instance["seed"],
                    require_nonzero=False,
                )
            except RuntimeError as error:
                if "without finding a feasible solution" not in str(error):
                    raise
                scip_solution = None
                scip_metadata = None
                scip_error = str(error)
                scip_execution_time = perf_counter() - scip_started_at
            else:
                scip_error = None
                scip_execution_time = scip_solution.execution_time
            scip_time_match_ratio = (
                scip_execution_time / matrix_method_time_limit
            )
            matrix_metrics = compute_method_metrics(
                matrix_method_solution.cost,
                dynamic_programming_solution.cost,
            )
            beam_metrics = compute_method_metrics(
                beam_solution.cost,
                dynamic_programming_solution.cost,
            )
            tabu_search_metrics = compute_method_metrics(
                tabu_search_solution.cost,
                dynamic_programming_solution.cost,
            )
            scip_metrics = (
                compute_method_metrics(
                    scip_solution.cost,
                    dynamic_programming_solution.cost,
                )
                if scip_solution is not None
                else None
            )

            gaps.append(matrix_metrics["optimality_gap"])
            relative_gaps.append(matrix_metrics["relative_gap"])
            if matrix_metrics["reached_optimal"]:
                optimal_count += 1

            beam_gaps.append(beam_metrics["optimality_gap"])
            beam_relative_gaps.append(beam_metrics["relative_gap"])
            beam_time_match_ratios.append(beam_time_match_ratio)
            if beam_metrics["reached_optimal"]:
                beam_optimal_count += 1

            tabu_search_gaps.append(tabu_search_metrics["optimality_gap"])
            tabu_search_relative_gaps.append(tabu_search_metrics["relative_gap"])
            tabu_search_time_match_ratios.append(tabu_search_time_match_ratio)
            if tabu_search_metrics["reached_optimal"]:
                tabu_search_optimal_count += 1

            if scip_metrics is not None:
                scip_gaps.append(scip_metrics["optimality_gap"])
                scip_relative_gaps.append(scip_metrics["relative_gap"])
                if scip_metrics["reached_optimal"]:
                    scip_optimal_count += 1
            scip_time_match_ratios.append(scip_time_match_ratio)

            results.append(
                {
                    "n_variables": n_variables,
                    "instance_index": index,
                    "instance_type": instance["instance_type"],
                    "seed": instance["seed"],
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
                    "tabu_search": {
                        "seed": instance["seed"],
                        "time_limit": matrix_method_time_limit,
                        "time_match_ratio": tabu_search_time_match_ratio,
                        "time": tabu_search_solution.execution_time,
                        "cost": tabu_search_solution.cost,
                        "reached_optimal": tabu_search_metrics["reached_optimal"],
                        "optimality_gap": tabu_search_metrics["optimality_gap"],
                        "relative_gap": tabu_search_metrics["relative_gap"],
                        "cost_difference": tabu_search_metrics["cost_difference"],
                    },
                    "scip": {
                        "seed": instance["seed"],
                        "time_limit": matrix_method_time_limit,
                        "time_match_ratio": scip_time_match_ratio,
                        "time": scip_execution_time,
                        "has_incumbent": scip_solution is not None,
                        "cost": (
                            scip_solution.cost
                            if scip_solution is not None
                            else None
                        ),
                        "status": (
                            scip_metadata.status
                            if scip_metadata is not None
                            else "no_incumbent"
                        ),
                        "error": scip_error,
                        "solving_time": (
                            scip_metadata.solving_time
                            if scip_metadata is not None
                            else None
                        ),
                        "nodes": (
                            scip_metadata.nodes
                            if scip_metadata is not None
                            else None
                        ),
                        "incumbent_objective": (
                            scip_metadata.objective
                            if scip_metadata is not None
                            else None
                        ),
                        "best_bound": (
                            scip_metadata.best_bound
                            if scip_metadata is not None
                            else None
                        ),
                        "gap": (
                            scip_metadata.gap
                            if scip_metadata is not None
                            else None
                        ),
                        "reached_optimal": (
                            scip_metrics["reached_optimal"]
                            if scip_metrics is not None
                            else False
                        ),
                        "optimality_gap": (
                            scip_metrics["optimality_gap"]
                            if scip_metrics is not None
                            else None
                        ),
                        "relative_gap": (
                            scip_metrics["relative_gap"]
                            if scip_metrics is not None
                            else None
                        ),
                        "cost_difference": (
                            scip_metrics["cost_difference"]
                            if scip_metrics is not None
                            else None
                        ),
                    },
                    "beam_dynamic_programming": {
                        "beam_width": beam_width_for_instance,
                        "is_calibration_instance": is_calibration_instance,
                        "beam_width_calibration_probes": (
                            beam_calibration_probes
                        ),
                        "calibration_within_budget": (
                            calibration_within_budget
                        ),
                        "calibration_wall_time": (
                            calibration_wall_time
                            if is_calibration_instance
                            else None
                        ),
                        "time_match_ratio": beam_time_match_ratio,
                        "lookahead_depth": BEAM["lookahead_depth"],
                        "local_search_passes": BEAM["local_search_passes"],
                        "time": beam_solution.execution_time,
                        "cost": beam_solution.cost,
                        "reached_optimal": beam_metrics["reached_optimal"],
                        "optimality_gap": beam_metrics["optimality_gap"],
                        "relative_gap": beam_metrics["relative_gap"],
                        "cost_difference": beam_metrics["cost_difference"],
                    },
                }
            )

        print("\\")
        print(f"Run with dits={dits}, k={n_neighbors}, n={n_variables}")
        total_trials = len(qudo_instances)

        stability_summary = compute_stability_summary(
            gaps=gaps,
            relative_gaps=relative_gaps,
            optimal_count=optimal_count,
            total_trials=total_trials,
        )
        beam_stability_summary = compute_stability_summary(
            gaps=beam_gaps,
            relative_gaps=beam_relative_gaps,
            optimal_count=beam_optimal_count,
            total_trials=total_trials,
        )
        beam_stability_summary["mean_time_match_ratio"] = (
            sum(beam_time_match_ratios) / total_trials
        )
        beam_stability_summary.update(
            {
                "beam_width": calibrated_beam_width,
                "calibration_instance_index": 0,
                "calibration_seed": calibration_seed,
                "calibration_probes": calibration_probes,
                "calibration_wall_time": calibration_wall_time,
                "calibration_time_match_ratio": calibration_time_match_ratio,
                "calibration_within_budget": calibration_within_budget,
            }
        )
        tabu_search_stability_summary = compute_stability_summary(
            gaps=tabu_search_gaps,
            relative_gaps=tabu_search_relative_gaps,
            optimal_count=tabu_search_optimal_count,
            total_trials=total_trials,
        )
        tabu_search_stability_summary["mean_time_match_ratio"] = (
            sum(tabu_search_time_match_ratios) / total_trials
        )
        scip_stability_summary = compute_partial_stability_summary(
            gaps=scip_gaps,
            relative_gaps=scip_relative_gaps,
            optimal_count=scip_optimal_count,
            total_trials=total_trials,
        )
        scip_stability_summary["mean_time_match_ratio"] = (
            sum(scip_time_match_ratios) / total_trials
        )
        optimal_summary.append(
            {
                "n_variables": n_variables,
                **stability_summary,
            }
        )
        beam_dynamic_programming_summary.append(
            {
                "n_variables": n_variables,
                **beam_stability_summary,
            }
        )
        tabu_search_summary.append(
            {
                "n_variables": n_variables,
                **tabu_search_stability_summary,
            }
        )
        scip_summary.append(
            {
                "n_variables": n_variables,
                **scip_stability_summary,
            }
        )

        print(
            f"n={n_variables}: matrix optimal {optimal_count}/{total_trials}, "
            f"mean_relative_gap={stability_summary['mean_relative_gap']:.6f}, "
            f"max_relative_gap={stability_summary['max_relative_gap']:.6f}"
        )
        print(
            f"n={n_variables}: Beam DP optimal {beam_optimal_count}/{total_trials}, "
            f"beam_width={calibrated_beam_width}, "
            f"mean_relative_gap={beam_stability_summary['mean_relative_gap']:.6f}, "
            f"max_relative_gap={beam_stability_summary['max_relative_gap']:.6f}, "
            f"mean_time_match_ratio={beam_stability_summary['mean_time_match_ratio']:.3f}"
        )
        print(
            f"n={n_variables}: tabu optimal {tabu_search_optimal_count}/{total_trials}, "
            f"mean_relative_gap={tabu_search_stability_summary['mean_relative_gap']:.6f}, "
            f"max_relative_gap={tabu_search_stability_summary['max_relative_gap']:.6f}, "
            f"mean_time_match_ratio="
            f"{tabu_search_stability_summary['mean_time_match_ratio']:.3f}"
        )
        print(
            f"n={n_variables}: SCIP optimal {scip_optimal_count}/{total_trials}, "
            f"incumbents={scip_stability_summary['feasible_count']}/{total_trials}, "
            f"mean_relative_gap={scip_stability_summary['mean_relative_gap']}, "
            f"max_relative_gap={scip_stability_summary['max_relative_gap']}, "
            f"mean_time_match_ratio="
            f"{scip_stability_summary['mean_time_match_ratio']:.3f}"
        )

    experiment_data = {
        "parameters": {
            "dits": dits,
            "n_neighbors": n_neighbors,
            "n_variables": CONFIG["n_variables"],
            "instance_generation": {
                "n_random_instances": len(CONFIG["seeds"]),
                "seeds": CONFIG["seeds"],
                "n_fixed_instances": CONFIG["n_fixed_instances"],
                "seeds": {
                    "random": CONFIG["seeds"],
                    "fixed": list(range(CONFIG["n_fixed_instances"])),
                },
            },
            "optimal_tolerance": CONFIG["optimal_tolerance"],
            "beam_dynamic_programming": {
                "beam_width_strategy": "calibrate_on_first_instance_per_d_k_n",
                "time_limit_source": "first_matrix_method.execution_time",
                "minimum_beam_width": BEAM["min_width"],
                "maximum_beam_width": BEAM["max_width"],
                "maximum_time_overrun_fraction": (
                    BEAM["time_match_tolerance"]
                ),
                "maximum_calibration_probes": BEAM["max_probes"],
                "lookahead_depth": BEAM["lookahead_depth"],
                "local_search_passes": BEAM["local_search_passes"],
            },
            "tabu_search": {
                "time_limit_source": "matrix_method.execution_time",
                "tabu_tenure": TABU["tenure"],
                "candidate_list_size": TABU["candidate_list_size"],
                "diversification_interval": TABU["diversification_interval"],
                "greedy_initialization": TABU["greedy_initialization"],
                "seed_source": "instance.seed",
            },
            "scip": {
                "time_limit_source": "matrix_method.execution_time",
                "time_limit_semantics": "total_wall_clock_budget",
                "seed_source": "instance.seed",
            },
        },
        "matrix_method_summary": optimal_summary,
        "beam_dynamic_programming_summary": beam_dynamic_programming_summary,
        "tabu_search_summary": tabu_search_summary,
        "scip_summary": scip_summary,
        "results": results,
    }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(experiment_data, indent=2))
    print(f"JSON saved in: {json_path}")


def main():
    for dits in CONFIG["dits_values"]:
        for n_neighbors in CONFIG["k_values"]:
            print(f"Starting experiment for dits={dits}, k={n_neighbors}")
            run_experiment(dits=dits, n_neighbors=n_neighbors)


if __name__ == "__main__":
    main()
    # n_variables = 5
    # dits = 2
    # n_neighbors = 2
    # seed = 7
    # qubo_problem = generate_k_qubo(
    #     n_variables=n_variables,
    #     k_neighbor=n_neighbors,
    #     seed=seed,
    # )
    # # qubo_problem = normalize_list_of_lists(qubo_problem)

    # # qubo_problem = generate_frustrated_k_qubo(
    # #     n_variables=n_variables,
    # #     k_neighbor=n_neighbors,
    # #     seed=seed,
    # #     frustration_probability=1.0,
    # # )
    # print(qubo_problem)
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

    # beam_solution = solver_beam_dynamic_programming(
    #     q_matrix=qubo_problem,
    #     dits=dits,
    #     n_neighbors=n_neighbors,
    #     beam_width=20,
    #     lookahead_depth=0,
    #     local_search_passes=0,
    # )
    # print("dynamic cost", dynamic_programming_solution.cost, dynamic_programming_solution.execution_time)
    # print("matrix cost", matrix_method_solution.cost, matrix_method_solution.execution_time)
    # print("Beam DP cost", beam_solution.cost, beam_solution.execution_time)
