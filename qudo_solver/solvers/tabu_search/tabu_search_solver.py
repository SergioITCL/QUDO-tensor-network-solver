"""Tabu Search specialized for banded, finite-range QUDO objectives.

The compact row for variable ``i`` contains interactions with consecutive
previous variables and ends with its diagonal coefficient.  Incident-neighbor
lists and an exact move-delta cache exploit this locality without introducing
any ``d**k`` dependency.

With ``n`` variables, ``d`` values, and band width ``k``, preprocessing costs
``O(n*(k+d))``.  An exhaustive best-move scan costs ``O(n*d)`` over the cache.
After changing one variable, only it and its ``O(k)`` neighbors are refreshed,
at cost ``O(k*(k+d))``.  Candidate-list selection costs ``O(n)`` plus
``O(candidate_list_size*d)`` for admissibility scanning.  Memory is ``O(n*d +
n*k)``.  Without the cache, an exhaustive iteration would cost ``O(n*d*k)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import ceil, isfinite
from numbers import Integral
from time import perf_counter
from typing import Callable, Iterable, Sequence

import numpy as np

from qudo_solver.qudo_solver_core.solution import SolutionClass

Neighbors = list[list[tuple[int, float]]]


@dataclass(frozen=True)
class _Move:
    variable: int
    new_value: int
    delta: float


@dataclass
class TabuTargetResult:
    """Result of one Tabu Search run stopped at a requested target cost."""

    reached: bool
    target_cost: float
    time_to_target: float | None
    solution: SolutionClass
    best_cost: float
    total_execution_time: float
    incumbent_history: list[tuple[float, float]]


@dataclass
class _TargetTracker:
    target_cost: float
    tolerance: float
    reached: bool = False
    time_to_target: float | None = None
    target_solution: list[int] | None = None
    best_external_cost: float | None = None
    history: list[tuple[float, float]] = field(default_factory=list)


def _validate_problem(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
) -> None:
    """Validate the compact band representation and finite coefficients."""
    if len(q_matrix) == 0:
        raise ValueError("q_matrix must contain at least one variable")
    if len(q_row) != len(q_matrix):
        raise ValueError("q_matrix and q_row must have the same length")
    if not isinstance(dits, Integral) or isinstance(dits, bool) or dits < 2:
        raise ValueError("dits must be an integer of at least 2")
    if (
        not isinstance(n_neighbors, Integral)
        or isinstance(n_neighbors, bool)
        or n_neighbors < 0
    ):
        raise ValueError("n_neighbors must be a non-negative integer")

    for position, (row, linear) in enumerate(zip(q_matrix, q_row)):
        if len(row) == 0:
            raise ValueError(f"q_matrix[{position}] must not be empty")
        maximum_length = min(position, n_neighbors) + 1
        if len(row) > maximum_length:
            raise ValueError(
                f"q_matrix[{position}] contains {len(row) - 1} interactions, "
                f"but at most {maximum_length - 1} are allowed for "
                f"n_neighbors={n_neighbors}"
            )
        for coefficient in row:
            try:
                finite = isfinite(float(coefficient))
            except (TypeError, ValueError) as error:
                raise ValueError("q_matrix coefficients must be numeric") from error
            if not finite:
                raise ValueError("q_matrix coefficients must be finite")
        try:
            finite_linear = isfinite(float(linear))
        except (TypeError, ValueError) as error:
            raise ValueError("q_row coefficients must be numeric") from error
        if not finite_linear:
            raise ValueError("q_row coefficients must be finite")


def _objective_energy(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    solution: Sequence[int],
) -> float:
    """Evaluate the complete compact QUDO objective in ``O(n*k)``."""
    if len(solution) != len(q_matrix) or len(q_row) != len(q_matrix):
        raise ValueError("q_matrix, q_row, and solution must have equal length")

    energy = 0.0
    for i, row in enumerate(q_matrix):
        value = solution[i]
        first_previous = i - len(row) + 1
        energy += float(row[-1]) * value * value + float(q_row[i]) * value
        energy += sum(
            float(coefficient) * solution[first_previous + offset] * value
            for offset, coefficient in enumerate(row[:-1])
        )
    return float(energy)


def _build_neighbors(q_matrix: Sequence[Sequence[float]]) -> Neighbors:
    """Build both directions of every pair interaction in ``O(n*k)``."""
    neighbors: Neighbors = [[] for _ in q_matrix]
    for later, row in enumerate(q_matrix):
        first_previous = later - len(row) + 1
        for offset, coefficient in enumerate(row[:-1]):
            earlier = first_previous + offset
            coefficient = float(coefficient)
            neighbors[earlier].append((later, coefficient))
            neighbors[later].append((earlier, coefficient))
    return neighbors


def _delta_energy(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    solution: Sequence[int],
    variable: int,
    new_value: int,
    neighbors: Neighbors | None = None,
) -> float:
    """Evaluate one replacement using only its incident terms in ``O(k)``."""
    if neighbors is None:
        neighbors = _build_neighbors(q_matrix)
    old_value = solution[variable]
    difference = new_value - old_value
    if difference == 0:
        return 0.0
    neighbor_field = sum(
        coefficient * solution[neighbor]
        for neighbor, coefficient in neighbors[variable]
    )
    diagonal = float(q_matrix[variable][-1])
    return float(
        diagonal * (new_value * new_value - old_value * old_value)
        + float(q_row[variable]) * difference
        + neighbor_field * difference
    )


def _compute_delta_row(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    solution: Sequence[int],
    variable: int,
    dits: int,
    neighbors: Neighbors,
) -> np.ndarray:
    """Compute deltas to every value in shared ``O(k+d)`` work."""
    old_value = solution[variable]
    values = np.arange(dits, dtype=float)
    differences = values - old_value
    neighbor_field = sum(
        coefficient * solution[neighbor]
        for neighbor, coefficient in neighbors[variable]
    )
    diagonal = float(q_matrix[variable][-1])
    return (
        diagonal * (values * values - old_value * old_value)
        + (float(q_row[variable]) + neighbor_field) * differences
    )


def _best_nontrivial_delta(row: np.ndarray, current_value: int) -> float:
    """Return a row's best actual move, excluding its zero-delta no-op."""
    if current_value == 0:
        return float(np.min(row[1:]))
    if current_value == len(row) - 1:
        return float(np.min(row[:-1]))
    return float(min(np.min(row[:current_value]), np.min(row[current_value + 1 :])))


def _initialize_delta_cache(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    solution: Sequence[int],
    dits: int,
    neighbors: Neighbors,
    deadline: float | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    cache = np.empty((len(solution), dits), dtype=float)
    scores = np.empty(len(solution), dtype=float)
    for variable in range(len(solution)):
        cache[variable] = _compute_delta_row(
            q_matrix, q_row, solution, variable, dits, neighbors
        )
        scores[variable] = _best_nontrivial_delta(
            cache[variable], solution[variable]
        )
        if variable % 64 == 63 and deadline is not None:
            if perf_counter() >= deadline:
                return None
    return cache, scores


def _refresh_delta_cache(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    solution: Sequence[int],
    dits: int,
    neighbors: Neighbors,
    variables: Iterable[int],
    cache: np.ndarray,
    scores: np.ndarray,
    deadline: float | None = None,
) -> bool:
    """Refresh exactly the rows affected by a one-variable move."""
    for position, variable in enumerate(variables):
        cache[variable] = _compute_delta_row(
            q_matrix, q_row, solution, variable, dits, neighbors
        )
        scores[variable] = _best_nontrivial_delta(
            cache[variable], solution[variable]
        )
        if position % 32 == 31 and deadline is not None:
            if perf_counter() >= deadline:
                return False
    return True


def _candidate_variables(
    scores: np.ndarray,
    candidate_list_size: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """Mix best local scores with a small random variable subset."""
    n = len(scores)
    if candidate_list_size is None or candidate_list_size >= n:
        return np.arange(n, dtype=int)

    size = candidate_list_size
    promising_count = max(1, int(0.8 * size))
    promising = np.argpartition(scores, promising_count - 1)[:promising_count]
    random_count = size - promising_count
    if random_count == 0:
        return promising

    available_mask = np.ones(n, dtype=bool)
    available_mask[promising] = False
    available = np.flatnonzero(available_mask)
    random_variables = rng.choice(available, size=random_count, replace=False)
    return np.concatenate((promising, random_variables)).astype(int, copy=False)


def _is_move_admissible(
    tabu_until: np.ndarray,
    variable: int,
    new_value: int,
    iteration: int,
    candidate_energy: float,
    best_energy: float,
) -> bool:
    """Apply tabu expiration and global-best aspiration."""
    is_tabu = iteration <= int(tabu_until[variable, new_value])
    return not is_tabu or candidate_energy < best_energy


def _best_admissible_move(
    delta_cache: np.ndarray,
    solution: Sequence[int],
    candidate_variables: Sequence[int],
    tabu_until: np.ndarray,
    iteration: int,
    current_energy: float,
    best_energy: float,
    require_nonzero: bool,
    nonzero_count: int,
    deadline: float | None = None,
) -> tuple[_Move | None, _Move | None, bool]:
    """Return best admissible and best feasible fallback moves.

    The fallback lets the caller clear the tabu list rather than terminate when
    every feasible move is temporarily tabu.  Positive-delta moves remain
    eligible, which is essential to Tabu Search.
    """
    best_move: _Move | None = None
    fallback: _Move | None = None
    evaluated_moves = 0
    for position, variable_value in enumerate(candidate_variables):
        variable = int(variable_value)
        old_value = solution[variable]
        for new_value, delta_value in enumerate(delta_cache[variable]):
            if new_value == old_value:
                continue
            if (
                require_nonzero
                and nonzero_count == 1
                and old_value != 0
                and new_value == 0
            ):
                continue
            evaluated_moves += 1
            move = _Move(variable, new_value, float(delta_value))
            if fallback is None or move.delta < fallback.delta:
                fallback = move
            candidate_energy = current_energy + move.delta
            if _is_move_admissible(
                tabu_until,
                variable,
                new_value,
                iteration,
                candidate_energy,
                best_energy,
            ) and (best_move is None or move.delta < best_move.delta):
                best_move = move

            if evaluated_moves % 256 == 0 and deadline is not None:
                if perf_counter() >= deadline:
                    return best_move, fallback, True

        if position % 32 == 31 and deadline is not None:
            if perf_counter() >= deadline:
                return best_move, fallback, True
    return best_move, fallback, False


def _short_greedy_descent(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    solution: list[int],
    dits: int,
    neighbors: Neighbors,
    current_energy: float,
    nonzero_count: int,
    require_nonzero: bool,
    deadline: float,
    on_improvement: Callable[[Sequence[int]], bool] | None = None,
) -> tuple[float, int, bool]:
    """Apply one inexpensive coordinate-descent pass for initialization."""
    for block_start in range(0, len(solution), 64):
        for variable in range(block_start, min(block_start + 64, len(solution))):
            row = _compute_delta_row(
                q_matrix, q_row, solution, variable, dits, neighbors
            )
            old_value = solution[variable]
            if require_nonzero and nonzero_count == 1 and old_value != 0:
                row[0] = np.inf
            new_value = int(np.argmin(row))
            delta = float(row[new_value])
            if delta < 0.0:
                solution[variable] = new_value
                nonzero_count += (new_value != 0) - (old_value != 0)
                current_energy += delta
                if on_improvement is not None and on_improvement(solution):
                    return current_energy, nonzero_count, True
        if perf_counter() >= deadline:
            break
    return current_energy, nonzero_count, False


def _diversify(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    solution: list[int],
    dits: int,
    require_nonzero: bool,
    rng: np.random.Generator,
) -> tuple[float, int, list[int]]:
    """Perturb 5--15% of variables and evaluate the result exactly once."""
    n = len(solution)
    fraction = float(rng.uniform(0.05, 0.15))
    perturbation_size = min(n, max(1, ceil(fraction * n)))
    variables = rng.choice(n, size=perturbation_size, replace=False).tolist()
    previous_solution = solution.copy()

    for variable_value in variables:
        variable = int(variable_value)
        old_value = solution[variable]
        candidate = int(rng.integers(dits - 1))
        new_value = candidate + (candidate >= old_value)
        solution[variable] = new_value

    nonzero_count = sum(value != 0 for value in solution)
    if require_nonzero and nonzero_count == 0:
        # Prefer a formerly-zero perturbed variable so the repair remains a
        # genuine change.  Only the one-variable binary problem makes that
        # impossible while satisfying both feasibility and distinctness.
        variable = next(
            (
                int(candidate)
                for candidate in variables
                if previous_solution[int(candidate)] == 0
            ),
            int(variables[0]),
        )
        solution[variable] = int(rng.integers(1, dits))
        nonzero_count = 1

    energy = _objective_energy(q_matrix, q_row, solution)
    return energy, nonzero_count, [int(variable) for variable in variables]


def _solve_tabu_search(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
    time_limit: float,
    require_nonzero: bool = False,
    tabu_tenure: int | None = None,
    candidate_list_size: int | None = None,
    diversification_interval: int = 500,
    seed: int | None = None,
    greedy_initialization: bool = True,
    max_iterations: int | None = None,
    *,
    started_at: float,
    target_cost: float | None = None,
    target_tolerance: float = 1e-9,
) -> tuple[SolutionClass, _TargetTracker | None]:
    """Shared implementation for budget and time-to-target modes.

    ``time_limit`` is a total wall-clock budget including validation,
    preprocessing, initialization, search, and result extraction. A small
    unavoidable overhead may occur after the deadline while returning.

    A move marks only the reverse value tabu.  Automatic tenure is
    ``min(100, max(5, int(0.1*n)))`` and each accepted move adds random jitter
    below ``max(2, tenure//4)``.  Aspiration admits a tabu move whenever it
    improves the global best.  If no admissible move exists, the tabu list is
    cleared and the best feasible move is used so the search does not stop at a
    local minimum.

    Candidate-list mode uses 80% locally promising variables and 20% random
    variables.  After ``diversification_interval`` iterations without global
    improvement, 5--15% of variables are perturbed and the tabu list is reset.
    Wall-clock runs can differ slightly across machines because throughput
    changes the number of completed iterations.
    """
    _validate_problem(q_matrix, q_row, dits, n_neighbors)
    if not isfinite(time_limit) or time_limit <= 0.0:
        raise ValueError("time_limit must be positive and finite")
    if tabu_tenure is not None and (
        not isinstance(tabu_tenure, Integral)
        or isinstance(tabu_tenure, bool)
        or tabu_tenure < 1
    ):
        raise ValueError("tabu_tenure must be a positive integer")
    if candidate_list_size is not None and (
        not isinstance(candidate_list_size, Integral)
        or isinstance(candidate_list_size, bool)
        or candidate_list_size < 1
    ):
        raise ValueError("candidate_list_size must be a positive integer")
    if (
        not isinstance(diversification_interval, Integral)
        or isinstance(diversification_interval, bool)
        or diversification_interval < 1
    ):
        raise ValueError("diversification_interval must be a positive integer")
    if max_iterations is not None and (
        not isinstance(max_iterations, Integral)
        or isinstance(max_iterations, bool)
        or max_iterations < 0
    ):
        raise ValueError("max_iterations must be a non-negative integer")
    try:
        numeric_target = None if target_cost is None else float(target_cost)
    except (TypeError, ValueError) as error:
        raise ValueError("target_cost must be numeric and finite") from error
    if numeric_target is not None and not isfinite(numeric_target):
        raise ValueError("target_cost must be numeric and finite")
    try:
        numeric_tolerance = float(target_tolerance)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "target_tolerance must be non-negative and finite"
        ) from error
    if not isfinite(numeric_tolerance) or numeric_tolerance < 0.0:
        raise ValueError("target_tolerance must be non-negative and finite")

    n = len(q_matrix)
    tenure = (
        min(100, max(5, int(0.1 * n)))
        if tabu_tenure is None
        else int(tabu_tenure)
    )
    candidate_list_size = (
        None if candidate_list_size is None else min(candidate_list_size, n)
    )
    neighbors = _build_neighbors(q_matrix)
    rng = np.random.default_rng(seed)
    deadline = started_at + time_limit
    tracker = (
        _TargetTracker(numeric_target, numeric_tolerance)
        if numeric_target is not None
        else None
    )

    def record_incumbent(solution: Sequence[int]) -> bool:
        if tracker is None:
            return False
        external_cost = _objective_energy(q_matrix, q_row, solution)
        if (
            tracker.best_external_cost is not None
            and external_cost >= tracker.best_external_cost
        ):
            return tracker.reached
        elapsed = perf_counter() - started_at
        tracker.best_external_cost = external_cost
        tracker.history.append((elapsed, external_cost))
        if (
            elapsed <= time_limit
            and external_cost <= tracker.target_cost + tracker.tolerance
        ):
            tracker.reached = True
            tracker.time_to_target = elapsed
            tracker.target_solution = [int(value) for value in solution]
        return tracker.reached

    current_solution = rng.integers(0, dits, size=n).tolist()
    nonzero_count = sum(value != 0 for value in current_solution)
    if require_nonzero and nonzero_count == 0:
        variable = int(rng.integers(n))
        current_solution[variable] = int(rng.integers(1, dits))
        nonzero_count = 1
    current_energy = _objective_energy(q_matrix, q_row, current_solution)
    target_reached = record_incumbent(current_solution)

    if not target_reached and greedy_initialization and perf_counter() < deadline:
        current_energy, nonzero_count, target_reached = _short_greedy_descent(
            q_matrix,
            q_row,
            current_solution,
            dits,
            neighbors,
            current_energy,
            nonzero_count,
            require_nonzero,
            deadline,
            record_incumbent if tracker is not None else None,
        )

    best_solution = (
        tracker.target_solution.copy()
        if tracker is not None and tracker.target_solution is not None
        else current_solution.copy()
    )
    best_energy = current_energy

    def build_result() -> SolutionClass:
        result = SolutionClass.from_solution_list(
            qudo_instance_matrix=[list(row) for row in q_matrix],
            qudo_instance_row=list(q_row),
            solution_list=best_solution,
            dits=dits,
            execution_time=0.0,
        )
        result.execution_time = perf_counter() - started_at
        return result

    if target_reached or perf_counter() >= deadline or max_iterations == 0:
        return build_result(), tracker

    initialized_cache = _initialize_delta_cache(
        q_matrix, q_row, current_solution, dits, neighbors, deadline
    )
    if initialized_cache is None:
        return build_result(), tracker
    delta_cache, scores = initialized_cache
    tabu_until = np.full((n, dits), -1, dtype=np.int64)
    iteration = 0
    iterations_without_improvement = 0

    while perf_counter() < deadline and (
        max_iterations is None or iteration < max_iterations
    ):
        variables = _candidate_variables(scores, candidate_list_size, rng)
        move, fallback, deadline_reached = _best_admissible_move(
            delta_cache,
            current_solution,
            variables,
            tabu_until,
            iteration,
            current_energy,
            best_energy,
            require_nonzero,
            nonzero_count,
            deadline,
        )
        if deadline_reached:
            break
        if move is None:
            if fallback is None:
                break
            tabu_until.fill(-1)
            move = fallback

        old_value = current_solution[move.variable]
        current_solution[move.variable] = move.new_value
        nonzero_count += (move.new_value != 0) - (old_value != 0)
        current_energy += move.delta

        jitter_bound = max(2, tenure // 4)
        effective_tenure = tenure + int(rng.integers(0, jitter_bound))
        tabu_until[move.variable, old_value] = iteration + effective_tenure

        if current_energy < best_energy:
            best_energy = current_energy
            best_solution = current_solution.copy()
            iterations_without_improvement = 0
            if record_incumbent(best_solution):
                break
        else:
            iterations_without_improvement += 1

        iteration += 1
        if perf_counter() >= deadline:
            break

        affected = {move.variable}
        affected.update(neighbor for neighbor, _ in neighbors[move.variable])
        cache_complete = _refresh_delta_cache(
            q_matrix,
            q_row,
            current_solution,
            dits,
            neighbors,
            sorted(affected),
            delta_cache,
            scores,
            deadline,
        )
        if not cache_complete:
            break

        if iterations_without_improvement >= diversification_interval:
            if perf_counter() >= deadline:
                break
            current_energy, nonzero_count, _ = _diversify(
                q_matrix,
                q_row,
                current_solution,
                dits,
                require_nonzero,
                rng,
            )
            tabu_until.fill(-1)
            if current_energy < best_energy:
                best_energy = current_energy
                best_solution = current_solution.copy()
                if record_incumbent(best_solution):
                    break
            initialized_cache = _initialize_delta_cache(
                q_matrix, q_row, current_solution, dits, neighbors, deadline
            )
            if initialized_cache is None:
                break
            delta_cache, scores = initialized_cache
            iterations_without_improvement = 0

    return build_result(), tracker


def solver_tabu_search(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
    time_limit: float,
    require_nonzero: bool = False,
    tabu_tenure: int | None = None,
    candidate_list_size: int | None = None,
    diversification_interval: int = 500,
    seed: int | None = None,
    greedy_initialization: bool = True,
    max_iterations: int | None = None,
) -> SolutionClass:
    """Minimize a banded QUDO objective within a wall-clock budget."""
    started_at = perf_counter()
    solution, _ = _solve_tabu_search(
        q_matrix,
        q_row,
        dits,
        n_neighbors,
        time_limit,
        require_nonzero,
        tabu_tenure,
        candidate_list_size,
        diversification_interval,
        seed,
        greedy_initialization,
        max_iterations,
        started_at=started_at,
    )
    return solution


def solver_tabu_search_time_to_target(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
    target_cost: float,
    max_time: float = 60.0,
    require_nonzero: bool = False,
    tabu_tenure: int | None = None,
    candidate_list_size: int | None = None,
    diversification_interval: int = 500,
    seed: int | None = None,
    greedy_initialization: bool = True,
    target_tolerance: float = 1e-9,
    max_iterations: int | None = None,
) -> TabuTargetResult:
    """Measure wall time until Tabu Search first reaches a target cost.

    The run stops at the first externally evaluated incumbent whose cost is at
    most ``target_cost + target_tolerance``. If the target is not reached,
    ``time_to_target`` is ``None``; ``max_time`` is never reported as if it
    were an observed target time.
    """
    started_at = perf_counter()
    solution, tracker = _solve_tabu_search(
        q_matrix,
        q_row,
        dits,
        n_neighbors,
        max_time,
        require_nonzero,
        tabu_tenure,
        candidate_list_size,
        diversification_interval,
        seed,
        greedy_initialization,
        max_iterations,
        started_at=started_at,
        target_cost=target_cost,
        target_tolerance=target_tolerance,
    )
    if tracker is None:  # Internal invariant: target_cost always creates it.
        raise RuntimeError("Tabu Search target tracker was not initialized")

    total_execution_time = perf_counter() - started_at
    solution.execution_time = total_execution_time
    return TabuTargetResult(
        reached=tracker.reached,
        target_cost=tracker.target_cost,
        time_to_target=tracker.time_to_target,
        solution=solution,
        best_cost=solution.cost,
        total_execution_time=total_execution_time,
        incumbent_history=list(tracker.history),
    )
