from dataclasses import dataclass
from typing import Dict, List, Tuple
from time import time

from qudo_solver.qudo_solver_core.solution import SolutionClass


@dataclass(frozen=True)
class _Candidate:
    route: Tuple[int, ...]
    hist: Tuple[int, ...]
    cost: float
    has_nonzero: bool


def _trim_hist(hist: Tuple[int, ...], max_hist: int) -> Tuple[int, ...]:
    if max_hist <= 0:
        return ()
    return hist[-max_hist:]


def _step_cost(
    q_matrix: List[List[float]],
    pos: int,
    hist: Tuple[int, ...],
    value: int,
) -> float:
    row = q_matrix[pos]
    hist_start = pos - len(hist)
    j_start = pos - len(row) + 1
    total = 0.0

    for offset, q_ij in enumerate(row):
        j = j_start + offset
        if j == pos:
            total += q_ij * value * value
        elif hist_start <= j < pos:
            total += q_ij * hist[j - hist_start] * value

    return float(total)


def _greedy_lookahead_cost(
    q_matrix: List[List[float]],
    pos: int,
    hist: Tuple[int, ...],
    dits: int,
    max_hist: int,
    depth: int,
) -> float:
    """Cheap state-dependent estimate used only to rank the beam."""
    estimate = 0.0
    lookahead_hist = hist
    n = len(q_matrix)

    for next_pos in range(pos, min(n, pos + depth)):
        best_value = 0
        best_cost = float("inf")

        for value in range(dits):
            cost = _step_cost(q_matrix, next_pos, lookahead_hist, value)
            if cost < best_cost:
                best_cost = cost
                best_value = value

        estimate += best_cost
        lookahead_hist = _trim_hist(lookahead_hist + (best_value,), max_hist)

    return estimate


def _variable_contribution(
    q_matrix: List[List[float]],
    solution: List[int],
    index: int,
    value: int,
) -> float:
    total = 0.0
    own_row = q_matrix[index]
    own_j_start = index - len(own_row) + 1

    for offset, q_ij in enumerate(own_row):
        j = own_j_start + offset
        if j == index:
            total += q_ij * value * value
        elif 0 <= j < index:
            total += q_ij * value * solution[j]

    for row_index in range(index + 1, len(q_matrix)):
        row = q_matrix[row_index]
        j_start = row_index - len(row) + 1

        if not j_start <= index <= row_index:
            continue

        q_ij = row[index - j_start]
        total += q_ij * solution[row_index] * value

    return float(total)


def _repair_zero_solution(
    q_matrix: List[List[float]],
    solution: List[int],
    dits: int,
) -> List[int]:
    if any(solution):
        return solution

    best_solution = solution[:]
    best_cost = float("inf")

    for index in range(len(solution)):
        for value in range(1, dits):
            trial = solution[:]
            trial[index] = value
            cost = _variable_contribution(q_matrix, trial, index, value)
            if cost < best_cost:
                best_cost = cost
                best_solution = trial

    return best_solution


def _local_search(
    q_matrix: List[List[float]],
    solution: List[int],
    dits: int,
    passes: int,
) -> List[int]:
    solution = _repair_zero_solution(q_matrix, solution, dits)

    for _ in range(passes):
        improved = False

        for index, old_value in enumerate(solution):
            old_contribution = _variable_contribution(q_matrix, solution, index, old_value)
            best_value = old_value
            best_delta = 0.0

            for value in range(dits):
                if value == old_value:
                    continue
                if value == 0 and old_value != 0 and sum(solution) == old_value:
                    continue

                new_contribution = _variable_contribution(q_matrix, solution, index, value)
                delta = new_contribution - old_contribution

                if delta < best_delta:
                    best_delta = delta
                    best_value = value

            if best_value != old_value:
                solution[index] = best_value
                improved = True

        if not improved:
            break

    return solution


def solver_dynamic_programming_heuristic(
    q_matrix: List[List[float]],
    dits: int,
    n_neighbors: int,
    beam_width: int = 256,
    lookahead_depth: int = 2,
    local_search_passes: int = 2,
) -> SolutionClass:
    """
    Fast approximate solver for k-neighbor QUDO/QUBO instances.

    It keeps the dynamic-programming state over the last variables that can still
    interact with future rows, but trims the frontier with a beam-search
    heuristic instead of keeping every possible state. A small coordinate search
    pass then improves the resulting assignment.
    """
    initial_time = time()
    if dits < 2:
        raise ValueError("dits must be at least 2")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if lookahead_depth < 0:
        raise ValueError("lookahead_depth must be non-negative")
    if local_search_passes < 0:
        raise ValueError("local_search_passes must be non-negative")

    n = len(q_matrix)
    if n == 0:
        raise ValueError("q_matrix must contain at least one variable")

    interaction_width = max(len(row) - 1 for row in q_matrix)
    max_hist = min(n - 1, max(n_neighbors, interaction_width))
    candidates = [_Candidate(route=(), hist=(), cost=0.0, has_nonzero=False)]

    for pos in range(n):
        best_by_state: Dict[Tuple[Tuple[int, ...], bool], _Candidate] = {}

        for candidate in candidates:
            for value in range(dits):
                new_hist = _trim_hist(candidate.hist + (value,), max_hist)
                new_candidate = _Candidate(
                    route=candidate.route + (value,),
                    hist=new_hist,
                    cost=candidate.cost + _step_cost(q_matrix, pos, candidate.hist, value),
                    has_nonzero=candidate.has_nonzero or value != 0,
                )
                state = (new_hist, new_candidate.has_nonzero)
                current_best = best_by_state.get(state)

                if current_best is None or new_candidate.cost < current_best.cost:
                    best_by_state[state] = new_candidate

        ranked_candidates = sorted(
            best_by_state.values(),
            key=lambda candidate: candidate.cost
            + _greedy_lookahead_cost(
                q_matrix=q_matrix,
                pos=pos + 1,
                hist=candidate.hist,
                dits=dits,
                max_hist=max_hist,
                depth=lookahead_depth,
            ),
        )
        candidates = ranked_candidates[:beam_width]

        if candidates and not any(candidate.has_nonzero for candidate in candidates):
            first_nonzero = next(
                (candidate for candidate in ranked_candidates if candidate.has_nonzero),
                None,
            )
            if first_nonzero is not None:
                candidates[-1] = first_nonzero

    feasible_candidates = [candidate for candidate in candidates if candidate.has_nonzero]
    if not feasible_candidates:
        solution = _repair_zero_solution(q_matrix, list(candidates[0].route), dits)
    else:
        solution = list(min(feasible_candidates, key=lambda candidate: candidate.cost).route)

    solution = _local_search(
        q_matrix=q_matrix,
        solution=solution,
        dits=dits,
        passes=local_search_passes,
    )

    return SolutionClass.from_solution_list(
        qudo_instance_list=q_matrix,
        solution_list=solution,
        dits=dits,
        execution_time=time()-initial_time
    )
