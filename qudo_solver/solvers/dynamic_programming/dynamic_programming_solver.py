from time import time
from typing import Dict, List, Optional, Tuple

from qudo_solver.qudo_solver_core.solution import SolutionClass


def _row_cost(
    Q_matrix: List[List[float]],
    Q_row: List[float],
    i: int,
    x: List[int],
) -> float:
    """Energy contribution of row i given the assignment x[0..i]."""
    row_vals = Q_matrix[i]
    j_start = i - len(row_vals) + 1
    x_i = x[i]
    quadratic_cost = sum(
        q_ij * x[j] * x_i
        for k, q_ij in enumerate(row_vals)
        if (j := j_start + k) >= 0
    )
    linear_cost = Q_row[i] * x_i
    return quadratic_cost + linear_cost


def solver_dynamic_programming(
    Q_matrix: List[List[float]],
    Q_row: List[float],
    dits: int,
    n_neighbors: int,
    require_nonzero: bool = False,
) -> SolutionClass:
    """
    Solves a k-neighbor QUBO problem via exact dynamic programming.

    Each variable takes values in {0, ..., dits - 1}. The band structure limits
    interactions to the last n_neighbors variables, so the DP state tracks only
    that window. Set ``require_nonzero=True`` to exclude the all-zero solution.

    Args:
        Q_matrix: Band/triangular QUBO representation.
        Q_row: Linear coefficient of each variable.
        dits: Number of discrete values per variable.
        n_neighbors: Neighborhood width (k) of the problem.

    Returns:
        SolutionClass with the optimal solution and its cost.
    """
    initial_time = time()
    n = len(Q_matrix)

    if len(Q_row) != n:
        raise ValueError("Q_matrix y Q_row deben tener la misma longitud")

    max_hist = n_neighbors

    State = Tuple[int, Tuple[int, ...], bool]
    dp: Dict[State, float] = {(0, (), False): 0.0}
    parent: Dict[State, Tuple[State, int]] = {}

    for pos in range(n):
        next_dp: Dict[State, float] = {}

        for (cur_pos, hist, has_nonzero), cost_so_far in dp.items():
            if cur_pos != pos:
                continue

            prefix_start = pos - len(hist)
            x_prefix = list(hist)

            for val in range(dits):
                x = [0] * (pos + 1)
                for idx, var_idx in enumerate(range(prefix_start, pos)):
                    x[var_idx] = x_prefix[idx]
                x[pos] = val

                step_cost = _row_cost(Q_matrix, Q_row, pos, x)
                new_cost = cost_so_far + step_cost
                new_has_nonzero = has_nonzero or val != 0
                new_hist = tuple((hist + (val,))[-max_hist:])
                state_key: State = (pos + 1, new_hist, new_has_nonzero)

                if state_key not in next_dp or new_cost < next_dp[state_key]:
                    next_dp[state_key] = new_cost
                    parent[state_key] = ((pos, hist, has_nonzero), val)

        dp = next_dp

    best_state: Optional[State] = None
    best_cost = float("inf")

    for (cur_pos, hist, has_nonzero), cost in dp.items():
        if cur_pos == n and (has_nonzero or not require_nonzero) and cost < best_cost:
            best_cost = cost
            best_state = (cur_pos, hist, has_nonzero)

    if best_state is None:
        raise RuntimeError("No feasible solution found.")

    solution = [0] * n
    state: Optional[State] = best_state

    while state is not None and state[0] > 0:
        prev_state, val = parent[state]
        solution[state[0] - 1] = val
        state = prev_state

    return SolutionClass.from_solution_list(
        qudo_instance_matrix=Q_matrix,
        qudo_instance_row=Q_row,
        solution_list=solution,
        dits=dits,
        execution_time=time()-initial_time
    )
