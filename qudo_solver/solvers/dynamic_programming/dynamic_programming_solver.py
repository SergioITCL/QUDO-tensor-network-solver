from time import time
from typing import Dict, List, Optional, Tuple

from qudo_solver.qudo_solver_core.solution import SolutionClass

History = Tuple[int, ...]
State = Tuple[History, bool]


def _step_cost(
    row: List[float],
    linear_coefficient: float,
    history: History,
    value: int,
) -> float:
    """Return the contribution of one band-matrix row."""
    interaction_count = len(row) - 1
    relevant_history = history[-interaction_count:] if interaction_count else ()

    total = row[-1] * value * value + linear_coefficient * value
    total += sum(
        coefficient * previous_value * value
        for coefficient, previous_value in zip(row[:-1], relevant_history)
    )
    return float(total)


def solver_dynamic_programming(
    q_matrix: List[List[float]],
    q_row: List[float],
    dits: int,
    n_neighbors: int,
    require_nonzero: bool = False,
) -> SolutionClass:
    """Solve a banded QUDO exactly without copying complete candidate routes.

    The state contains only the last ``n_neighbors`` values and a flag recording
    whether the partial assignment contains a nonzero value. The flag is used
    only when ``require_nonzero=True``; by default the all-zero assignment is
    feasible. Backpointers are stored separately for reconstruction, so
    extending a state never copies its length-``n`` route.

    For ``n`` variables, ``d`` dits and neighborhood width ``k``, the running
    time is O(n * k * d**(k + 1)) and the reconstruction storage is
    O(n * d**k).  In particular, for fixed ``d`` and ``k`` the running time is
    linear, rather than quadratic, in ``n``.
    """
    initial_time = time()

    if not q_matrix:
        raise ValueError("q_matrix must contain at least one variable")
    if len(q_row) != len(q_matrix):
        raise ValueError("q_matrix and q_row must have the same length")
    if dits < 2:
        raise ValueError("dits must be at least 2")
    if n_neighbors < 0:
        raise ValueError("n_neighbors must be non-negative")

    for position, row in enumerate(q_matrix):
        if not row:
            raise ValueError(f"q_matrix[{position}] must not be empty")

        maximum_row_length = min(position, n_neighbors) + 1
        if len(row) > maximum_row_length:
            raise ValueError(
                f"q_matrix[{position}] has {len(row) - 1} previous-variable "
                f"interactions, but at most {maximum_row_length - 1} are "
                "representable with n_neighbors"
            )

    current_costs: Dict[State, float] = {((), False): 0.0}
    parents: List[Dict[State, Tuple[State, int]]] = []

    for row, linear_coefficient in zip(q_matrix, q_row):
        next_costs: Dict[State, float] = {}
        next_parents: Dict[State, Tuple[State, int]] = {}

        for state, cost_so_far in current_costs.items():
            history, has_nonzero = state

            for value in range(dits):
                if n_neighbors:
                    new_history = (history + (value,))[-n_neighbors:]
                else:
                    new_history = ()

                new_state: State = (
                    new_history,
                    has_nonzero or value != 0,
                )
                new_cost = cost_so_far + _step_cost(
                    row, linear_coefficient, history, value
                )

                if new_state not in next_costs or new_cost < next_costs[new_state]:
                    next_costs[new_state] = new_cost
                    next_parents[new_state] = (state, value)

        current_costs = next_costs
        parents.append(next_parents)

    best_state: Optional[State] = None
    best_cost = float("inf")

    for state, cost in current_costs.items():
        if (state[1] or not require_nonzero) and cost < best_cost:
            best_state = state
            best_cost = cost

    if best_state is None:
        raise RuntimeError("No feasible solution found.")

    solution = [0] * len(q_matrix)
    state = best_state

    for position in range(len(q_matrix) - 1, -1, -1):
        previous_state, value = parents[position][state]
        solution[position] = value
        state = previous_state

    return SolutionClass.from_solution_list(
        qudo_instance_matrix=q_matrix,
        qudo_instance_row=q_row,
        solution_list=solution,
        dits=dits,
        execution_time=time() - initial_time,
    )
