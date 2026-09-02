"""Exact min-sum message passing for banded QUDO problems.

The usual sum-product algorithm computes probabilities and marginals.  To solve
the optimization problem, this module uses the same factor-graph message
passing in the min-sum (tropical) semiring.  On the resulting clique chain,
message passing is exact and returns the global MAP/minimum-energy assignment.
"""

from time import time
from typing import Dict, List, Optional, Tuple

from qudo_solver.qudo_solver_core.solution import SolutionClass


History = Tuple[int, ...]
MessageKey = Tuple[History, bool]
Backpointer = Tuple[MessageKey, int]


def _factor_energy(
    row: List[float],
    linear_coefficient: float,
    history: History,
    value: int,
) -> float:
    """Evaluate the local factor associated with the current matrix row."""
    number_of_interactions = len(row) - 1
    previous_values = (
        history[-number_of_interactions:] if number_of_interactions else ()
    )
    energy = row[-1] * value * value + linear_coefficient * value
    energy += sum(
        coefficient * previous_value * value
        for coefficient, previous_value in zip(row[:-1], previous_values)
    )
    return float(energy)


def _validate_problem(
    q_matrix: List[List[float]],
    q_row: List[float],
    dits: int,
    n_neighbors: int,
) -> None:
    if not q_matrix:
        raise ValueError("q_matrix must contain at least one variable")
    if len(q_row) != len(q_matrix):
        raise ValueError("q_matrix y q_row deben tener la misma longitud")
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
                f"interactions, but n_neighbors={n_neighbors} can represent "
                f"at most {maximum_row_length - 1} at that position"
            )


def solver_sum_product(
    q_matrix: List[List[float]],
    q_row: List[float],
    dits: int,
    n_neighbors: int,
    require_nonzero: bool = False,
) -> SolutionClass:
    """Return an exact minimum-energy solution using min-sum messages.

    Each forward message is indexed by the separator between two consecutive
    factors: the values of the last ``n_neighbors`` variables.  A Boolean is
    included when ``require_nonzero`` is true so the final assignment can be
    constrained by ``sum(x) >= 1``.  Backpointers reconstruct the joint MAP
    assignment after the final message.

    Args:
        q_matrix: Banded lower-triangular QUDO representation.  Row ``i`` holds
            coefficients for consecutive columns ending at the diagonal ``i``.
        q_row: Linear coefficient of each variable.
        dits: Domain size; every variable is in ``{0, ..., dits - 1}``.
        n_neighbors: Maximum number of preceding interacting variables.
        require_nonzero: Enforce ``sum(x) >= 1`` when true.

    Complexity:
        With ``n`` variables, domain size ``d`` and width ``k``, there are at
        most ``2*d**k`` message entries and ``d`` outgoing values per entry.
        This implementation takes ``O(n*k*d**(k+1))`` time because evaluating
        and hashing a transition touches up to ``k`` values.  Working-message
        memory is ``O(d**k)``; retained backpointers use ``O(n*d**k)`` memory.
    """
    initial_time = time()
    _validate_problem(q_matrix, q_row, dits, n_neighbors)

    # In the min-sum semiring, an incoming message value is a partial energy.
    messages: Dict[MessageKey, float] = {((), False): 0.0}
    traceback: List[Dict[MessageKey, Backpointer]] = []

    for row, linear_coefficient in zip(q_matrix, q_row):
        outgoing_messages: Dict[MessageKey, float] = {}
        outgoing_traceback: Dict[MessageKey, Backpointer] = {}

        for source_key, incoming_energy in messages.items():
            history, has_nonzero = source_key

            for value in range(dits):
                extended_history = history + (value,)
                destination_history = (
                    extended_history[-n_neighbors:] if n_neighbors else ()
                )
                destination_key: MessageKey = (
                    destination_history,
                    (has_nonzero or value != 0) if require_nonzero else False,
                )
                candidate_energy = incoming_energy + _factor_energy(
                    row,
                    linear_coefficient,
                    history,
                    value,
                )

                # Semiring addition: retain the minimum incoming product/path.
                if (
                    destination_key not in outgoing_messages
                    or candidate_energy < outgoing_messages[destination_key]
                ):
                    outgoing_messages[destination_key] = candidate_energy
                    outgoing_traceback[destination_key] = (source_key, value)

        messages = outgoing_messages
        traceback.append(outgoing_traceback)

    best_key: Optional[MessageKey] = None
    best_energy = float("inf")
    for key, energy in messages.items():
        is_feasible = key[1] or not require_nonzero
        if is_feasible and energy < best_energy:
            best_key = key
            best_energy = energy

    if best_key is None:
        raise RuntimeError("No feasible solution found (sum(x) >= 1).")

    solution = [0] * len(q_matrix)
    key = best_key
    for position in range(len(q_matrix) - 1, -1, -1):
        key, solution[position] = traceback[position][key]

    return SolutionClass.from_solution_list(
        qudo_instance_matrix=q_matrix,
        qudo_instance_row=q_row,
        solution_list=solution,
        dits=dits,
        execution_time=time() - initial_time,
    )
