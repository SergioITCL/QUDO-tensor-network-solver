"""Vectorized exact dynamic-programming solver for banded QUDO instances.

This module implements the same boundary-state recurrence as
``solver_dynamic_programming`` from the project, but stores the frontier in
NumPy arrays instead of Python dictionaries of tuple states.

For the normal case ``require_nonzero=False`` it is intended as a drop-in
performance baseline for the paper experiments.  The mathematical algorithm is
still exact dynamic programming: only the representation and execution of the
state transitions are changed.

The QUDO band format is the same as in the repository.  Row ``i`` contains the
coefficients coupling ``x_i`` to the previous variables represented in that
row, followed by the diagonal coefficient ``Q_ii``.
"""

from __future__ import annotations

from time import time
from typing import List

import numpy as np

from qudo_solver.qudo_solver_core.solution import SolutionClass
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (
    solver_dynamic_programming as _reference_dynamic_programming,
)


def _validate_inputs(
    q_matrix: List[List[float]],
    q_row: List[float],
    dits: int,
    n_neighbors: int,
) -> None:
    """Validate inputs using the same structural rules as the reference DP."""
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


def _parent_dtype(dits: int) -> np.dtype:
    """Return the smallest unsigned dtype able to store one dit value."""
    maximum_value = dits - 1
    if maximum_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if maximum_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if maximum_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _solve_independent_variables(
    q_matrix: List[List[float]],
    q_row: List[float],
    dits: int,
) -> list[int]:
    """Solve the k=0 case, where every variable is independent."""
    values = np.arange(dits, dtype=np.float64)
    values_squared = values * values
    solution: list[int] = []

    for row, linear_coefficient in zip(q_matrix, q_row):
        local_costs = row[-1] * values_squared + linear_coefficient * values
        solution.append(int(np.argmin(local_costs)))

    return solution


def solver_vectorized_dynamic_programming(
    q_matrix: List[List[float]],
    q_row: List[float],
    dits: int,
    n_neighbors: int,
    require_nonzero: bool = False,
) -> SolutionClass:
    """Solve a banded QUDO exactly using a NumPy boundary-state DP.

    Parameters are intentionally identical to ``solver_dynamic_programming``.

    The frontier after processing a variable is stored as an array with one
    axis per retained boundary variable.  Once the frontier reaches width
    ``k = n_neighbors``, a transition has the form

        (x_{i-k}, ..., x_{i-1}) -> (x_{i-k+1}, ..., x_i)

    and the oldest value is eliminated with ``np.min(axis=0)``.  Only a small
    Python loop over the ``d`` possible values of the new variable remains;
    all ``d**k`` frontier states are processed in compiled NumPy operations.

    Time complexity is O(n * d**(k+1)) for the transition recurrence, plus
    O(n * k * d**k) to form the local interaction fields.  Working frontier
    memory is O(d**k).  Backpointers require O(n * d**k), but each one stores
    only the eliminated dit value, using the smallest suitable unsigned dtype.

    Notes
    -----
    The project-specific ``require_nonzero=True`` option adds an extra global
    state flag.  It is not used in the paper experiments.  To preserve its
    exact semantics without mixing that special constraint into the benchmark
    fast path, this implementation delegates that case to the reference solver.
    """
    initial_time = time()
    _validate_inputs(q_matrix, q_row, dits, n_neighbors)

    if require_nonzero:
        # Keep exact compatibility for this special constraint.  Do not use
        # this branch when benchmarking the vectorized implementation.
        return _reference_dynamic_programming(
            q_matrix=q_matrix,
            q_row=q_row,
            dits=dits,
            n_neighbors=n_neighbors,
            require_nonzero=True,
        )

    n_variables = len(q_matrix)
    k = n_neighbors

    if k == 0:
        solution = _solve_independent_variables(q_matrix, q_row, dits)
        elapsed = time() - initial_time
        return SolutionClass.from_solution_list(
            qudo_instance_matrix=q_matrix,
            qudo_instance_row=q_row,
            solution_list=solution,
            dits=dits,
            execution_time=elapsed,
        )

    values = np.arange(dits, dtype=np.float64)
    values_squared = values * values
    parent_dtype = _parent_dtype(dits)

    # Before x_0 there is a single empty boundary state with cost zero.
    current_costs = np.asarray(0.0, dtype=np.float64)

    # parent_choices[i] is only populated once i >= k.  For a next boundary
    # state it stores the value x_{i-k} that was eliminated at step i.
    parent_choices: list[np.ndarray | None] = [None] * n_variables

    for position, (row, linear_coefficient) in enumerate(zip(q_matrix, q_row)):
        boundary_width = min(position, k)
        interaction_count = len(row) - 1

        # Build
        #     h(history) = sum_j Q_{j,i} x_j
        # over the interactions represented in this row.  The new value x_i
        # then contributes x_i * h(history).
        if interaction_count:
            interaction_field = np.zeros(current_costs.shape, dtype=np.float64)
            first_relevant_axis = boundary_width - interaction_count

            for offset, coefficient in enumerate(row[:-1]):
                axis = first_relevant_axis + offset
                broadcast_shape = [1] * boundary_width
                broadcast_shape[axis] = dits
                previous_values = values.reshape(broadcast_shape)
                interaction_field += float(coefficient) * previous_values
        else:
            interaction_field = 0.0

        diagonal_coefficient = float(row[-1])
        linear_coefficient = float(linear_coefficient)

        if boundary_width < k:
            # During the first k variables there is no elimination yet.  Every
            # extended state has a unique predecessor, so no backpointer is
            # necessary for these prefix steps.
            next_shape = current_costs.shape + (dits,)
            next_costs = np.empty(next_shape, dtype=np.float64)

            for value in range(dits):
                local_cost = (
                    diagonal_coefficient * values_squared[value]
                    + linear_coefficient * values[value]
                    + values[value] * interaction_field
                )
                next_costs[..., value] = current_costs + local_cost

        else:
            # The frontier already contains k variables.  Appending x_i gives
            # k+1 variables temporarily; eliminating the oldest one is exactly
            # the DP minimization over all predecessors of the next state.
            next_costs = np.empty_like(current_costs)
            step_parents = np.empty(current_costs.shape, dtype=parent_dtype)

            for value in range(dits):
                local_cost = (
                    diagonal_coefficient * values_squared[value]
                    + linear_coefficient * values[value]
                    + values[value] * interaction_field
                )
                candidate_costs = current_costs + local_cost

                # axis 0 is x_{i-k}.  The remaining axes plus ``value`` form
                # the next boundary state.
                next_costs[..., value] = np.min(candidate_costs, axis=0)
                step_parents[..., value] = np.argmin(
                    candidate_costs, axis=0
                ).astype(parent_dtype, copy=False)

            parent_choices[position] = step_parents

        current_costs = next_costs

    # np.argmin uses first-occurrence tie breaking.  With C-order state arrays
    # this matches the lexicographic insertion order of the reference DP.
    best_flat_index = int(np.argmin(current_costs))
    best_state = tuple(
        int(value)
        for value in np.unravel_index(best_flat_index, current_costs.shape)
    )

    if n_variables <= k:
        # No variable has ever left the boundary, so the final state already is
        # the complete assignment.
        solution = list(best_state)
    else:
        solution = [0] * n_variables
        solution[n_variables - k :] = best_state

        next_state = best_state
        for position in range(n_variables - 1, k - 1, -1):
            step_parents = parent_choices[position]
            if step_parents is None:
                raise RuntimeError(
                    f"Missing backpointer array at DP position {position}"
                )

            eliminated_value = int(step_parents[next_state])
            solution[position - k] = eliminated_value

            # Previous boundary:
            # (x_{i-k}, x_{i-k+1}, ..., x_{i-1})
            next_state = (eliminated_value,) + next_state[:-1]

    elapsed = time() - initial_time
    return SolutionClass.from_solution_list(
        qudo_instance_matrix=q_matrix,
        qudo_instance_row=q_row,
        solution_list=solution,
        dits=dits,
        execution_time=elapsed,
    )


# Convenience alias if you prefer the noun order used by the existing solver.
solver_dynamic_programming_vectorized = solver_vectorized_dynamic_programming


__all__ = [
    "solver_vectorized_dynamic_programming",
    "solver_dynamic_programming_vectorized",
]
