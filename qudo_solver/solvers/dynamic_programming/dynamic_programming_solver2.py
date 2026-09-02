from collections import namedtuple
from typing import List
from time import time

from qudo_solver.qudo_solver_core.solution import SolutionClass



def solver_dynamic_programming2(
    q_matrix: List[List[float]],
    q_row: List[float],
    dits: int,
    n_neighbors: int,
) -> SolutionClass:
    """
    Solves a k-neighbor QUBO problem via exact dynamic programming.

    Each variable takes values in {0, ..., dits - 1} and the assignment must
    satisfy sum(x) >= 1. The band structure limits interactions to the last
    n_neighbors variables, so the DP state tracks only that window.

    Args:
        Q_matrix: Band/triangular QUBO representation.
        q_row: Linear coefficient of each variable.
        dits: Number of discrete values per variable.
        n_neighbors: Neighborhood width (k) of the problem.

    Returns:
        SolutionClass with the optimal solution and its cost.
    """
    initial_time = time()
    n = len(q_matrix)
    if len(q_row) != n:
        raise ValueError("q_matrix y q_row deben tener la misma longitud")

    Candidate = namedtuple('Candidate', ['route', 'cost'])
    current_candidates = [
        Candidate([dit], q_matrix[0][0] * dit**2 + q_row[0] * dit)
        for dit in range(dits)
    ]

    for digit in range(1, n):
        new_candidates = []
        for dit in range(dits):
      
            posible_dit_candidates = []
            index = 0
            for candidate in current_candidates:
                possible_configuration = candidate.route + [dit]
                window = possible_configuration[-len(q_matrix[digit]):]
                step_cost = sum(q_ij * window[k] * window[-1] for k, q_ij in enumerate(q_matrix[digit]))
                step_cost += q_row[digit] * dit
                cost = candidate.cost + step_cost
                new_candidate = Candidate(possible_configuration, cost)

                posible_dit_candidates.append(new_candidate)
                index += 1

                if digit < n_neighbors:
                    new_candidates.extend(posible_dit_candidates)
                    posible_dit_candidates = []
                else:
                    if index == dits:
                        new_candidates.append(min(posible_dit_candidates, key=lambda c: c.cost))
                        posible_dit_candidates = []
                        index = 0


        current_candidates = new_candidates

    candidate = min(current_candidates, key=lambda c: c.cost)
    return SolutionClass.from_solution_list(
        qudo_instance_matrix=q_matrix,
        qudo_instance_row=q_row,
        solution_list=candidate.route,
        dits=dits,
        execution_time=time()-initial_time
        )
