import math

import pytest

from qudo_solver.data_generator.qudo_problem_generator import (
    qudo_problem_generation,
)
from qudo_solver.solvers.dynamic_programming.beam_dynamic_programming_solver import (
    solver_beam_dynamic_programming,
)
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (
    solver_dynamic_programming,
)
from qudo_solver.solvers.smvc.smvc import solver_smvc
from qudo_solver.solvers.tabu_search import solver_tabu_search


def test_smvc_solves_a_single_variable_problem():
    result = solver_smvc([[0.0]], [-2.0], dits=3, n_neighbors=1)

    assert result.solution_list == [2]
    assert result.cost == pytest.approx(-4.0)


def test_exact_solvers_agree_on_a_small_problem_with_zero_coefficients():
    q_matrix = [[0.0], [0.0, 1.0], [-2.0, 0.0, 0.0], [0.0, 0.5, 0.0]]
    q_row = [0.0, 0.0, 0.0, -1.0]

    dynamic = solver_dynamic_programming(q_matrix, q_row, 3, 2)
    smvc = solver_smvc(q_matrix, q_row, 3, 2)
    beam = solver_beam_dynamic_programming(
        q_matrix, q_row, 3, 2, beam_width=100, lookahead_depth=0,
        local_search_passes=0,
    )

    assert smvc.cost == pytest.approx(dynamic.cost)
    assert beam.cost == pytest.approx(dynamic.cost)
    assert all(0 <= value < 3 for value in smvc.solution_list)


@pytest.mark.parametrize(
    "solver",
    [solver_dynamic_programming, solver_beam_dynamic_programming, solver_smvc],
)
def test_solvers_reject_mismatched_dimensions(solver):
    with pytest.raises(ValueError):
        solver([[1.0]], [], dits=2, n_neighbors=1)


def test_tabu_search_returns_a_feasible_solution_for_n_one():
    result = solver_tabu_search(
        [[0.0]], [-2.0], dits=3, n_neighbors=1, time_limit=0.1, seed=0
    )

    assert len(result.solution_list) == 1
    assert 0 <= result.solution_list[0] < 3
    assert math.isfinite(result.cost)
