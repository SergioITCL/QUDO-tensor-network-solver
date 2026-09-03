import pytest

from qudo_solver.auxiliar_functions import qubo_value_from_lists, qudo_value


def test_objective_handles_zero_coefficients_and_compact_lower_triangle():
    matrix = [[0.0], [0.0, 2.0], [-1.5, 0.0, 0.0]]
    row = [0.0, 0.0, 3.0]
    solution = [1, 2, 1]

    # 2*2^2 - 1.5*1*1 + 3*1 = 9.5
    assert qudo_value(solution, matrix, row) == pytest.approx(9.5)
    assert qubo_value_from_lists(solution, matrix) == pytest.approx(6.5)


def test_objective_rejects_inconsistent_dimensions_and_non_integer_values():
    with pytest.raises(ValueError):
        qudo_value([0], [[1.0], [1.0, 1.0]], [0.0, 0.0])
    with pytest.raises(TypeError):
        qubo_value_from_lists([0.5], [[1.0]])
