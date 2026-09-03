import pytest

from qudo_solver.data_generator.qudo_problem_generator import (
    generate_k_random_qudo,
    normalize_problem,
    qudo_problem_generation,
)


def test_random_generation_is_reproducible_and_has_compact_rows():
    first = qudo_problem_generation(5, 2, 1, 0, random_seeds=[23])[0]
    second = qudo_problem_generation(5, 2, 1, 0, random_seeds=[23])[0]

    assert first == second
    assert [len(row) for row in first["q_matrix"]] == [1, 2, 3, 3, 3]
    assert len(first["q_row"]) == 5


def test_generator_rejects_invalid_instance_counts_and_seed_lengths():
    with pytest.raises(ValueError):
        qudo_problem_generation(5, 1, 0, 0)
    with pytest.raises(ValueError):
        qudo_problem_generation(5, 1, 1, 0, random_seeds=[1, 2])
    with pytest.raises(ValueError):
        generate_k_random_qudo(0, 1)


def test_normalization_preserves_zero_coefficients():
    matrix, row = normalize_problem([[0.0], [0.0, 2.0]], [0.0, 0.0])

    assert matrix[0] == [0.0]
    assert matrix[1][0] == 0.0
    assert row == [0.0, 0.0]


def test_normalization_rejects_an_entirely_zero_problem():
    with pytest.raises(ValueError, match="zero-norm"):
        normalize_problem([[0.0], [0.0, 0.0]], [0.0, 0.0])
