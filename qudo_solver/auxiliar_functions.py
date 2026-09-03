from numbers import Integral

import numpy as np


def qubo_value_from_lists(x, lists_tri):
    """
    Evaluates E(x) = sum_{(i,j) present in lists_tri} Q[i,j] * x[i] * x[j]
    using directly the band/triangular list format.

    Parameters:
        x          : iterable of integers (..., -2, -1, 0, 1, 2, ...) of length n
        lists_tri  : list of lists; row i contains the values of columns
                     j from (i - len(row) + 1) to i, both inclusive.

    Returns:
        float: value of the QUBO energy.
    """
    n = len(lists_tri)
    if len(x) != n:
        raise ValueError(f"len(x)={len(x)} must match n={n}")

    # Validate that x contains integers (includes numpy.int*, bool counts as int)
    for idx, v in enumerate(x):
        if not isinstance(v, Integral):
            raise TypeError(f"x[{idx}]={v} is not an integer (type {type(v)})")

    total = 0.0
    for i, row_vals in enumerate(lists_tri):
        j_start = i - len(row_vals) + 1
        for k, q_ij in enumerate(row_vals):
            j = j_start + k
            if 0 <= j < n:
                total += q_ij * int(x[i]) * int(x[j])
    return float(total)

def qudo_value(
    x: list[int],
    q_matrix: list[list[float]],
    q_row: list[float],
) -> float:
    if len(x) != len(q_matrix) or len(x) != len(q_row):
        raise ValueError(
            "x, q_matrix, and q_row must have the same length"
        )

    total = 0.0

    for i, row in enumerate(q_matrix):
        j_start = i - len(row) + 1

        for offset, coefficient in enumerate(row):
            j = j_start + offset
            total += coefficient * x[i] * x[j]

        total += q_row[i] * x[i]

    return float(total)

def estimate_tau_max(
    n_variables: int,
    dits: int,
    n_neighbors: int,
) -> float:
    """
    Estimates a numerically stable tau including quadratic and linear terms.
    """
    if n_variables <= 0:
        raise ValueError("n_variables must be positive")
    if dits < 2:
        raise ValueError("dits must be at least 2")
    if n_neighbors < 0:
        raise ValueError("n_neighbors must be non-negative")

    effective_neighbors = min(n_neighbors, n_variables - 1)
    max_terms_per_row = effective_neighbors + 1

    n_quadratic_coefficients = sum(
        min(row + 1, max_terms_per_row)
        for row in range(n_variables)
    )

    n_coefficients = n_quadratic_coefficients + n_variables

    estimated_coefficient_abs = np.sqrt(3.0 / n_coefficients)

    max_dit = dits - 1
    max_dit_product = max_dit**2

    max_local_energy = estimated_coefficient_abs * (
        max_terms_per_row * max_dit_product
        + max_dit
    )

    target_exponent = 300.0
    return float(target_exponent / max_local_energy)
