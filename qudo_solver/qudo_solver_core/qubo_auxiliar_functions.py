import numpy as np
from scipy.sparse import lil_matrix
import random
from collections.abc import Iterable
from numbers import Integral


def qudo_evaluation(Q_matrix, x):
    """
    Function that evaluates the cost of a given solution of a QUBO problem.
    Args:
        Q_matrix (_type_): QUBO matrix representation of the problem.
        x (_type_): solution to check the value.

    Returns:
        _type_: cost of the solution.
    """
    x = np.array(x)
    return np.dot(x, np.dot(Q_matrix, x))


def generate_1k_qubo(n_variables: int, seed=None):
    if seed is not None:
        random.seed(seed)  # Set the global seed

    Q_list = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(n_variables)]
    Q_list[0] = random.uniform(-1, 1)  # first term is Q[0,0] -> scalar

    return Q_list

def generate_k_qubo(n_variables: int, k_neighbour: int, seed=None):
    if seed is not None:
        random.seed(seed)  

    Q_list = [[] for _ in range(n_variables)]
    n_elements_per_row = 1
    for _ in range(n_variables):
        for element in range(n_elements_per_row):
            Q_list[_].append(random.uniform(-1, 1))
        
        if n_elements_per_row <= k_neighbour:
            n_elements_per_row += 1
 
    return Q_list

def normalize_list_of_lists(Q_matrix):
    # Flatten all values, even if some "rows" are float
    all_values = []

    for row in Q_matrix:
        if isinstance(row, Iterable) and not isinstance(row, (str, bytes)):
            all_values.extend(row)
        else:
            all_values.append(row)

    norm = np.linalg.norm(all_values)

    # Normalize respecting the original shape
    Q_normalized = []
    for row in Q_matrix:
        if isinstance(row, Iterable) and not isinstance(row, (str, bytes)):
            Q_normalized.append([elem / norm for elem in row])
        else:
            Q_normalized.append(row / norm)

    return Q_normalized


def qubo_list_to_matrix(lists, fill=0.0):
    n = len(lists)
    M = np.full((n, n), fill, dtype=float)
    for i, row in enumerate(lists):
        k = len(row)
        j_end = i                     # diagonal column
        j_start = j_end - k + 1       # start to align to the right

        # Adjustment if j_start < 0 (row shorter than diagonal position)
        s = max(0, j_start)
        # Choose the correct section from the end of the sublist
        section = row[-(j_end + 1 - s):]
        M[i, s:j_end + 1] = section
    return M
    

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


