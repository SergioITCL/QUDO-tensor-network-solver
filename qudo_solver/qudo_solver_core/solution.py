from numbers import Integral
from typing import List
from pydantic import BaseModel

class SolutionClass(BaseModel):
    solution_list: List[int]
    dits: int
    cost: float
    execution_time: float

    @classmethod
    def from_solution_list(cls, 
        qudo_instance_list: List[List[float]],
        solution_list: List[int],
        dits: int,
        execution_time: float):
        return cls(
            solution_list=solution_list,
            dits=dits,
            cost=qubo_value_from_lists(solution_list, qudo_instance_list),
            execution_time=execution_time)


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