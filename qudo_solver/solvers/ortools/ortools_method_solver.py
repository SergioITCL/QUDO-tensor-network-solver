"""OR-Tools CP-SAT solver for compact triangular QUDO lists."""

from time import time
from typing import Sequence

from ortools.sat.python import cp_model
from qudo_solver.qudo_solver_core.solution import SolutionClass


def solver_ortools(
    Q_list: Sequence[Sequence[float]],
    Q_row: Sequence[float],
    dits: int,
    max_time: float,
    require_nonzero: bool = False,
) -> SolutionClass | None:
    """Solve a compact banded QUDO instance with OR-Tools.

    Row i stores coefficients for columns from
    i - len(Q_list[i]) + 1 through i. Therefore, the final coefficient
    of each row is its diagonal term. Q_row stores the linear coefficients.
    """
    if dits < 2:
        raise ValueError("dits must be at least 2")
    if max_time <= 0:
        raise ValueError("max_time must be positive")

    initial_time = time()
    n_variables = len(Q_list)
    if len(Q_row) != n_variables:
        raise ValueError("Q_list y Q_row deben tener la misma longitud")

    model = cp_model.CpModel()
    x = [model.NewIntVar(0, dits - 1, f"x_{i}") for i in range(n_variables)]
    if require_nonzero:
        model.Add(sum(x) >= 1)

    objective_terms = []
    for i, coefficient in enumerate(Q_row):
        if coefficient != 0:
            objective_terms.append(float(coefficient) * x[i])

    for i, row in enumerate(Q_list):
        if not row:
            raise ValueError(f"Q_list[{i}] cannot be empty")
        first_column = i - len(row) + 1
        if first_column < 0:
            raise ValueError(
                f"Q_list[{i}] has length {len(row)}; it may contain at most i + 1 coefficients"
            )

        for offset, coefficient in enumerate(row):
            if coefficient == 0:
                continue
            j = first_column + offset
            product = model.NewIntVar(0, (dits - 1) ** 2, f"product_{i}_{j}")
            model.AddMultiplicationEquality(product, [x[i], x[j]])
            objective_terms.append(float(coefficient) * product)

    model.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    solution_list = [solver.Value(variable) for variable in x]
    return SolutionClass.from_solution_list(
        qudo_instance_matrix=[list(row) for row in Q_list],
        qudo_instance_row=list(Q_row),
        solution_list=solution_list,
        dits=dits,
        execution_time=time() - initial_time,
    )
