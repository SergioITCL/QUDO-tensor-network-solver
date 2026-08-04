from typing import List, Optional
from time import time

from ortools.sat.python import cp_model

from qudo_solver.qudo_solver_core.qubo_auxiliar_functions import qubo_list_to_matrix
from qudo_solver.qudo_solver_core.solution import SolutionClass


def solver_ortools(
    Q_matrix: List[List[float]],
    dits: int,
    max_time: float,
) -> Optional[SolutionClass]:
    initial_time = time()
    Q = qubo_list_to_matrix(Q_matrix)
    n = len(Q_matrix)

    model = cp_model.CpModel()
    x = [model.NewIntVar(0, dits - 1, f"x_{i}") for i in range(n)]
    model.Add(sum(x) >= 1)
    products = {}
    for i in range(n):
        for j in range(n):
            if Q[i, j] != 0:
                var_name = f"prod_{i}_{j}"
                products[(i, j)] = model.NewIntVar(0, (dits - 1) ** 2, var_name)
                model.AddMultiplicationEquality(products[(i, j)], x[i], x[j])

    objective_terms = [
        Q[i, j] * products[(i, j)]
        for i in range(n)
        for j in range(n)
        if (i, j) in products
    ]
    model.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    status = solver.Solve(model)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        solution_list = [solver.Value(x[i]) for i in range(n)]
        return SolutionClass.from_solution_list(
            qudo_instance_list=Q_matrix,
            solution_list=solution_list,
            dits=dits,
            execution_time=time()-initial_time
        )
    return None
