
from ortools.sat.python import cp_model

def ortools_qudo_solver(Q, num_values, time):
    model = cp_model.CpModel()
    n = len(Q)
    x = [model.NewIntVar(0, num_values - 1, f'x_{i}') for i in range(n)]
    model.Add(sum(x) >= 1)  
    products = {}
    for i in range(n):
        for j in range(n):
            if Q[i][j] != 0:
                var_name = f'prod_{i}_{j}'
                products[(i, j)] = model.NewIntVar(0, (num_values - 1) ** 2, var_name)
                model.AddMultiplicationEquality(products[(i, j)], x[i], x[j])  
    
    objective_terms = [Q[i][j] * products[(i, j)] for i in range(n) for j in range(n) if (i, j) in products]
    model.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return [solver.Value(x[i]) for i in range(n)]
    else:
        return None