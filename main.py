from time import time

from qudo_solver.data_generator.qudo_problem_generator import qudo_problem_generation
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (
    solver_dynamic_programming,
)
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver2 import (
    solver_dynamic_programming2,
)
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver3 import (
    solver_dynamic_programming3,
)
from qudo_solver.solvers.dynamic_programming.heuristic_dynamic_programming_solver import (
    solver_dynamic_programming_heuristic,
)
from qudo_solver.solvers.matrix_method.matrix_method_solver import solver_matrix_method
from qudo_solver.solvers.ortools.ortools_method_solver import solver_ortools
from qudo_solver.solvers.scip import (
    solver_scip,
    solver_scip_time_to_target,
    solver_scip_with_metadata,
)
from qudo_solver.solvers.sum_product import solver_sum_product
from qudo_solver.solvers.tabu_search import solver_tabu_search
from qudo_solver.solvers.tensorkrowch_tn.tensorkrowch_solver import solver_tensorkrowch

n = 100
k = 5
d = 5

instancia = qudo_problem_generation(n, k, 1, 0)[0]
Q = instancia["q_matrix"]
q = instancia["q_row"]
# q = [0.0]*n
in1 = time()
resultado = solver_matrix_method(Q, q, d, k)
matrix_time = time() - in1
print("Matriz:", resultado.cost, "time", matrix_time)

resultado = solver_tabu_search(
    Q,
    q,
    d,
    k,
    time_limit=matrix_time,
    seed=instancia["seed"],
)
print("Tabu Search:", resultado.cost)

resultado = solver_scip(
    Q,
    q,
    d,
    k,
    time_limit=3*matrix_time,
    seed=instancia["seed"],
)
print("SCIP:", resultado.cost)

resultado = solver_tensorkrowch(Q, q, None, d, k)
print("TensorKrowch:", resultado.cost)

resultado = solver_dynamic_programming(Q, q, d, k)
print("Dinámica 1:", resultado.cost)

resultado = solver_dynamic_programming2(Q, q,d, k)
print("Dinámica 2:", resultado.cost)

resultado = solver_dynamic_programming3(Q, q,d, k)
print("Dinámica 3:", resultado.cost)

resultado = solver_dynamic_programming_heuristic(Q, q,d, k)
print("Heurística:", resultado.cost)

resultado = solver_sum_product(Q, q, d, k)
print("Min-sum:", resultado.cost)

resultado = solver_ortools(Q, q, d, 10)
print("OR-Tools:", resultado.cost)

