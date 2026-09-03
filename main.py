from time import time

from qudo_solver.data_generator.qudo_problem_generator import qudo_problem_generation
from qudo_solver.solvers.dynamic_programming.beam_dynamic_programming_solver import (
    solver_beam_dynamic_programming,
)
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (
    solver_dynamic_programming,
)
from qudo_solver.solvers.dynamic_programming.vectorized_dynamic_programin import (
    solver_dynamic_programming_vectorized,
)
from qudo_solver.solvers.scip import (
    solver_scip,
)
from qudo_solver.solvers.smvc.smvc import solver_smvc
from qudo_solver.solvers.stc.stc_solver import solver_stc
from qudo_solver.solvers.tabu_search import solver_tabu_search

n = 100
k = 5
d = 5

instancia = qudo_problem_generation(n, k, 1, 0)[0]
Q = instancia["q_matrix"]
q = instancia["q_row"]
# q = [0.0]*n
in1 = time()
resultado = solver_smvc(Q, q, d, k)
matrix_time = time() - in1
print("Matrix:", resultado.cost, "time", matrix_time)

# resultado = solver_tabu_search(
#     Q,
#     q,
#     d,
#     k,
#     time_limit=matrix_time,
#     seed=instancia["seed"],
# )
# print("Tabu Search:", resultado.cost)

# resultado = solver_scip(
#     Q,
#     q,
#     d,
#     k,
#     time_limit=3*matrix_time,
#     seed=instancia["seed"],
# )
# print("SCIP:", resultado.cost)

# resultado = solver_stc(Q, q, None, d, k)
# print("TensorKrowch:", resultado.cost)


resultado = solver_dynamic_programming(Q, q,d, k)
print("Dynamic programming:", resultado.cost, "time", resultado.execution_time)

# resultado = solver_beam_dynamic_programming(Q, q,d, k)
# print("Heuristic:", resultado.cost)


resultado = solver_dynamic_programming_vectorized(Q, q,d, k)
print("Vectorized:", resultado.cost, "time", resultado.execution_time)