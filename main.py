
from qudo_solver.auxiliar_functions import estimate_tau_max
from qudo_solver.data_generator.qudo_problem_generator import generate_k_qubo
from qudo_solver.solvers.dynamic_programming.heuristic_dynamic_programming_solver import solver_dynamic_programming_heuristic
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import solver_dynamic_programming
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver2 import solver_dynamic_programming2
from qudo_solver.solvers.matrix_method.matrix_method_solver import solver_matrix_method
from qudo_solver.solvers.ortools.ortools_method_solver import solver_ortools
from qudo_solver.solvers.tensorkrowch_tn.tensorkrowch_solver import solver_tensorkrowch



def main():
    pass
if __name__ == "__main__":
    n_variables = 1000
    n_neighbors = 5
    seed = 455
    tau = 60
    dits = 6

    qubo_problem_list = generate_k_qubo(
        n_variables=n_variables,
        k_neighbor=n_neighbors,
        seed=seed
    )

    tau = estimate_tau_max(
        q_matrix=qubo_problem_list,
        dits=dits,
        n_neighbors=n_neighbors,
    )

    # print("tau", tau)
 
    solution_t = solver_matrix_method(
        Q_list=qubo_problem_list,
        tau=tau,
        dits=dits,
        n_neighbors=n_neighbors)
   

    print("Matrix method")
    print(solution_t.cost, "tiempo", solution_t.execution_time)
    solution_t = solver_tensorkrowch(
        Q_matrix=qubo_problem_list,
        tau=tau,
        dits=dits,
        n_neighbors=n_neighbors)
    print("Tensorkrowch method")
    print(solution_t.cost, solution_t.execution_time)
    solution_orto = solver_ortools(
        Q_matrix=qubo_problem_list,
        dits=dits,
        max_time=solution_t.execution_time*2
        )
    print("Ortools method")
    print(solution_orto.cost, "Tiempo", solution_orto.execution_time)

    solution_d = solver_dynamic_programming(
        Q_matrix=qubo_problem_list,
        dits=dits,
        n_neighbors=n_neighbors
        )
    print("Dynamic programming method")
    print(solution_d.cost, "tiempo", solution_d.execution_time)


    solution_h = solver_dynamic_programming_heuristic(
        q_matrix=qubo_problem_list,
        dits=dits,
        n_neighbors=n_neighbors,
        beam_width=256,
        lookahead_depth=3,
        local_search_passes=3,
        )
    print("Dynamic programming heuristic method")
    print(solution_h.cost, "tiempo", solution_h.execution_time)
    

    solution_d = solver_dynamic_programming2(
        q_matrix=qubo_problem_list,
        dits=dits,
        n_neighbors=n_neighbors
        )
    print("Dynamic programming method manual")
    print(solution_d.cost, "tiempo", solution_d.execution_time)

    main()