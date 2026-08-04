from pathlib import Path
from time import perf_counter

import io
import sys
from contextlib import redirect_stdout

from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver2 import solver_dynamic_programming2
from qudo_solver.solvers.ortools import ortools_method_solver


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experimentation.experimentation_general_functions import generate_paper_pdf_plot  # noqa: E402
from qudo_solver.auxiliar_functions import estimate_tau_max  # noqa: E402
from qudo_solver.data_generator.qudo_problem_generator import generate_k_qubo  # noqa: E402
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver import (  # noqa: E402
    solver_dynamic_programming,
)
from qudo_solver.solvers.matrix_method.matrix_method_solver import solver_matrix_method  # noqa: E402


N_VARIABLES = list(range(10, 10001, 1000))
N_NEIGHBORS = 2
DITS = 2
SEED = 42
OUTPUT_PATH = PROJECT_ROOT / "experimentation" / "results" / "solver_times.png"


def time_dynamic_programming(qubo_problem):
    initial_time = perf_counter()
    solver_dynamic_programming(
        Q_matrix=qubo_problem,
        dits=DITS,
        n_neighbors=N_NEIGHBORS,
    )
    return perf_counter() - initial_time

def time_dynamic_programming2(qubo_problem):
    initial_time = perf_counter()
    solver_dynamic_programming2(
        q_matrix=qubo_problem,
        dits=DITS,
        n_neighbors=N_NEIGHBORS,
    )
    return perf_counter() - initial_time


def main():
    dynamic_programming_times = []
    matrix_method_times = []
    ortools_method_times = []
    for n_variables in N_VARIABLES:
        qubo_problem = generate_k_qubo(
            n_variables=n_variables,
            k_neighbor=N_NEIGHBORS,
            seed=SEED + n_variables,
        )

        dynamic_programming_solution = solver_dynamic_programming2(
            q_matrix=qubo_problem,
            dits=DITS,
            n_neighbors=N_NEIGHBORS,
        )
   
        matrix_method_solution = solver_matrix_method(
            Q_list=qubo_problem,
            dits=DITS,
            n_neighbors=N_NEIGHBORS,
        )

        ortools_solution = ortools_method_solver.solver_ortools(
            Q_matrix=qubo_problem,
            dits=DITS,
            max_time=max(matrix_method_solution.execution_time*2, 1)
            )    
        dynamic_programming_times.append(dynamic_programming_solution.execution_time)
        matrix_method_times.append(matrix_method_solution.execution_time)
        ortools_method_times.append(ortools_solution.execution_time)
        print("\\")
        print(f"Prueba con n: {n_variables}")
        print(
            f"dynamic_programming={dynamic_programming_solution.execution_time:.6f} s,  "
            f"cost={dynamic_programming_solution.cost:.5f}  "


        )
        print(
            f"matrix_method={matrix_method_solution.execution_time:.6f} s   "
            f"cost={matrix_method_solution.cost:.5f}"  ,

        )
        print(
            f"ortools={ortools_solution.execution_time:.6f} s,  "
            f"cost={ortools_solution.cost:.5f}  "


        )
    generate_paper_pdf_plot(
        x_values=N_VARIABLES,
        y_values={
            "Dynamic programming": dynamic_programming_times,
            "Matrix method": matrix_method_times,
         #   "Ortools": ortools_method_times
        },
        output_path=OUTPUT_PATH,
        x_label="Number of variables",
        y_label="Time (s)",
        title="Solver runtime comparison",
    )

    print(f"Plot saved in: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
