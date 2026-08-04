from typing import List
import numpy as np
from time import time

from qudo_solver.auxiliar_functions import estimate_tau_max, qubo_value_from_lists
from qudo_solver.data_generator.qudo_problem_generator import normalize_list_of_lists
from qudo_solver.qudo_solver_core.solution import SolutionClass
from qudo_solver.solvers.matrix_method.matrix_method_nodes import last_tensor, new_initial_tensor, node_0, node_grow, node_intermediate

def solver_matrix_method(
    Q_list: List[List[float]], 
    dits: int, 
    n_neighbors: int,
    tau: float | None = None, 
    ) -> SolutionClass:
    """
    Solves a QUBO (Quadratic Unconstrained Binary Optimization) problem using tensor network contraction.

    Args:
        Q_matrix (np.array): The Q matrix representing the QUBO problem.
        tau (float): The parameter for imaginary time evolution.
        dits (int): The number of digits (e.g., bits, trits, etc.).
        n_neighbors (int): The number of neighbors in the problem.

    Returns:
        np.array: The solution vector to the QUBO problem.
    """
    
    initial_time = time()
    # Initialize variables and create a copy of the Q matrix
    Q_matrix = normalize_list_of_lists(Q_list)
    # Q_matrix = Q_list
    n_variables = len(Q_matrix)
    solution = np.zeros(n_variables, dtype=int)

    if tau is None:
        tau = estimate_tau_max(
            n_variables=n_variables,
            dits=dits,
            n_neighbors=n_neighbors,
        )
    # Generate the tensor network
  
    tensor_network = tensor_network_generator(Q_matrix, dits, n_neighbors, tau)

    # Perform the tensor network contraction

    result_contraction, intermediate_tensors = tensor_network_contraction(tensor_network)
 
    # Set the first solution based on the contraction result
    solution[0] = np.argmax(abs(result_contraction))

    # Iterate over the remaining nodes to solve the QUBO problem
    for node in range(1, n_variables - 1):
        if node < n_neighbors:
            sol_aux = solution[max(0, node - n_neighbors - 1):node]
        else:
            sol_aux = solution[node - n_neighbors + 1:node]

        new_tensor = new_initial_tensor(Q_matrix[node], dits, intermediate_tensors[2].shape[0], sol_aux, n_neighbors, tau, solution[node - n_neighbors])
        solution[node] = np.argmax(abs(new_tensor @ intermediate_tensors[2]))
        intermediate_tensors.pop(0)
   
    # Iterate over all possible solutions for the last digit
    cost = qubo_value_from_lists(solution, Q_matrix)
    solution2 = solution.copy()
    for dit in range(1, dits):
        solution2[-1] = dit
        cost2 = qubo_value_from_lists(solution2, Q_matrix)
        
        # If a better solution is found, update the solution and cost
        if cost2 < cost:
            solution[-1] = dit
            cost = cost2

    return SolutionClass.from_solution_list(
        qudo_instance_list=Q_list,
        solution_list=list(solution),
        dits=dits,
        execution_time=time()-initial_time
    )

def tensor_network_generator(
    Q_matrix: List[List[float]], 
    dits: int, 
    n_neighbors: int, 
    tau: float):
    """
    Generates the tensor network for a given Q matrix and the parameters.

    Args:
        Q_matrix (np.array): The Q matrix representing the problem.
        dits (int): Dinary description (e.g., bits, trits, etc.).
        n_neighbors (int): Number of neighbors to consider.
        tau (float): Parameter for the imaginary time evolution.

    Returns:
        list: A list of tensors representing the tensor network.
    """
    n_variables = len(Q_matrix)
    intermediate_tensors = []

    # Generate the first node
    tensor = node_0(Q_matrix[0][0], dits, tau)

    intermediate_tensors.append(tensor)

    # Generate the intermediate nodes
    for variable in range(1, n_variables - 1):
        if variable < n_neighbors:
            tensor = node_grow(Q_matrix[variable], dits, variable, tau)
            
        else:  
            tensor = node_intermediate(Q_matrix[variable], dits, n_neighbors, tau)
        intermediate_tensors.append(tensor)

    # Generate the last tensor
    tensor = last_tensor(Q_matrix[-1], dits, tau)
    intermediate_tensors.append(tensor)

    return intermediate_tensors

def tensor_network_contraction(tensor_list: list):
    """
    Performs the contraction of a tensor network by multiplying tensors sequentially.

    Args:
        tensor_list (list): A list of tensors representing the network.

    Returns:
        tuple: The final contracted tensor and a list of intermediate tensors.
    """
    # Initialize with the last tensor in the network
    tensor = tensor_list[-1]
    intermediate_tensors = [tensor]

    # Contract the tensors in reverse order
    for current_tensor in reversed(tensor_list[:-1]):
        
        tensor = current_tensor @ tensor  # Matrix multiplication
        tensor /= np.linalg.norm(tensor)  # Normalize the tensor after multiplication
        intermediate_tensors.append(tensor)

    # Reverse the list of intermediate tensors to maintain the order of contraction
    intermediate_tensors.reverse()

    return tensor, intermediate_tensors