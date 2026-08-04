from typing import List, Tuple
from time import time

import numpy as np
import torch
import tensorkrowch as tk

from qudo_solver.auxiliar_functions import qubo_value_from_lists
from qudo_solver.qudo_solver_core.solution import SolutionClass
from qudo_solver.solvers.tensorkrowch_tn.tensorkrowch_nodes import new_initial_tensor_, node_control, node_final, node_initial, node_intermediate, node_last_superposition

def solver_tensorkrowch(
    Q_matrix: List[List[float]], 
    tau: float, 
    dits: int, 
    n_neighbors: int
    ) -> SolutionClass:
    """
    Solves a QUBO (Quadratic Unconstrained Binary Optimization) problem using tensor network contraction.

    Args:
        Q_matrix (np.array): The Q matrix representing the QUBO problem.
        tau (float): The parameter for imaginary time evolution.
        dits (int): The number of digits (e.g., bits, trits, etc.).
        n_neighbors (int): The number of neighbors in the problem.

    Returns:
        SolutionClass: The solution in SolutionClass format.
    """
    initial_time = time()
    # Initialize variables and create a copy of the Q matrix
    n_variables = len(Q_matrix)
    solution = np.zeros(n_variables, dtype=int)

    # Generation the tensor network
    tensor_network_matrix = tensor_network_generator(Q_matrix, dits, n_neighbors, tau)
    # Contraction of the tensor network
    result, intermediate_tensors = tensor_network_contraction(tensor_network_matrix)

    result_tensor = result.tensor
    assert result_tensor is not None
    solution[0] = np.argmax(result_tensor.abs())
    
    # Calculation of each solution employing intermediate calculations
    for node in range(1, n_variables -1):
        intermediate_tensor = intermediate_tensors[1].tensor
        assert intermediate_tensor is not None
        if node < n_neighbors:
            sol_aux = solution[max(0, node - n_neighbors - 1):node]
            tensor_ = intermediate_tensor[tuple(sol_aux)].flatten()
        else:
            sol_aux = solution[node - n_neighbors:node ]
            tensor_ = intermediate_tensor[tuple(sol_aux[1:])].flatten()

        tensor = tk.Node(tensor = tensor_, name = 'aux_tensor', axes_names = ['up']) 
        new_tensor = new_initial_tensor_(Q_matrix[node], sol_aux, dits, tau)
        tensor['up'] ^ new_tensor['down']
        contracted_tensor = tk.contract_between_(tensor, new_tensor).tensor
        assert contracted_tensor is not None
        solution[node] = np.argmax(contracted_tensor)
        intermediate_tensors.pop(0)

    # Last digit calculation
    cost = qubo_value_from_lists(solution, Q_matrix)
    solution_2 = solution.copy()
    for dit in range(1, dits):
        solution_2[-1] = dit
        cost_2 = qubo_value_from_lists(solution_2, Q_matrix)
        # If a better solution is found, update the solution and cost
        if cost_2 < cost:
            solution[-1] = dit
            cost = cost_2
    return SolutionClass.from_solution_list(
        qudo_instance_list=Q_matrix,
        solution_list=list(solution),
        dits=dits,
        execution_time=time() - initial_time
    )

def tensor_network_generator(
    Q_matrix: List[List[float]], 
    dits: int, 
    n_neighbors: int, 
    tau: float
    ) -> list[list[tk.Node]]:
    """
    Generates the tensor network for a given Q matrix and parameters.

    Args:
        Q_matrix (np.array): The Q matrix representing the problem.
        dits (int): Dinary description (e.g., bits, trits, etc.).
        n_neighbors (int): Number of neighbors to consider.
        tau (float): Parameter for the imaginary time evolution.

    Returns:
        tensor_network_matrix: A list of tensors representing the tensor network.
    """
    n_variables = len(Q_matrix)
    tensor_network_matrix = [[] for _ in range(n_variables)]
    
    for row in range(n_variables):
        last_node_activation = False
        initial_node = node_initial(dits, Q_matrix[row][-1], tau, row, 0)
        tensor_network_matrix[row].append(initial_node)

        if row >= n_neighbors and row != (n_variables - 1):
            last_node = node_final(dits, Q_matrix[row][0], tau, row, 1)
            tensor_network_matrix[row].append(last_node)
            last_node_activation = True

        for column, Q_row_column in enumerate(Q_matrix[row][:-1]):

            if last_node_activation == True and column == 0:
                continue
            elif row == n_variables - 1:
                last_node = node_final(dits, Q_matrix[row][column], tau, row, column + 1)
                tensor_network_matrix[row].append(last_node)
                
            else:
                intermediate_node = node_intermediate(dits, Q_row_column, tau, row, column + 1)
                tensor_network_matrix[row].append(intermediate_node)

        if row != n_variables - 1:
            control_node = node_control(dits, row, len(tensor_network_matrix[row]))
            tensor_network_matrix[row].append(control_node)

        if row != 0:
            superposition_node = node_last_superposition(dits, row, len(tensor_network_matrix[row]))
            tensor_network_matrix[row].append(superposition_node)
     
    return tensor_network_matrix

def tensor_network_contraction(
    tensor_list: List[List[tk.Node]]
    ) -> Tuple[tk.Node, List[tk.Node]]:
    """
    This function is designed to contract a tensor network starting from the bottom, while preserving the intermediate tensors during the process. 
    The contraction is performed layer by layer, specifically from right to left within each layer.

    Args:
        tensor_list (list): A list of tensors representing the network.

    Returns:
        result (np.ndarray): An array that represents the whole contraction of the tensor network.
        intermediate_tensors (list): A list that includes all the intermediate tensors computed during the contraction.
    """ 
    n_variables = len(tensor_list)
    intermediate_tensors = []

    
    # Tensor network horizontal tensorkrowch conexion
    for row in range(n_variables):
        for column in range(len(tensor_list[row]) - 1):
            tensor_list[row][column]['right'] ^ tensor_list[row][column + 1]['left']

    # Tensor network vertical tensorkrowch conexion
    for row in range(n_variables - 1, 0, -1):
        up_index = None
        for index, first_up_tensor in enumerate(reversed(tensor_list[row])):
            if 'up' in first_up_tensor.axes_names:
                up_index = len(tensor_list[row]) - index - 1
                break

        down_index = None
        for index, first_down_tensor in enumerate(reversed(tensor_list[row - 1])):
            if 'down' in first_down_tensor.axes_names:
                down_index = len(tensor_list[row - 1]) - index - 1
                break
            
        if up_index is not None and down_index is not None:
            while (down_index >= 0 and up_index >= 0 and 'up' in tensor_list[row][up_index].axes_names and 'down' in tensor_list[row - 1][down_index].axes_names):
                tensor_list[row][up_index]['up'] ^ tensor_list[row - 1][down_index]['down']
                up_index -= 1
                down_index -= 1

    # Tensor network tensorkrowch contraction
    # Last row contraction
    result = tensor_list[-1][-1]
    for column in range(len(tensor_list[-1]) - 1,0, -1):
        result = tk.contract_between(tensor_list[-1][column - 1],result) 
    intermediate_tensors.append(result)

    # Intermediate layer contraction
    for row in range(n_variables -2, 0, -1):
        result = tk.contract_between(result, tensor_list[row][-2])
        
        for column in range(len(tensor_list[row]) - 3, -1, -1):
            result = tk.contract_between(tensor_list[row][column], result)
        result = tk.div(result, torch.norm(result.tensor))
        result = tk.contract_between(result, tensor_list[row][-1])
        intermediate_tensors.append(result)

    # First row
    # first layer contraction
    result_row = tk.contract_between(tensor_list[0][0], tensor_list[0][1])
    result = tk.div(result, torch.norm(result.tensor))
    result = tk.contract_between(result_row, result)
    intermediate_tensors.reverse()

    return result, intermediate_tensors