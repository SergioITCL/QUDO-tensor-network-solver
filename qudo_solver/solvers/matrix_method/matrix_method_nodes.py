from itertools import product

import numpy as np
from scipy.sparse import coo_matrix, csr_array, dia_matrix, diags


def node_0(
    Q_matrix_0: float, 
    Q_row_0: float,
    dits: int, 
    tau: float
    ) -> dia_matrix:
    """
    Matrix that represents the row of tensors of the first variable.

    Args:
        Q_matrix_0 (float): Q_matrix[0][0].
        dits (int): dinary description: bits, trits,...
        tau (float): parameter of the imaginary time evolution.

    Returns:
        tensor (csr_matrix): tensor of the first node
    """
    diagonal_values = np.exp(-tau * (Q_matrix_0 * np.arange(dits)**2 + Q_row_0*np.arange(dits)))
    tensor = diags(diagonal_values, offsets=0, format='csr')
    
    return tensor

def node_grow(
    Q_matrix_row: list[float], 
    Q_row_row: float,
    dits: int, 
    n_neight: int, 
    tau: float
    ) -> np.ndarray:
    """
    Template that generates the matrix that represents the row of tensors of the variables with fewer neighbors than k from above.

    Args:
        Q_matrix_row (np.array): Q_matrix[row]
        dits (int): dinary description: bits, trits,...
        n_neight (_type_): maximum number of neighbors of the problem.
        tau (float): parameter of the imaginary time evolution.

    Returns:
        tensor (np.ndarray): Matrix of a variable with fewer neighbors above than below.
    """
    tensor = np.zeros((dits**n_neight, dits**(n_neight+1)))

    combinations_up = product(range(dits), repeat=n_neight)
    
    for element in combinations_up:
        index_up = sum(dits**aux * element[aux] for aux in range(n_neight))
        
        for index_last in range(dits):
            index_down = index_up + dits**n_neight * index_last
            full_element = list(element) + [index_last]
            
            # Calculate the tensor value
            value = 1*np.exp(-tau * Q_row_row * full_element[-1])
            for aux in range(len(full_element)):
                value *= np.exp(-tau * Q_matrix_row[aux] * full_element[-1] * full_element[aux])

            # Assign the computed value to the tensor
            tensor[index_up, index_down] = value
    
    return tensor

def node_intermediate(
    Q_matrix_row: list[float], 
    Q_row_row: float,
    dits: int, 
    n_neigh: int, 
    tau: float
    ) -> csr_array:
    """
   Function that builds a sparse matrix (CSR) of shape (dits**n_neigh, dits**n_neigh) fully vectorized that represents the row of tensors of the variables with equal number of neighbors above and below.

    Parameters:
    Q_matrix_row (np.ndarray): 1D array with length n_neigh + 1; last entry is the quadratic coefficient.
    dits (int): dinary description: bits, trits, ...
    n_neigh (int) : Number of neighbor variables considered in the problem
    tau (float): parameter of the imaginary time evolution.

    Returns:
    scipy.sparse.csr_matrix
        The sparse matrix in CSR format.
    """
    size = dits ** n_neigh
    n_blocks = dits ** (n_neigh - 1)
    digits = np.array(np.unravel_index(np.arange(size), (dits,) * n_neigh)).T  

    Q_lin = Q_matrix_row[:n_neigh][::-1]             
    S = digits @ Q_lin                                

    k = np.arange(dits)                              
    Q_last = Q_matrix_row[-1]
  
    exponents = (S[:, None] * k[None, :]) + (Q_last * (k ** 2))[None, :] + Q_row_row * k[None, :]
    data_mat = np.exp(-tau * exponents)            

    rows = np.repeat(np.arange(size), dits)          
    base = (np.arange(size) // dits)                
    cols = (base[:, None] + k[None, :] * n_blocks).ravel(order="C")

    data = data_mat.ravel(order="C")
    A = coo_matrix((data, (rows, cols)), shape=(size, size)).tocsr()
    return A  

def last_tensor(
    Q_matrix_row: list[float], 
    Q_row_row: float,
    dits: int, 
    tau:float
    ) -> np.ndarray:
    """
        Function that generates the vector that represents the row of nodes of the last variable.

    Args:
        Q_matrix_row (np.array): Row of the Q matrix.
        dits (int): Dinary description (e.g., bits, trits, etc.).
        num_neight_updown (int): Number of neighbors both up and down.
        tau (float): Parameter for the imaginary time evolution.

    Returns:
    """
    n_neighbors = len(Q_matrix_row) - 1
    tensor_size = dits**n_neighbors
    tensor = np.zeros(tensor_size)
    
    # Generate all combinations of dits^n_neighbors
    combinations_up = product(range(dits), repeat=n_neighbors)
    
    for element in combinations_up:
        index_up = sum(dits**aux * element[aux] for aux in range(n_neighbors))
        
        for index_last in range(dits):
            full_element = list(element) + [index_last]
            # Compute the tensor value
            tensor_aux = 1.0 * np.exp(-tau * Q_row_row * full_element[-1])
            for el in range(len(full_element)):
                tensor_aux *= np.exp(-tau * Q_matrix_row[el] * full_element[el] * full_element[-1])
            tensor[index_up] += tensor_aux
    
    return tensor



def new_initial_tensor(
    Q_matrix_row, 
    Q_row_row,
    dits: int, 
    size_2, 
    solution, 
    n_neigh, 
    tau: float, 
    last_solution
    ):
    """
    Function that generates the new initial tensor that has information of the already known solution and allows to take advantange of intermediate calculations.

    Args:
        Q_matrix_row (np.array): Row of the Q matrix.
        dits (int): Dinary description (e.g., bits, trits, etc.).
        size_2 (int): The second dimension of the tensor.
        solution (list or tuple): Current solution configuration.
        n_neigh (int): Number of neighbors in the problem.
        tau (float): Parameter for the imaginary time evolution.
        last_solution (int): The value of the last solution element.

    Returns:
        np.array: The constructed initial tensor.
    """
    size_1 = dits
    tensor = np.zeros((size_1, size_2))

    n = len(solution) + 1
    solution = tuple(solution)

    # Calculate index_down based on the current solution
    index_down = sum(dits**aux * solution[aux] for aux in range(len(solution)))

    # Generate all combinations for the current dimension
    combinations_up = product(range(dits), repeat=n)

    if len(Q_matrix_row) == 2 + len(solution):
        # Case where Q_matrix_row includes last_solution
        for element in combinations_up:
            if element[:-1] == solution:
                index_down_aux = index_down + dits**(n - 1) * element[-1]
                tensor[element[-1], index_down_aux] = np.exp(-tau * (Q_matrix_row[0] * last_solution * element[-1] + Q_row_row * element[-1]))
                # Ensure the indexing is valid by checking that el+1 is within range
                for el in range(len(element)):
                    if el + 1 < len(Q_matrix_row):  # Ensure we're not out of bounds
                        tensor[element[-1], index_down_aux] *= np.exp(-tau * Q_matrix_row[el + 1] * element[el] * element[-1])
    else:
        # Case without last_solution in Q_matrix_row
        for element in combinations_up:
            if element[:-1] == solution:
                index_down_aux = index_down + dits**(n - 1) * element[-1]
                tensor[element[-1], index_down_aux] = np.exp(
                    -tau * Q_row_row * element[-1]
                )
                
                # Ensure the indexing is valid by checking that el is within range
                for el in range(len(element)):
                    if el < len(Q_matrix_row):  # Ensure we're not out of bounds
                        tensor[element[-1], index_down_aux] *= np.exp(-tau * Q_matrix_row[el] * element[el] * element[-1])
    return tensor