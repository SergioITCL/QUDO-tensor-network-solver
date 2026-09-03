import numpy as np
import tensorkrowch as tk
import torch


def node_last_superposition(dits, row, column) -> tk.Node:
    """ Template to generate tensor network nodes of one qudit
    Args:
        dits (int): dinary description: bits, trits,...
        row (int): label that determines the row of the node.
        column (int): label that determines the column of the node.
    Returns:
        node (tk.Node): node
    """
    aux_array  = [1]*dits
    node = tk.Node(tensor = torch.tensor(aux_array, dtype=torch.float64), name = f'A_({row},{column})', axes_names = ['left'])
    return node

def node_initial(dits: int, Q_element, Q_row_element, tau: float, row:int, column:int) -> tk.Node:
    """ Template that generates the first layer of the tensor network.

    Args:
        dits (int): dinary description: bits, trits,...
        Q_element (float): Q_matrix element of the node.
        row (int): label that determines the row of the node.
        column (int): label that determines the column of the node.
        tau (float): parameter of the imaginary time evolution.

    Returns:
        node (tk.Node): node
    """
    tensor = torch.zeros((dits), dtype=torch.float64)
    for dit in range(dits):
        tensor[dit] = np.exp(-tau * (Q_element * dit**2 + Q_row_element * dit))
    node = tk.Node(tensor = tensor, name = f'A_({row},{column})', axes_names = ['right'])
    return node

def node_control(dits: int, row: int, column:int) -> tk.Node:
    """ Template that generates the control nodes of the tensor network.

    Args:
        dits (int): dinary description: bits, trits,...
        row (int): label that determines the row of the node.
        column (int): label that determines the column of the node.

    Returns:
        node (tk.Node): node
    """
    tensor = torch.zeros((dits, dits, dits), dtype = torch.float64)
    for dit in range(dits):
        tensor[dit, dit, dit]= 1
    node = tk.Node(tensor = tensor, name = f'A_({row},{column})', axes_names = ['left','right','down'])
    return node

def node_intermediate(dits: int, Q_element:float, tau:float, row:int, column:int) -> tk.Node:
    """ Template that generates the intermediate tensors of the tensor network.

    Args:
        dits (int): dinary description: bits, trits,...
        Q_element (float): Weight element Q_element[row][column]
        tau (float): parameter of the imaginary time evolution.
        row (int): label that determines the row of the node.
        column (int): label that determines the column of the node.

    Returns:
        node (tk.Node): node
    """

    tensor = torch.zeros((dits, dits, dits, dits), dtype = torch.float64)
    for up in range(dits):
        for left in range(dits):
            down= up
            right = left
            if up * left != 0:
                tensor[left, right, up, down] = np.exp(-tau * (Q_element * up * left))
            else:
                tensor[left, right, up, down] = 1
    node = tk.Node(tensor = tensor, name = f'A_({row},{column})', axes_names=['left','right','up','down'])
    return node

def node_final(dits:int, Q_element:float, tau:float, row,  column) -> tk.Node:
    """ Template that generates the last row tensors.

    Args:
        dits (int): dinary description: bits, trits,...
        Q_element (float): Weight element Q_element[-1][column].
        row (int): label that determines the row of the node.
        column (_type_): label that determines the column of the node.


    Returns:
        node (tk.Node): node
    """
    tensor = torch.zeros((dits, dits, dits), dtype = torch.float64)
    for up in range(dits):
        for left in range(dits):
            right=left
            if up * left != 0:
                tensor[left, right,up] = np.exp(-tau * (Q_element * up * left))
            else:
                tensor[left, right,up] = 1

    node = tk.Node(tensor=tensor, name = f'A_({row},{column})',axes_names=['left','right','up'])

    return node

def new_initial_tensor_(Q_matrix_row: list[float], Q_row_row: float, sol_aux, dits: int, tau: float) -> tk.Node:
    """Function that generates the vector tensor of the solution
    Args:
        tn (tk.TensorNetwork): tensorkrowch object that represents the tensor network. 
        dits (int): dinary description: bits, trits, ...
        solution (int): value of the solution.

    Returns:
        tk.Node: node
    """
    tensor = torch.zeros((dits, dits), dtype = torch.float64)
    for down in range(dits):
        tensor[down, down] = np.exp(-tau * (Q_matrix_row[-1] * down**2 + Q_row_row * down))
        for index, sol in enumerate(sol_aux):
            tensor[down, down] *=  np.exp(-tau * Q_matrix_row[index] * sol * down)
    node = tk.Node(tensor = tensor, name = 'A_extra', axes_names = ['right', 'down'])
    return node