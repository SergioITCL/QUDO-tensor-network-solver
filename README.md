# QUDO Solver

QUDO Solver provides algorithms and utilities to solve QUDO/QUBO (Quadratic Unconstrained Discrete/Binary Optimization) problems using tensor networks and other approaches. It includes two optimized implementations of the MeLoCoTN method for fixed k-neighbors: one matrix-based (`numpy/scipy`) and one tensor-based using `tensorkrowch`.

- Author: Sergio Muñiz Subiñas (<sergio.muniz@itcl.es>)
- Organization: ITCL
- License: (add if applicable)

## Features
- Solve QUDO/QUBO with discrete values of base `d` (bits, trits, ...).
- Two k-neighbors solver implementations:
  - Matrix-based with `numpy`/`scipy` (`qubo_k_neighbors_matrix_optimized.py`).
  - Tensor-based with `tensorkrowch`/`torch` (`qubo_k_neighbors_tensorkrowch_optimized.py`).
- Auxiliary functions to generate/evaluate instances in triangular-list format and convert them to dense matrices (`qudo_solver_core/qubo_auxiliar_functions.py`).
- Alternative reference solver using `OR-Tools` in `qudo_solver_core/qubo_solvers.py` (`ortools_qudo_solver`).

## Requirements
- Python 3.10+
- Recommended: virtual environment via `poetry`.

Main dependencies (see `pyproject.toml` for exact versions):
- `numpy`, `scipy`, `matplotlib`
- `torch`, `tensorkrowch`
- `dimod`, `dwave-neal`, `dwave-system` (optional, for the D-Wave ecosystem)
- `ortools`

## Installation

### Option 1: Poetry (recommended)
```bash
# Install Poetry (Windows PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# From the project root
poetry install

# Activate the virtualenv
poetry shell
```

### Option 2: pip + venv
```bash
# Create and activate venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# If you have Poetry available, export deps to a requirements file
poetry export -f requirements.txt --without-hashes -o requirements.txt
pip install -r requirements.txt

# Otherwise, install manually based on pyproject.toml
```

## Project structure
```
qudo_solver/
  ├─ qubo_k_neighbors_matrix_optimized.py
  ├─ qubo_k_neighbors_tensorkrowch_optimized.py
  ├─ experimentation.ipynb
  ├─ qubo-k-neighbors-matrix_optimized.ipynb
  ├─ qubo-k-neighbors-tensorkrowch_optimized.ipynb
  └─ qudo_solver_core/
      ├─ qubo_auxiliar_functions.py
      └─ qubo_solvers.py
```

## QUBO/QUDO input format
For k-neighbors problems, a "right-aligned triangular list" format (`lists_tri`) is used where:
- Row `i` contains the coefficients of columns `j` from `i - len(row) + 1` to `i` (inclusive).
- The quadratic term `Q[i, i]` is the last element of row `i`.

You can convert this format to a dense matrix with `qubo_list_to_matrix`.

Relevant functions in `qudo_solver_core/qubo_auxiliar_functions.py`:
- `qudo_evaluation(Q_matrix, x)` — evaluates the cost `x^T Q x`.
- `generate_1k_qubo(n_variables, seed=None)` — generates a 1-neighbor instance.
- `generate_k_qubo(n_variables, k_neighbour, seed=None)` — generates an instance with up to `k` neighbors.
- `normalize_list_of_lists(Q_matrix)` — normalizes while preserving shape.
- `qubo_list_to_matrix(lists, fill=0.0)` — converts triangular lists to a dense matrix.
- `qubo_value_from_lists(x, lists_tri)` — evaluates the energy directly on the triangular format.

## Quick start
Below are examples to solve an instance with both approaches.

### Matrix solver (NumPy/Scipy)
```python
from qudo_solver.qudo_solver_core.qubo_auxiliar_functions import generate_k_qubo, normalize_list_of_lists
from qudo_solver.qubo_k_neighbors_matrix_optimized import qubo_solver_matrix

# Problem parameters
n_variables = 10
k_neighbors = 3
d = 2              # bits (2), trits (3), etc.
tau = 0.5

# Generate and normalize instance
Q_lists = generate_k_qubo(n_variables, k_neighbors, seed=123)
Q_lists = normalize_list_of_lists(Q_lists)

# Solve
solution = qubo_solver_matrix(Q_lists, tau=tau, dits=d, n_neighbors=k_neighbors)
print("Solution:", solution)
```

### Tensorkrowch solver (Torch)
```python
from qudo_solver.qudo_solver_core.qubo_auxiliar_functions import generate_k_qubo, normalize_list_of_lists
from qudo_solver.qubo_k_neighbors_tensorkrowch_optimized import qubo_solver_tensor

n_variables = 10
k_neighbors = 3
d = 2
tau = 0.5

Q_lists = generate_k_qubo(n_variables, k_neighbors, seed=123)
Q_lists = normalize_list_of_lists(Q_lists)

solution = qubo_solver_tensor(Q_lists, tau=tau, dits=d, n_neighbors=k_neighbors)
print("Solution:", solution)
```

### Reference solver with OR-Tools
`ortools_qudo_solver(Q, num_values, time)` expects a dense `Q` (`n x n`), the number of discrete values `num_values`, and a time limit in seconds.
```python
import numpy as np
from qudo_solver.qudo_solver_core.qubo_auxiliar_functions import qubo_list_to_matrix
from qudo_solver.qudo_solver_core.qubo_solvers import ortools_qudo_solver

# Convert triangular lists to a dense matrix
Q_dense = qubo_list_to_matrix(Q_lists)
sol = ortools_qudo_solver(Q_dense, num_values=d, time=10)
print(sol)
```

## Notebooks
- `qudo_solver/experimentation.ipynb`: experiments and plots.
- `qudo_solver/qubo-k-neighbors-*.ipynb`: notebook versions of the optimized implementations.

## Tips and notes
- Tune `tau` and `dits` for your problem. `tau` controls the "temperature" of the imaginary-time evolution.
- For large problems, prefer the matrix version if you do not need autodiff; use `tensorkrowch` if you want to leverage GPU or tensor workflows.
- The triangular-list format must be properly aligned; use `normalize_list_of_lists` for homogeneous scaling if needed.

## Development
- Dev requirement: `ipykernel` to run notebooks.
- Open and run the notebooks with the environment kernel.

## Citation
If you use this repository in academic work, please cite the MeLoCoTN methodology and this repository.
