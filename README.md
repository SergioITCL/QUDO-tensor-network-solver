# QUDO Solver

QUDO Solver provides two tensor network based methods to solve QUDO/QUBO (Quadratic Unconstrained Discrete/Binary Optimization) problems. It includes two optimized implementations of the MeLoCoTN method for fixed k-neighbors: one matrix-based `numpy` and one tensor-based using `tensorkrowch`.

- Author: Sergio Muñiz Subiñas (<sergio.muniz@itcl.es>)
- Organization: ITCL
- License: (add if applicable)

## Features
- Solve QUDO/QUBO with Two k-neighbors and discrete values of base `d` (bits, trits, ...) using two different methods.
  - Matrix-based with `numpy`/ (`qubo_k_neighbors_matrix_optimized.py`).
  - Tensor-based with `tensorkrowch`/`torch` (`qubo_k_neighbors_tensorkrowch_optimized.py`).
- Alternative reference solver using `OR-Tools` in `qudo_solver_core/qubo_solvers.py` (`ortools_qudo_solver`).

## Requirements
- Python 3.10+
- Recommended: virtual environment via `poetry`.


## Installation

### Option 1: Poetry (recommended)
```bash
# From the project root
poetry install

# Activate the virtualenv
poetry shell
```
## Quick start

This repository includes three notebooks, one for each variant of the method:

- A notebook for the matrix-based implementation (NumPy/SciPy), showing step-by-step instance generation, network construction, and solution via matrix operations.

- A notebook for the tensor-based implementation (tensorkrowch/Torch), following the same workflow but using tensors and backend-specific operations.

- A notebook that reproduces all the experiments reported in the paper.
```
Additionally, it includes to python scripts contianing each of the algorithms as a module.

## Tips and notes
- Tune `tau` for your problem. `tau` controls the "temperature" of the imaginary-time evolution.


## Development
- Dev requirement: `ipykernel` to run notebooks.
- Open and run the notebooks with the environment kernel.

## Citation
If you use this repository in academic work, please cite "" and this repository.
