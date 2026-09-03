# QUDO Solver

Python implementations of exact and approximate methods for QUDO/QUBO problems with local `k`-neighbor interactions. The main methods used by the experiments are:

- `SMVC`: Sparse Matrix--Vector Contraction, based on NumPy/SciPy.
- `STC`: Staircase Tensor Contraction, based on PyTorch/TensorKrowch.
- Exact Dynamic Programming.
- Beam Dynamic Programming.
- Tabu Search.
- SCIP.

All experiment parameters are centralized in [`experimentation/experiments.json`](experimentation/experiments.json). See [`experimentation/README.md`](experimentation/README.md) for reproducibility commands.

## Requirements

- Python 3.10 or later.
- Poetry, recommended for environment management.
- PyTorch and TensorKrowch for `STC`.
- A working PySCIPOpt/SCIP installation for the SCIP experiments.

The main dependencies are listed in `pyproject.toml` and resolved versions are recorded in `poetry.lock`.

## Installation

From the repository root:

```bash
poetry install
poetry run python -c "import qudo_solver; print('QUDO Solver installed')"
```

You may also activate the Poetry environment with `poetry shell`.

## Basic usage

Solvers receive a compact lower-triangular matrix `Q`, a vector of linear coefficients `q`, the local dimension `dits`, and the neighbor range `k`:

```python
from qudo_solver.data_generator.qudo_problem_generator import qudo_problem_generation
from qudo_solver.solvers.smvc.smvc import solver_smvc

instance = qudo_problem_generation(20, 2, 1, 0)[0]
result = solver_smvc(
    instance["q_matrix"], instance["q_row"], dits=2, n_neighbors=2
)
print(result.solution_list, result.cost)
```

Solvers return `SolutionClass`, containing the solution, objective value, and execution time.

## Tests

Run the test suite with:

```bash
poetry run pytest
```

The current copy still contains historical tests for solvers that are no longer present, so full collection may fail. Tests for the currently maintained components can be run with:

```bash
poetry run pytest \
  tests/test_scip_solver.py \
  tests/test_tabu_search_solver.py \
  tests/test_experiment_6.py
```

## Repository layout

```text
qudo_solver/       Solver implementations, instance generation, and result model
experimentation/   Experiment scripts and result processors
tests/             Unit and integration tests
main.py            Legacy comparison script; it may reference retired modules
```

## Reproducibility

Experiments generate instances with deterministic integer seeds and write intermediate results to `results/` directories. Generated JSON and PNG files are excluded from Git to avoid committing large artifacts; processed tables and selected results are retained under `processed_results/`.

To reproduce a table or figure, use the generation script, the experiment configuration, and the corresponding raw results. See [`experimentation/README.md`](experimentation/README.md) for the exact inventory.

## Known status

- `main.py` is not fully aligned with the current `qudo_solver` layout.
- Some historical tests and documentation references point to removed modules.
- Numerical reproduction depends on CPU, BLAS/PyTorch/TensorKrowch versions, SCIP, and the operating system.
- Runtime and memory results should not be compared across machines without recording the environment.

## License and citation

The project is distributed under the license in [`LICENSE`](LICENSE). Complete paper citation metadata should be added before the repository is released as a final publication companion.
