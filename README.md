# QUDO Solver

QUDO Solver es un conjunto de algoritmos y utilidades para resolver problemas QUDO/QUBO (Quadratic Unconstrained Discrete/Binary Optimization) usando redes tensoriales y otros enfoques. Incluye dos implementaciones optimizadas del método MeLoCoTN para el caso de k-vecinos fijos: una basada en matrices (`numpy/scipy`) y otra basada en tensores con `tensorkrowch`.

- **Autor**: Sergio Muñiz Subiñas (<sergio.muniz@itcl.es>)
- **Organización**: ITCL
- **Licencia**: (añadir si aplica)

## Características
- Resolución de QUDO/QUBO con valores discretos de base `d` (bits, trits, ...).
- Dos implementaciones del solver k-vecinos:
  - Implementación matricial con `numpy`/`scipy` (`qubo_k_neighbors_matrix_optimized.py`).
  - Implementación con `tensorkrowch`/`torch` (`qubo_k_neighbors_tensorkrowch_optimized.py`).
- Funciones auxiliares para generar y evaluar instancias en formato de listas triangulares y convertirlas a matrices (`qudo_solver_core/qubo_auxiliar_functions.py`).
- Solver alternativo con `OR-Tools` para referencia en `qudo_solver_core/qubo_solvers.py` (`ortools_qudo_solver`).

## Requisitos
- Python 3.10+
- Recomendado: entorno virtual con `poetry`.

Dependencias principales (ver `pyproject.toml` para versiones exactas):
- `numpy`, `scipy`, `matplotlib`
- `torch`, `tensorkrowch`
- `dimod`, `dwave-neal`, `dwave-system` (para ecosistema D-Wave si se desea)
- `ortools`

## Instalación

### Opción 1: Poetry (recomendada)
```bash
# Instalar Poetry si no lo tienes
# Windows PowerShell
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# En la raíz del proyecto
poetry install

# Activar el entorno
poetry shell
```

### Opción 2: pip + venv
```bash
# Crear y activar venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Instalar dependencias principales
pip install -r <(poetry export -f requirements.txt --without-hashes)
# Si no usas poetry, instala manualmente según pyproject.toml
```

## Estructura del proyecto
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

## Formatos de entrada QUBO/QUDO
Para problemas de k-vecinos, se usa un formato de "lista triangular alineada a la derecha" (`lists_tri`) donde:
- La fila `i` contiene los coeficientes de las columnas `j` desde `i - len(row) + 1` hasta `i` (inclusive).
- El término cuadrático `Q[i,i]` es el último elemento de la fila `i`.

Puedes convertir este formato a matriz densa con `qubo_list_to_matrix`.

Funciones relevantes en `qudo_solver_core/qubo_auxiliar_functions.py`:
- `qudo_evaluation(Q_matrix, x)` — evalúa el coste `x^T Q x`.
- `generate_1k_qubo(n_variables, seed=None)` — genera instancia con 1 vecino.
- `generate_k_qubo(n_variables, k_neighbour, seed=None)` — genera instancia con hasta `k` vecinos.
- `normalize_list_of_lists(Q_matrix)` — normaliza manteniendo forma.
- `qubo_list_to_matrix(lists, fill=0.0)` — convierte listas triangulares a matriz densa.
- `qubo_value_from_lists(x, lists_tri)` — evalúa energía directamente sobre el formato triangular.

## Uso rápido
A continuación se muestra cómo resolver una instancia con ambos enfoques.

### Solver matricial (NumPy/Scipy)
```python
from qudo_solver.qudo_solver_core.qubo_auxiliar_functions import generate_k_qubo, normalize_list_of_lists
from qudo_solver.qubo_k_neighbors_matrix_optimized import qubo_solver_matrix

# Parámetros del problema
n_variables = 10
k_neighbors = 3
 d = 2              # bits (2), trits (3), etc.
 tau = 0.5

# Generar instancia y normalizar
Q_lists = generate_k_qubo(n_variables, k_neighbors, seed=123)
Q_lists = normalize_list_of_lists(Q_lists)

# Resolver
solution = qubo_solver_matrix(Q_lists, tau=tau, dits=d, n_neighbors=k_neighbors)
print("Solución:", solution)
```

### Solver con Tensorkrowch (Torch)
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
print("Solución:", solution)
```

### Solver de referencia con OR-Tools
`ortools_qudo_solver(Q, num_values, time)` acepta `Q` denso (`n x n`), el número de valores discretos `num_values` y tiempo límite en segundos.
```python
import numpy as np
from qudo_solver.qudo_solver_core.qubo_auxiliar_functions import qubo_list_to_matrix
from qudo_solver.qudo_solver_core.qubo_solvers import ortools_qudo_solver

# Convertir listas triangulares a matriz densa
Q_dense = qubo_list_to_matrix(Q_lists)
sol = ortools_qudo_solver(Q_dense, num_values=d, time=10)
print(sol)
```

## Notebooks
- `qudo_solver/experimentation.ipynb`: experimentos y gráficos.
- `qudo_solver/qubo-k-neighbors-*.ipynb`: versiones notebook de las implementaciones optimizadas.

## Consejos y límites
- Ajusta `tau` y `dits` según el problema. `tau` controla la "temperatura" de la evolución imaginaria.
- En problemas grandes, prefiere la versión matricial si no necesitas autodiff; usa `tensorkrowch` si deseas explotar GPU o flujos con tensores.
- El formato de listas triangulares debe estar correctamente alineado; usa `normalize_list_of_lists` para escalado homogéneo si tu instancia lo requiere.

## Desarrollo
- Requisitos de desarrollo: `ipykernel` para ejecutar notebooks.
- Ejecuta los notebooks con el kernel del entorno creado.

## Citar
Si usas este repositorio en trabajos académicos, por favor cita la metodología MeLoCoTN y este repositorio.
