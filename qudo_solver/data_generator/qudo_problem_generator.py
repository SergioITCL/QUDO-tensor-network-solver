import random
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal, TypedDict

import numpy as np


class QudoInstance(TypedDict):
    instance_type: Literal["random", "fixed"]
    seed: int
    q_matrix: list[list[float]]
    q_row: list[float]

def normalize_list_of_lists(Q_matrix: list[list[float]]):
    # Flatten all values, even if some "rows" are float
    all_values = []

    for row in Q_matrix:
        if isinstance(row, Iterable) and not isinstance(row, (str, bytes)):
            all_values.extend(row)
        else:
            all_values.append(row)

    norm = np.linalg.norm(all_values)
    if norm == 0:
        raise ValueError("No se puede normalizar una matriz de norma cero")

    # Normalize respecting the original shape
    Q_normalized = []
    for row in Q_matrix:
        if isinstance(row, Iterable) and not isinstance(row, (str, bytes)):
            Q_normalized.append([elem / norm for elem in row])
        else:
            Q_normalized.append(float(row) / norm)

    return Q_normalized

def normalize_problem(
    Q_matrix: list[list[float]],
    Q_row: list[float],
) -> tuple[list[list[float]], list[float]]:

    if len(Q_matrix) != len(Q_row):
        raise ValueError(
            "Q_matrix y Q_row deben tener la misma longitud"
        )

    all_values = []

    for row in Q_matrix:
        all_values.extend(row)

    all_values.extend(Q_row)

    norm = np.linalg.norm(all_values)

    if norm == 0:
        raise ValueError("No se puede normalizar un problema de norma cero")

    Q_matrix_normalized = [
        [float(value / norm) for value in row]
        for row in Q_matrix
    ]

    Q_row_normalized = [
        float(value / norm)
        for value in Q_row
    ]

    return Q_matrix_normalized, Q_row_normalized

def qudo_problem_generation(
    n_variables: int,
    n_neighbors: int,
    n_random_instances: int,
    n_fixed_instances: int,
) -> list[QudoInstance]:
    if n_variables < 1:
        raise ValueError("n_variables debe ser mayor que 0")
    if n_neighbors < 1:
        raise ValueError("n_neighbors debe ser mayor que 0")
    if n_random_instances < 0 or n_fixed_instances < 0:
        raise ValueError("El número de instancias no puede ser negativo")
    if n_random_instances + n_fixed_instances == 0:
        raise ValueError("Debe generarse al menos una instancia")

    qudo_instances: list[QudoInstance] = []

    for seed in range(n_random_instances):
        qudo_instances.append(
            {
                "instance_type": "random",
                "seed": seed,
                "q_matrix": generate_k_random_qudo(
                    n_variables, n_neighbors, seed
                ),
                "q_row": generate_random_q_row(
                    n_variables, seed
                )
            }
        )

    for seed in range(n_fixed_instances):
        qudo_instances.append(
            {
                "instance_type": "fixed",
                "seed": seed,
                "q_matrix": generate_fixed_interactions_qudo(
                    n_variables, n_neighbors, seed
                ),
                "q_row": generate_fixed_q_row(
                    n_variables, seed
                )
            }
        )

    return qudo_instances




def generate_k_random_qudo(
    n_variables: int,
    k_neighbor: int,
    seed: int | None = None,
) -> list[list[float]]:
    if n_variables < 1:
        raise ValueError("n_variables debe ser mayor que 0")
    if k_neighbor < 1:
        raise ValueError("k_neighbor debe ser mayor que 0")

    rng = random.Random(seed)
    Q_list = [[] for _ in range(n_variables)]
    n_elements_per_row = 1
    for row_index in range(n_variables):
        for _ in range(n_elements_per_row):
            Q_list[row_index].append(rng.uniform(-10, 10))

        if n_elements_per_row <= k_neighbor:
            n_elements_per_row += 1

    return Q_list

def generate_random_q_row(
    n_variables: int,
    seed: int | None = None,
) -> list[float]:
    """Genera coeficientes lineales independientes."""
    if n_variables < 1:
        raise ValueError("n_variables debe ser mayor que 0")

    # Prefijo para que q_row no repita la secuencia usada por q_matrix.
    rng = random.Random(
        f"q-row-random-{seed}" if seed is not None else None
    )

    return [
        rng.uniform(-10, 10)
        for _ in range(n_variables)
    ]

def generate_fixed_interactions_qudo(
    n_variables: int,
    k_neighbors: int,
    seed: int | None = None,
) -> list[list[float]]:
    """
    Genera una QUDO triangular inferior con interacciones aleatorias
    repetibles para una misma semilla.
    """
    if n_variables < 1:
        raise ValueError("n_variables debe ser mayor que 0")
    if k_neighbors < 1:
        raise ValueError("k_neighbors debe ser mayor que 0")

    rng = random.Random(seed)
    neighbor_values = [
        rng.uniform(-5, 5)
        for _ in range(k_neighbors)
    ]
    diagonal_value = rng.uniform(-5, 5)

    Q_list = []

    for i in range(n_variables):
        max_distance = min(i, k_neighbors)

        row = [
            neighbor_values[distance - 1]
            for distance in range(max_distance, 0, -1)
        ]

        # La diagonal es fija entre filas, pero independiente de los acoplamientos.
        row.append(diagonal_value)

        Q_list.append(row)

    return Q_list

def generate_fixed_q_row(
    n_variables: int,
    seed: int | None = None,
) -> list[float]:
    """Genera un mismo coeficiente lineal para todas las variables."""
    if n_variables < 1:
        raise ValueError("n_variables debe ser mayor que 0")

    rng = random.Random(
        f"q-row-fixed-{seed}" if seed is not None else None
    )
    linear_value = rng.uniform(-5, 5)

    return [linear_value] * n_variables