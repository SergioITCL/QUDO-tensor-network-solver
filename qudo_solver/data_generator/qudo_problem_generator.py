from typing import List, Optional, Tuple
import numpy as np
import random
from collections.abc import Iterable
from numbers import Integral


def generate_k_qubo(
    n_variables: int, 
    k_neighbor: int, 
    seed=None
    ) -> List[List[float]]:
    if seed is not None:
        random.seed(seed)  

    Q_list = [[] for _ in range(n_variables)]
    n_elements_per_row = 1
    for _ in range(n_variables):
        for element in range(n_elements_per_row):
            Q_list[_].append(random.uniform(-5, 5))
        
        if n_elements_per_row <= k_neighbor:
            n_elements_per_row += 1
 
    return Q_list
    #return normalize_list_of_lists(Q_list)

def normalize_list_of_lists(Q_matrix: List[List[float]]):
    # Flatten all values, even if some "rows" are float
    all_values = []

    for row in Q_matrix:
        if isinstance(row, Iterable) and not isinstance(row, (str, bytes)):
            all_values.extend(row)
        else:
            all_values.append(row)

    norm = np.linalg.norm(all_values)

    # Normalize respecting the original shape
    Q_normalized = []
    for row in Q_matrix:
        if isinstance(row, Iterable) and not isinstance(row, (str, bytes)):
            Q_normalized.append([elem / norm for elem in row])
        else:
            Q_normalized.append(float(row) / norm)

    return Q_normalized

def generate_frustrated_k_qubo(
    n_variables: int,
    k_neighbor: int,
    seed: Optional[int] = None,
    coupling_range: Tuple[float, float] = (1.0, 5.0),
    field_strength: float = 0.25,
    frustration_probability: float = 1.0,
) -> List[List[float]]:
    """
    Genera un QUBO binario con interacciones locales y triángulos frustrados.

    La representación compacta de cada fila es:

        Q_list[i][0] = Q[i, i]
        Q_list[i][1] = Q[i, i - 1]
        Q_list[i][2] = Q[i, i - 2]
        ...

    El objetivo QUBO asumido es:

        f(x) = sum_i Q[i, i] * x_i
             + sum_{i > j} Q[i, j] * x_i * x_j

    con x_i en {0, 1}.

    La instancia se construye inicialmente como un modelo de Ising:

        E(s) = sum_{i > j} J[i, j] * s_i * s_j
             + sum_i h_i * s_i

    con s_i en {-1, +1}, usando la transformación s_i = 2*x_i - 1.

    Para minimización:
        J[i, j] > 0 favorece spins distintos.
        J[i, j] < 0 favorece spins iguales.

    Un triángulo es frustrado cuando contiene un número impar de
    interacciones antiferromagnéticas. Con esta convención, esto equivale a:

        sign(J_ab) * sign(J_bc) * sign(J_ac) = +1

    Parameters
    ----------
    n_variables:
        Número de variables binarias.

    k_neighbor:
        Número máximo de variables anteriores conectadas con cada variable.

    seed:
        Semilla aleatoria.

    coupling_range:
        Intervalo de magnitudes absolutas de las interacciones de Ising.

    field_strength:
        Magnitud máxima de los campos locales h_i. Un valor pequeño evita
        que los términos diagonales dominen las interacciones frustradas.

    frustration_probability:
        Probabilidad de hacer frustrado cada triángulo consecutivo.
        Debe estar entre 0 y 1.

    Returns
    -------
    List[List[float]]
        QUBO en representación triangular compacta.
    """
    if n_variables <= 0:
        raise ValueError("n_variables debe ser mayor que cero")

    if k_neighbor < 0:
        raise ValueError("k_neighbor no puede ser negativo")

    min_coupling, max_coupling = coupling_range

    if min_coupling <= 0:
        raise ValueError("La magnitud mínima debe ser positiva")

    if min_coupling > max_coupling:
        raise ValueError(
            "coupling_range debe satisfacer min_coupling <= max_coupling"
        )

    if not 0.0 <= frustration_probability <= 1.0:
        raise ValueError(
            "frustration_probability debe estar entre 0 y 1"
        )

    rng = random.Random(seed)

    # Interacciones Ising J_(i,j), almacenadas con i > j.
    couplings: dict[tuple[int, int], float] = {}

    # Signos de las interacciones entre vecinos consecutivos:
    # nearest_sign[i] corresponde a la arista (i, i-1).
    nearest_sign: dict[int, int] = {}

    for i in range(1, n_variables):
        sign = rng.choice((-1, 1))
        magnitude = rng.uniform(min_coupling, max_coupling)

        nearest_sign[i] = sign
        couplings[(i, i - 1)] = sign * magnitude

    # Interacciones de distancia 2.
    #
    # Para el triángulo (i-2, i-1, i), imponemos:
    #
    # sign(J_(i,i-1)) *
    # sign(J_(i-1,i-2)) *
    # sign(J_(i,i-2)) = +1
    #
    # Por tanto:
    #
    # sign(J_(i,i-2)) =
    # sign(J_(i,i-1)) * sign(J_(i-1,i-2))
    if k_neighbor >= 2:
        for i in range(2, n_variables):
            should_be_frustrated = (
                rng.random() < frustration_probability
            )

            triangle_product_sign = (
                nearest_sign[i] * nearest_sign[i - 1]
            )

            if should_be_frustrated:
                sign = triangle_product_sign
            else:
                sign = -triangle_product_sign

            magnitude = rng.uniform(min_coupling, max_coupling)
            couplings[(i, i - 2)] = sign * magnitude

    # Las interacciones con distancia mayor que 2 se generan aleatoriamente.
    for distance in range(3, k_neighbor + 1):
        for i in range(distance, n_variables):
            sign = rng.choice((-1, 1))
            magnitude = rng.uniform(min_coupling, max_coupling)
            couplings[(i, i - distance)] = sign * magnitude

    # Campos locales pequeños.
    local_fields = [
        rng.uniform(-field_strength, field_strength)
        for _ in range(n_variables)
    ]

    # Conversión Ising -> QUBO usando s_i = 2*x_i - 1:
    #
    # J_ij s_i s_j =
    # 4 J_ij x_i x_j - 2 J_ij x_i - 2 J_ij x_j + J_ij
    #
    # h_i s_i = 2 h_i x_i - h_i
    diagonal = [2.0 * h for h in local_fields]
    qubo_interactions: dict[tuple[int, int], float] = {}

    for (i, j), coupling in couplings.items():
        qubo_interactions[(i, j)] = 4.0 * coupling

        diagonal[i] -= 2.0 * coupling
        diagonal[j] -= 2.0 * coupling

    # Construcción de la representación compacta.
    Q_list: List[List[float]] = []

    for i in range(n_variables):
        row = [diagonal[i]]

        max_distance = min(i, k_neighbor)

        for distance in range(1, max_distance + 1):
            j = i - distance
            row.append(qubo_interactions[(i, j)])

        Q_list.append(row)

    return Q_list