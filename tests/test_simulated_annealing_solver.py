import itertools
from time import perf_counter
import unittest
from unittest.mock import patch

import numpy as np

from qudo_solver.solvers.simulated_annealing.simulated_annealing_solver import (
    _build_forward_interactions,
    _build_neighbors,
    _coordinate_descent,
    _delta_energy,
    _local_energies,
    _objective_energy,
    solver_simulated_annealing,
    solver_structure_aware_simulated_annealing,
)


def _minimum_energy(q_matrix, q_row, dits, require_nonzero=False):
    assignments = itertools.product(range(dits), repeat=len(q_matrix))
    if require_nonzero:
        assignments = (values for values in assignments if any(values))
    return min(_objective_energy(q_matrix, q_row, values) for values in assignments)


class SimulatedAnnealingSolverTests(unittest.TestCase):
    def _solve(self, q_matrix, q_row, dits, n_neighbors, **kwargs):
        return solver_structure_aware_simulated_annealing(
            q_matrix,
            q_row,
            dits,
            n_neighbors,
            time_limit=0.03,
            initial_temperature=2.0,
            final_temperature=0.02,
            cooling_rate=0.8,
            sweeps_per_temperature=1,
            seed=12345,
            **kwargs,
        )

    def test_binary_qubo_is_compared_with_brute_force(self):
        matrix = [[-1.0], [2.0, -1.5], [-0.5, 1.0, -0.25]]
        row = [0.0, 0.0, 0.0]
        result = self._solve(matrix, row, 2, 2, require_nonzero=False)
        optimum = _minimum_energy(matrix, row, 2)
        self.assertGreaterEqual(result.cost + 1e-12, optimum)
        self.assertTrue(all(value in (0, 1) for value in result.solution_list))

    def test_qudo_and_linear_terms_are_compared_with_brute_force(self):
        matrix = [[0.5], [-1.0, 0.25], [0.75, -0.5, 0.4]]
        row = [-1.25, 0.6, -0.8]
        result = self._solve(matrix, row, 4, 2, require_nonzero=False)
        optimum = _minimum_energy(matrix, row, 4)
        self.assertGreaterEqual(result.cost + 1e-12, optimum)
        self.assertTrue(all(0 <= value < 4 for value in result.solution_list))

    def test_intensive_configuration_frequently_recovers_the_optimum(self):
        matrix = [[0.5], [-1.0, 0.25], [0.75, -0.5, 0.4]]
        row = [-1.25, 0.6, -0.8]
        optimum = _minimum_energy(matrix, row, 4)
        successes = 0
        number_of_runs = 12
        for seed in range(number_of_runs):
            result = solver_simulated_annealing(
                matrix,
                row,
                dits=4,
                n_neighbors=2,
                time_limit=0.02,
                require_nonzero=False,
                initial_temperature=2.0,
                final_temperature=0.02,
                cooling_rate=0.8,
                sweeps_per_temperature=1,
                seed=seed,
            )
            successes += abs(result.cost - optimum) < 1e-10

        # This deliberately permits failures: SA has no per-run optimality
        # guarantee, but an intensive setup should work reliably on this case.
        self.assertGreaterEqual(successes, 8)

    def test_zero_neighbors(self):
        matrix = [[1.0], [-2.0], [0.5]]
        row = [-2.0, 1.0, -1.0]
        result = self._solve(matrix, row, 3, 0, require_nonzero=False)
        self.assertGreaterEqual(
            result.cost + 1e-12, _minimum_energy(matrix, row, 3)
        )
        self.assertAlmostEqual(
            result.cost, _objective_energy(matrix, row, result.solution_list)
        )

    def test_nonzero_constraint_is_preserved(self):
        result = self._solve(
            [[2.0], [1.0, 3.0], [0.5, 0.5, 4.0]],
            [0.0, 0.0, 0.0],
            2,
            2,
            require_nonzero=True,
        )
        self.assertTrue(any(result.solution_list))
        self.assertAlmostEqual(
            result.cost,
            _objective_energy(
                [[2.0], [1.0, 3.0], [0.5, 0.5, 4.0]],
                [0.0, 0.0, 0.0],
                result.solution_list,
            ),
        )

    def test_seed_is_reproducible(self):
        arguments = dict(
            q_matrix=[[-1.0], [0.4, -0.2], [0.8, -0.7, 0.1]],
            q_row=[0.2, -0.3, 0.5],
            dits=3,
            n_neighbors=2,
            time_limit=0.01,
            sweeps_per_temperature=2,
            seed=987,
        )

        def deterministic_run():
            clock = iter(index * 1e-4 for index in range(100_000))
            with patch(
                "qudo_solver.solvers.simulated_annealing."
                "simulated_annealing_solver.perf_counter",
                side_effect=lambda: next(clock),
            ):
                return solver_simulated_annealing(**arguments)

        first = deterministic_run()
        second = deterministic_run()
        self.assertEqual(first.solution_list, second.solution_list)
        self.assertEqual(first.cost, second.cost)

    def test_delta_matches_complete_objective(self):
        rng = np.random.default_rng(314159)
        for n in range(1, 9):
            for width in range(min(3, n - 1) + 1):
                matrix = [
                    rng.normal(size=min(i, width) + 1).tolist() for i in range(n)
                ]
                row = rng.normal(size=n).tolist()
                forward = _build_forward_interactions(matrix)
                for _ in range(100):
                    solution = rng.integers(0, 4, size=n).tolist()
                    variable = int(rng.integers(n))
                    new_value = int(rng.integers(4))
                    changed = solution.copy()
                    changed[variable] = new_value
                    expected = _objective_energy(matrix, row, changed) - _objective_energy(
                        matrix, row, solution
                    )
                    actual = _delta_energy(
                        matrix, row, solution, variable, new_value, forward
                    )
                    self.assertAlmostEqual(actual, expected, places=11)

    def test_objective_energy_matches_manual_calculation(self):
        matrix = [[1.5], [-2.0, 0.25], [3.0, -0.5, 2.0]]
        row = [0.75, -1.25, 0.5]
        solution = [2, 1, 3]
        expected = (
            1.5 * 2**2
            + 0.75 * 2
            - 2.0 * 2 * 1
            + 0.25 * 1**2
            - 1.25 * 1
            + 3.0 * 2 * 3
            - 0.5 * 1 * 3
            + 2.0 * 3**2
            + 0.5 * 3
        )
        self.assertAlmostEqual(_objective_energy(matrix, row, solution), expected)

    def test_local_energy_difference_matches_complete_objective(self):
        rng = np.random.default_rng(271828)
        for n in range(1, 8):
            width = min(3, n - 1)
            matrix = [
                rng.normal(size=min(i, width) + 1).tolist() for i in range(n)
            ]
            row = rng.normal(size=n).tolist()
            neighbors = _build_neighbors(matrix)
            for _ in range(40):
                solution = rng.integers(0, 4, size=n).tolist()
                variable = int(rng.integers(n))
                energies = _local_energies(
                    matrix, row, solution, variable, 4, neighbors
                )
                before = _objective_energy(matrix, row, solution)
                for candidate in range(4):
                    changed = solution.copy()
                    changed[variable] = candidate
                    global_delta = _objective_energy(matrix, row, changed) - before
                    local_delta = energies[candidate] - energies[solution[variable]]
                    self.assertAlmostEqual(global_delta, local_delta, places=11)

    def test_coordinate_descent_never_increases_energy(self):
        matrix = [[0.5], [-1.0, 0.25], [0.75, -0.5, 0.4], [0.2, -0.8, 0.3]]
        row = [-1.25, 0.6, -0.8, 0.1]
        solution = [3, 0, 2, 1]
        neighbors = _build_neighbors(matrix)
        before = _objective_energy(matrix, row, solution)

        energy, _, interrupted = _coordinate_descent(
            matrix,
            row,
            solution,
            dits=4,
            neighbors=neighbors,
            current_energy=before,
            nonzero_count=sum(value != 0 for value in solution),
            require_nonzero=True,
        )

        self.assertFalse(interrupted)
        self.assertLessEqual(energy, before)
        self.assertAlmostEqual(energy, _objective_energy(matrix, row, solution))
        self.assertTrue(any(solution))

    def test_solution_cost_uses_the_same_objective(self):
        matrix = [[0.3], [-0.4, 0.2], [0.1, 0.8, -0.5]]
        row = [0.7, -0.2, 0.9]
        result = self._solve(matrix, row, 3, 2, require_nonzero=False)
        self.assertAlmostEqual(
            result.cost, _objective_energy(matrix, row, result.solution_list)
        )

    def test_automatic_temperature_and_short_rows(self):
        result = solver_simulated_annealing(
            [[0.0], [0.0], [-1.0, 0.5]],
            [0.0, 0.0, 0.0],
            dits=2,
            n_neighbors=2,
            time_limit=0.01,
            cooling_rate=0.5,
            sweeps_per_temperature=2,
            seed=7,
        )
        self.assertTrue(any(result.solution_list))

    def test_rejects_rows_wider_than_n_neighbors(self):
        with self.assertRaisesRegex(ValueError, "at most"):
            solver_simulated_annealing(
                [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]],
                [0.0, 0.0, 0.0],
                dits=2,
                n_neighbors=1,
                time_limit=0.01,
            )

    def test_rejects_nonfinite_temperatures(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            solver_simulated_annealing(
                [[1.0]],
                [0.0],
                dits=2,
                n_neighbors=0,
                time_limit=0.01,
                initial_temperature=float("inf"),
            )

    def test_rejects_invalid_time_limit(self):
        with self.assertRaisesRegex(ValueError, "time_limit"):
            solver_simulated_annealing(
                [[1.0]], [0.0], dits=2, n_neighbors=0, time_limit=0.0
            )

    def test_search_respects_time_limit_approximately(self):
        matrix = [
            [0.1] * (min(position, 5) + 1) for position in range(500)
        ]
        started_at = perf_counter()
        result = solver_simulated_annealing(
            matrix,
            [0.0] * len(matrix),
            dits=3,
            n_neighbors=5,
            time_limit=0.05,
            seed=123,
        )
        elapsed = perf_counter() - started_at

        self.assertLess(elapsed, 0.25)
        self.assertLess(result.execution_time, 0.25)
        self.assertTrue(any(result.solution_list))

    def test_tiny_time_limit_still_returns_a_valid_solution(self):
        result = solver_structure_aware_simulated_annealing(
            [[1.0], [0.5, -1.0], [0.25, 0.75, 0.1]],
            [0.2, -0.3, 0.4],
            dits=3,
            n_neighbors=2,
            time_limit=1e-12,
            require_nonzero=True,
            seed=42,
        )
        self.assertEqual(len(result.solution_list), 3)
        self.assertTrue(any(result.solution_list))
        self.assertTrue(all(0 <= value < 3 for value in result.solution_list))


if __name__ == "__main__":
    unittest.main()
