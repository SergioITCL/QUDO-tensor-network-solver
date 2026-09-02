import itertools
from time import perf_counter
import unittest

import numpy as np

from qudo_solver.solvers.tabu_search.tabu_search_solver import (
    _best_admissible_move,
    _build_neighbors,
    _delta_energy,
    _diversify,
    _initialize_delta_cache,
    _is_move_admissible,
    _objective_energy,
    _refresh_delta_cache,
    solver_tabu_search,
    solver_tabu_search_time_to_target,
)


def _minimum_energy(q_matrix, q_row, dits, require_nonzero=False):
    assignments = itertools.product(range(dits), repeat=len(q_matrix))
    if require_nonzero:
        assignments = (values for values in assignments if any(values))
    return min(_objective_energy(q_matrix, q_row, values) for values in assignments)


class TabuSearchSolverTests(unittest.TestCase):
    def _solve(self, q_matrix, q_row, dits, n_neighbors, **kwargs):
        return solver_tabu_search(
            q_matrix,
            q_row,
            dits,
            n_neighbors,
            time_limit=1.0,
            max_iterations=80,
            diversification_interval=20,
            seed=12345,
            **kwargs,
        )

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

    def test_delta_matches_complete_objective(self):
        rng = np.random.default_rng(314159)
        for n in range(1, 9):
            for width in range(min(3, n - 1) + 1):
                matrix = [
                    rng.normal(size=min(i, width) + 1).tolist() for i in range(n)
                ]
                row = rng.normal(size=n).tolist()
                neighbors = _build_neighbors(matrix)
                for _ in range(80):
                    solution = rng.integers(0, 4, size=n).tolist()
                    variable = int(rng.integers(n))
                    new_value = int(rng.integers(4))
                    changed = solution.copy()
                    changed[variable] = new_value
                    expected = _objective_energy(matrix, row, changed) - _objective_energy(
                        matrix, row, solution
                    )
                    actual = _delta_energy(
                        matrix, row, solution, variable, new_value, neighbors
                    )
                    self.assertAlmostEqual(actual, expected, places=11)

    def test_incremental_cache_refreshes_only_affected_rows_correctly(self):
        matrix = [[0.3], [-0.2, 0.7], [0.5, 0.1], [-0.6, 0.2], [0.8, -0.4]]
        row = [0.2, -0.1, 0.4, -0.3, 0.1]
        solution = [2, 0, 1, 2, 0]
        neighbors = _build_neighbors(matrix)
        initialized = _initialize_delta_cache(matrix, row, solution, 3, neighbors)
        self.assertIsNotNone(initialized)
        cache, scores = initialized

        changed_variable = 1
        solution[changed_variable] = 2
        affected = {changed_variable}
        affected.update(neighbor for neighbor, _ in neighbors[changed_variable])
        self.assertTrue(
            _refresh_delta_cache(
                matrix,
                row,
                solution,
                3,
                neighbors,
                sorted(affected),
                cache,
                scores,
            )
        )

        for variable in range(len(solution)):
            for new_value in range(3):
                self.assertAlmostEqual(
                    cache[variable, new_value],
                    _delta_energy(
                        matrix,
                        row,
                        solution,
                        variable,
                        new_value,
                        neighbors,
                    ),
                )

    def test_binary_qubo_and_brute_force_reference(self):
        matrix = [[-1.0], [2.0, -1.5], [-0.5, 1.0, -0.25]]
        row = [0.0, 0.0, 0.0]
        result = self._solve(matrix, row, 2, 2, require_nonzero=False)
        optimum = _minimum_energy(matrix, row, 2)
        self.assertGreaterEqual(result.cost + 1e-12, optimum)
        self.assertTrue(all(value in (0, 1) for value in result.solution_list))

    def test_qudo_with_linear_terms(self):
        matrix = [[0.5], [-1.0, 0.25], [0.75, -0.5, 0.4]]
        row = [-1.25, 0.6, -0.8]
        result = self._solve(matrix, row, 4, 2, require_nonzero=False)
        optimum = _minimum_energy(matrix, row, 4)
        self.assertGreaterEqual(result.cost + 1e-12, optimum)
        self.assertAlmostEqual(
            result.cost, _objective_energy(matrix, row, result.solution_list)
        )

    def test_zero_neighbors(self):
        matrix = [[1.0], [-2.0], [0.5]]
        row = [-2.0, 1.0, -1.0]
        result = self._solve(matrix, row, 3, 0, require_nonzero=False)
        self.assertAlmostEqual(
            result.cost, _objective_energy(matrix, row, result.solution_list)
        )

    def test_nonzero_constraint(self):
        result = self._solve(
            [[2.0], [1.0, 3.0], [0.5, 0.5, 4.0]],
            [0.0, 0.0, 0.0],
            2,
            2,
            require_nonzero=True,
        )
        self.assertTrue(any(result.solution_list))

    def test_allows_zero_assignment_by_default(self):
        result = self._solve(
            [[2.0], [1.0, 3.0], [0.5, 0.5, 4.0]],
            [0.0, 0.0, 0.0],
            2,
            2,
        )

        self.assertEqual(result.solution_list, [0, 0, 0])
        self.assertEqual(result.cost, 0.0)

    def test_iteration_limited_runs_are_reproducible(self):
        arguments = dict(
            q_matrix=[[-1.0], [0.4, -0.2], [0.8, -0.7, 0.1]],
            q_row=[0.2, -0.3, 0.5],
            dits=3,
            n_neighbors=2,
            time_limit=10.0,
            candidate_list_size=2,
            diversification_interval=7,
            max_iterations=50,
            seed=987,
        )
        first = solver_tabu_search(**arguments)
        second = solver_tabu_search(**arguments)
        self.assertEqual(first.solution_list, second.solution_list)
        self.assertEqual(first.cost, second.cost)

    def test_aspiration_overrides_tabu_status(self):
        tabu_until = np.full((1, 2), -1, dtype=np.int64)
        tabu_until[0, 1] = 10
        self.assertFalse(
            _is_move_admissible(tabu_until, 0, 1, 5, 4.0, best_energy=3.0)
        )
        self.assertTrue(
            _is_move_admissible(tabu_until, 0, 1, 5, 2.0, best_energy=3.0)
        )

    def test_tabu_move_expires_after_tenure(self):
        tabu_until = np.full((1, 2), -1, dtype=np.int64)
        tabu_until[0, 1] = 5
        self.assertFalse(
            _is_move_admissible(tabu_until, 0, 1, 5, 2.0, best_energy=1.0)
        )
        self.assertTrue(
            _is_move_admissible(tabu_until, 0, 1, 6, 2.0, best_energy=1.0)
        )

    def test_best_improvement_can_select_a_positive_move(self):
        cache = np.array([[0.0, 1.0]])
        tabu_until = np.full((1, 2), -1, dtype=np.int64)
        move, _, interrupted = _best_admissible_move(
            cache,
            solution=[0],
            candidate_variables=[0],
            tabu_until=tabu_until,
            iteration=0,
            current_energy=0.0,
            best_energy=0.0,
            require_nonzero=False,
            nonzero_count=0,
        )
        self.assertFalse(interrupted)
        self.assertIsNotNone(move)
        self.assertEqual(move.delta, 1.0)

    def test_diversification_changes_state_and_preserves_feasibility(self):
        matrix = [[0.1] for _ in range(40)]
        row = [0.0] * len(matrix)
        solution = [1] + [0] * 39
        before = solution.copy()
        energy, nonzero_count, variables = _diversify(
            matrix,
            row,
            solution,
            dits=3,
            require_nonzero=True,
            rng=np.random.default_rng(99),
        )
        self.assertTrue(variables)
        self.assertNotEqual(solution, before)
        self.assertGreater(nonzero_count, 0)
        self.assertAlmostEqual(energy, _objective_energy(matrix, row, solution))

    def test_global_best_never_loses_the_initial_solution(self):
        matrix = [[0.7], [-0.2, 0.4], [0.6, -0.5, 0.8]]
        row = [0.3, -0.1, 0.2]
        seed = 2468
        initial = np.random.default_rng(seed).integers(0, 3, size=3).tolist()
        initial_energy = _objective_energy(matrix, row, initial)
        result = solver_tabu_search(
            matrix,
            row,
            dits=3,
            n_neighbors=2,
            time_limit=1.0,
            require_nonzero=False,
            greedy_initialization=False,
            max_iterations=30,
            seed=seed,
        )
        self.assertLessEqual(result.cost, initial_energy + 1e-12)

    def test_small_time_limit_returns_quickly_and_validly(self):
        matrix = [[0.1] * (min(position, 5) + 1) for position in range(300)]
        started_at = perf_counter()
        result = solver_tabu_search(
            matrix,
            [0.0] * len(matrix),
            dits=3,
            n_neighbors=5,
            time_limit=0.02,
            seed=123,
        )
        elapsed = perf_counter() - started_at
        self.assertLess(elapsed, 0.25)
        self.assertEqual(len(result.solution_list), len(matrix))
        self.assertTrue(all(value in range(3) for value in result.solution_list))

    def test_easy_target_is_reached_from_initial_incumbent(self):
        result = solver_tabu_search_time_to_target(
            [[1.0], [0.5, 2.0]],
            [0.0, 0.0],
            dits=3,
            n_neighbors=1,
            target_cost=100.0,
            max_time=1.0,
            greedy_initialization=False,
            seed=7,
        )

        self.assertTrue(result.reached)
        self.assertIsNotNone(result.time_to_target)
        self.assertLessEqual(result.time_to_target, result.total_execution_time)
        self.assertLessEqual(result.best_cost, result.target_cost + 1e-9)
        self.assertEqual(result.solution.execution_time, result.total_execution_time)

    def test_target_equal_to_optimum_is_reached(self):
        result = solver_tabu_search_time_to_target(
            [[1.0], [2.0], [3.0]],
            [0.0, 0.0, 0.0],
            dits=3,
            n_neighbors=0,
            target_cost=0.0,
            max_time=1.0,
            seed=123,
        )

        self.assertTrue(result.reached)
        self.assertEqual(result.best_cost, 0.0)
        self.assertEqual(result.solution.solution_list, [0, 0, 0])

    def test_impossible_target_is_not_reported_as_timeout_time(self):
        result = solver_tabu_search_time_to_target(
            [[1.0], [2.0], [3.0]],
            [0.0, 0.0, 0.0],
            dits=3,
            n_neighbors=0,
            target_cost=-1.0,
            max_time=1.0,
            max_iterations=30,
            seed=123,
        )

        self.assertFalse(result.reached)
        self.assertIsNone(result.time_to_target)
        self.assertEqual(result.best_cost, 0.0)

    def test_target_incumbent_history_is_monotone_and_coherent(self):
        tolerance = 1e-9
        result = solver_tabu_search_time_to_target(
            [[1.0], [2.0], [3.0]],
            [0.0, 0.0, 0.0],
            dits=3,
            n_neighbors=0,
            target_cost=0.0,
            max_time=1.0,
            target_tolerance=tolerance,
            seed=321,
        )

        self.assertTrue(result.incumbent_history)
        for (previous_time, previous_cost), (current_time, current_cost) in zip(
            result.incumbent_history, result.incumbent_history[1:]
        ):
            self.assertLessEqual(previous_time, current_time)
            self.assertLessEqual(current_cost, previous_cost + tolerance)
        self.assertTrue(result.reached)
        self.assertTrue(
            any(
                abs(elapsed - result.time_to_target) <= 1e-12
                and cost <= result.target_cost + tolerance
                for elapsed, cost in result.incumbent_history
            )
        )

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "time_limit"):
            solver_tabu_search([[1.0]], [0.0], 2, 0, time_limit=0.0)
        with self.assertRaisesRegex(ValueError, "at most"):
            solver_tabu_search(
                [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]],
                [0.0, 0.0, 0.0],
                2,
                1,
                time_limit=0.1,
            )
        with self.assertRaisesRegex(ValueError, "finite"):
            solver_tabu_search([[float("nan")]], [0.0], 2, 0, time_limit=0.1)
        with self.assertRaisesRegex(ValueError, "target_cost"):
            solver_tabu_search_time_to_target(
                [[1.0]], [0.0], 2, 0, target_cost=float("nan")
            )
        with self.assertRaisesRegex(ValueError, "target_tolerance"):
            solver_tabu_search_time_to_target(
                [[1.0]],
                [0.0],
                2,
                0,
                target_cost=0.0,
                target_tolerance=-1.0,
            )


if __name__ == "__main__":
    unittest.main()
