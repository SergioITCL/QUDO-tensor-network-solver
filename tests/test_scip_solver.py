import itertools
import random
import time
import unittest

from pyscipopt import Model

from qudo_solver.solvers.scip.solver_scip import (
    _build_quadratic_expression,
    _objective_energy,
    _validate_problem,
    solver_scip,
    solver_scip_time_to_target,
    solver_scip_with_metadata,
)


def _brute_force(q_matrix, q_row, dits, require_nonzero=False):
    assignments = itertools.product(range(dits), repeat=len(q_matrix))
    if require_nonzero:
        assignments = (assignment for assignment in assignments if any(assignment))
    assignment = min(
        assignments,
        key=lambda candidate: _objective_energy(q_matrix, q_row, candidate),
    )
    return assignment, _objective_energy(q_matrix, q_row, assignment)


class SCIPSolverTests(unittest.TestCase):
    TARGET_Q_MATRIX = [[1.0], [-2.0, 1.0], [0.5, -3.0, 2.0]]
    TARGET_Q_ROW = [0.0, -1.0, 0.5]

    def assert_matches_brute_force(
        self, q_matrix, q_row, dits, n_neighbors, require_nonzero=False
    ):
        _, expected_cost = _brute_force(
            q_matrix, q_row, dits, require_nonzero=require_nonzero
        )
        result = solver_scip(
            q_matrix,
            q_row,
            dits=dits,
            n_neighbors=n_neighbors,
            require_nonzero=require_nonzero,
            seed=17,
        )
        self.assertAlmostEqual(result.cost, expected_cost, places=7)
        self.assertEqual(
            result.cost,
            _objective_energy(q_matrix, q_row, result.solution_list),
        )

    def test_binary_qubo_matches_brute_force(self):
        self.assert_matches_brute_force(
            [[1.5], [-4.0, 2.0], [1.25, -3.0, 0.5]],
            [0.0, 0.0, 0.0],
            dits=2,
            n_neighbors=2,
        )

    def test_qudo_matches_brute_force(self):
        self.assert_matches_brute_force(
            [[1.0], [-2.0, 0.75], [1.5, -3.0, 2.0]],
            [0.0, 0.0, 0.0],
            dits=4,
            n_neighbors=2,
        )

    def test_linear_terms(self):
        self.assert_matches_brute_force(
            [[2.0], [0.0, 1.0], [-0.5, 0.25]],
            [-5.0, 3.0, -2.0],
            dits=3,
            n_neighbors=1,
        )

    def test_positive_and_negative_nonconvex_interactions(self):
        self.assert_matches_brute_force(
            [[-0.5], [3.0, 1.0], [-4.0, 2.0, -1.5], [2.5, -3.5, 0.25]],
            [1.0, -2.0, 0.5, 1.25],
            dits=3,
            n_neighbors=2,
        )

    def test_zero_neighbors_is_separable(self):
        self.assert_matches_brute_force(
            [[2.0], [-1.0], [0.5]],
            [-3.0, 2.0, -1.5],
            dits=4,
            n_neighbors=0,
        )

    def test_require_nonzero(self):
        q_matrix = [[1.0], [2.0]]
        q_row = [0.0, 0.0]

        unconstrained = solver_scip(q_matrix, q_row, 3, 0)
        constrained = solver_scip(
            q_matrix, q_row, 3, 0, require_nonzero=True
        )

        self.assertTrue(any(constrained.solution_list))
        self.assertEqual(constrained.cost, 1.0)
        self.assertEqual(unconstrained.solution_list, [0, 0])
        self.assertEqual(unconstrained.cost, 0.0)

    def test_compact_expression_matches_external_energy(self):
        rng = random.Random(271828)
        q_matrix = [
            [rng.uniform(-3, 3) for _ in range(min(i, 3) + 1)]
            for i in range(6)
        ]
        q_row = [rng.uniform(-2, 2) for _ in q_matrix]
        model = Model()
        variables = [model.addVar(vtype="I", lb=0, ub=3) for _ in q_matrix]
        expression = _build_quadratic_expression(q_matrix, q_row, variables)

        for _ in range(20):
            assignment = [rng.randrange(4) for _ in q_matrix]
            scip_solution = model.createSol()
            for variable, value in zip(variables, assignment):
                model.setSolVal(scip_solution, variable, value)
            self.assertAlmostEqual(
                model.getSolVal(scip_solution, expression),
                _objective_energy(q_matrix, q_row, assignment),
                places=10,
            )

    def test_time_limit_returns_an_incumbent_or_reports_none(self):
        rng = random.Random(1234)
        n_variables = 30
        n_neighbors = 8
        q_matrix = [
            [rng.uniform(-1, 1) for _ in range(min(i, n_neighbors) + 1)]
            for i in range(n_variables)
        ]
        q_row = [rng.uniform(-1, 1) for _ in q_matrix]

        started_at = time.perf_counter()
        try:
            result, metadata = solver_scip_with_metadata(
                q_matrix,
                q_row,
                dits=4,
                n_neighbors=n_neighbors,
                time_limit=0.1,
                seed=5,
            )
        except RuntimeError as error:
            self.assertIn("without finding a feasible solution", str(error))
        else:
            self.assertEqual(len(result.solution_list), n_variables)
            self.assertIn(metadata.status, {"optimal", "timelimit"})
            self.assertGreaterEqual(metadata.nodes, 0)
        self.assertLess(time.perf_counter() - started_at, 5.0)

    def test_time_limited_small_instance_returns_a_feasible_solution(self):
        result = solver_scip(
            self.TARGET_Q_MATRIX,
            self.TARGET_Q_ROW,
            dits=3,
            n_neighbors=2,
            time_limit=1.0,
            seed=3,
        )

        self.assertEqual(len(result.solution_list), 3)
        self.assertTrue(any(result.solution_list))
        self.assertLess(result.execution_time, 1.5)

    def test_metadata_for_optimal_run(self):
        result, metadata = solver_scip_with_metadata(
            [[1.0], [-2.0, 1.0]], [0.0, 0.0], 2, 1
        )

        self.assertEqual(metadata.status, "optimal")
        self.assertAlmostEqual(metadata.objective, result.cost, places=6)
        self.assertAlmostEqual(metadata.best_bound, result.cost, places=6)
        self.assertAlmostEqual(metadata.gap, 0.0)
        self.assertGreaterEqual(metadata.solving_time, 0.0)

    def test_easy_target_is_reached(self):
        assignments = list(itertools.product(range(3), repeat=3))
        worst_cost = max(
            _objective_energy(self.TARGET_Q_MATRIX, self.TARGET_Q_ROW, candidate)
            for candidate in assignments
            if any(candidate)
        )

        result = solver_scip_time_to_target(
            self.TARGET_Q_MATRIX,
            self.TARGET_Q_ROW,
            dits=3,
            n_neighbors=2,
            target_cost=worst_cost + 1.0,
            max_time=2.0,
            seed=3,
        )

        self.assertTrue(result.reached)
        self.assertIsNotNone(result.time_to_target)
        self.assertIsNotNone(result.solution)
        self.assertLessEqual(result.best_cost, result.target_cost + 1e-9)

    def test_target_equal_to_optimum_is_reached(self):
        _, optimum = _brute_force(
            self.TARGET_Q_MATRIX, self.TARGET_Q_ROW, dits=3
        )

        result = solver_scip_time_to_target(
            self.TARGET_Q_MATRIX,
            self.TARGET_Q_ROW,
            dits=3,
            n_neighbors=2,
            target_cost=optimum,
            max_time=2.0,
            seed=3,
        )

        self.assertTrue(result.reached)
        self.assertAlmostEqual(result.best_cost, optimum)
        self.assertIsNotNone(result.time_to_target)

    def test_impossible_target_returns_best_incumbent(self):
        _, optimum = _brute_force(
            self.TARGET_Q_MATRIX, self.TARGET_Q_ROW, dits=3
        )

        result = solver_scip_time_to_target(
            self.TARGET_Q_MATRIX,
            self.TARGET_Q_ROW,
            dits=3,
            n_neighbors=2,
            target_cost=optimum - 1.0,
            max_time=0.5,
            seed=3,
        )

        self.assertFalse(result.reached)
        self.assertIsNone(result.time_to_target)
        self.assertIsNotNone(result.solution)
        self.assertAlmostEqual(result.best_cost, optimum)

    def test_incumbent_history_and_target_timestamp_are_coherent(self):
        _, optimum = _brute_force(
            self.TARGET_Q_MATRIX, self.TARGET_Q_ROW, dits=3
        )
        tolerance = 1e-9

        result = solver_scip_time_to_target(
            self.TARGET_Q_MATRIX,
            self.TARGET_Q_ROW,
            dits=3,
            n_neighbors=2,
            target_cost=optimum,
            max_time=2.0,
            seed=3,
            target_tolerance=tolerance,
        )

        self.assertTrue(result.incumbent_history)
        for (previous_time, previous_cost), (current_time, current_cost) in zip(
            result.incumbent_history, result.incumbent_history[1:]
        ):
            self.assertLessEqual(previous_time, current_time)
            self.assertLessEqual(current_cost, previous_cost + tolerance)

        self.assertTrue(result.reached)
        self.assertLessEqual(result.time_to_target, result.total_execution_time)
        matching_entries = [
            (elapsed, cost)
            for elapsed, cost in result.incumbent_history
            if abs(elapsed - result.time_to_target) <= 1e-12
            and cost <= result.target_cost + tolerance
        ]
        self.assertTrue(matching_entries)

    def test_target_mode_final_solution_matches_brute_force(self):
        _, optimum = _brute_force(
            self.TARGET_Q_MATRIX, self.TARGET_Q_ROW, dits=3
        )

        result = solver_scip_time_to_target(
            self.TARGET_Q_MATRIX,
            self.TARGET_Q_ROW,
            dits=3,
            n_neighbors=2,
            target_cost=optimum - 1.0,
            max_time=2.0,
        )

        self.assertAlmostEqual(result.best_cost, optimum)
        self.assertAlmostEqual(result.solution.cost, optimum)

    def test_validation(self):
        invalid_cases = [
            ([], [], 2, 0),
            ([[1.0]], [], 2, 0),
            ([[1.0]], [0.0], 1, 0),
            ([[1.0]], [0.0], 2, -1),
            ([[]], [0.0], 2, 0),
            ([[float("nan")]], [0.0], 2, 0),
            ([[1.0]], [float("inf")], 2, 0),
            ([[1.0], [2.0, 3.0]], [0.0, 0.0], 2, 0),
        ]
        for q_matrix, q_row, dits, n_neighbors in invalid_cases:
            with self.subTest(q_matrix=q_matrix, q_row=q_row):
                with self.assertRaises(ValueError):
                    _validate_problem(q_matrix, q_row, dits, n_neighbors)


if __name__ == "__main__":
    unittest.main()
