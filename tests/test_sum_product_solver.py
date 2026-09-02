import itertools
import random
import unittest

from qudo_solver.auxiliar_functions import qubo_value_from_lists
from qudo_solver.solvers.sum_product import solver_sum_product


def _brute_force(q_matrix, dits, require_nonzero):
    assignments = itertools.product(range(dits), repeat=len(q_matrix))
    if require_nonzero:
        assignments = (assignment for assignment in assignments if any(assignment))
    solution = min(
        assignments,
        key=lambda assignment: qubo_value_from_lists(assignment, q_matrix),
    )
    return solution, qubo_value_from_lists(solution, q_matrix)


class SumProductSolverTests(unittest.TestCase):
    def test_allows_zero_assignment_by_default(self):
        result = solver_sum_product(
            [[1.0], [0.0, 2.0], [0.0, 0.0, 3.0]],
            [0.0, 0.0, 0.0],
            dits=2,
            n_neighbors=2,
        )

        self.assertEqual(result.solution_list, [0, 0, 0])
        self.assertEqual(result.cost, 0.0)

    def test_can_enforce_the_nonzero_variant(self):
        result = solver_sum_product(
            [[1.0], [0.0, 2.0]],
            [0.0, 0.0],
            dits=2,
            n_neighbors=1,
            require_nonzero=True,
        )

        self.assertEqual(result.solution_list, [1, 0])
        self.assertEqual(result.cost, 1.0)

    def test_matches_brute_force_on_random_instances(self):
        rng = random.Random(271828)

        for n_variables in range(1, 7):
            for dits in (2, 3):
                for n_neighbors in range(0, min(3, n_variables - 1) + 1):
                    q_matrix = [
                        [
                            rng.uniform(-5.0, 5.0)
                            for _ in range(min(position, n_neighbors) + 1)
                        ]
                        for position in range(n_variables)
                    ]

                    for require_nonzero in (False, True):
                        _, expected_cost = _brute_force(
                            q_matrix,
                            dits,
                            require_nonzero,
                        )
                        result = solver_sum_product(
                            q_matrix,
                            [0.0] * len(q_matrix),
                            dits=dits,
                            n_neighbors=n_neighbors,
                            require_nonzero=require_nonzero,
                        )
                        self.assertAlmostEqual(result.cost, expected_cost)

    def test_rejects_a_row_wider_than_the_separator(self):
        with self.assertRaisesRegex(ValueError, "can represent"):
            solver_sum_product(
                [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]],
                [0.0, 0.0, 0.0],
                dits=2,
                n_neighbors=1,
            )


if __name__ == "__main__":
    unittest.main()
