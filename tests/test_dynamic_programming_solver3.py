import itertools
import random
import unittest

from qudo_solver.auxiliar_functions import qubo_value_from_lists
from qudo_solver.solvers.dynamic_programming.dynamic_programming_solver3 import (
    solver_dynamic_programming3,
)


def _brute_force(q_matrix, dits, require_nonzero=False):
    feasible_assignments = itertools.product(range(dits), repeat=len(q_matrix))
    if require_nonzero:
        feasible_assignments = (
            assignment for assignment in feasible_assignments if any(assignment)
        )
    return min(
        feasible_assignments,
        key=lambda assignment: qubo_value_from_lists(assignment, q_matrix),
    )


class DynamicProgrammingSolver3Tests(unittest.TestCase):
    def test_allows_the_all_zero_assignment_by_default(self):
        q_matrix = [[1.0], [0.0, 2.0], [0.0, 0.0, 3.0]]

        result = solver_dynamic_programming3(
            q_matrix, [0.0] * len(q_matrix), dits=2, n_neighbors=2
        )

        self.assertEqual(result.solution_list, [0, 0, 0])
        self.assertEqual(result.cost, 0.0)

    def test_can_explicitly_exclude_the_all_zero_assignment(self):
        result = solver_dynamic_programming3(
            [[1.0], [0.0, 2.0]],
            [0.0, 0.0],
            dits=2,
            n_neighbors=1,
            require_nonzero=True,
        )

        self.assertEqual(result.solution_list, [1, 0])
        self.assertEqual(result.cost, 1.0)

    def test_matches_brute_force_on_small_random_instances(self):
        rng = random.Random(314159)

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
                    expected = _brute_force(q_matrix, dits)
                    result = solver_dynamic_programming3(
                        q_matrix,
                        [0.0] * len(q_matrix),
                        dits=dits,
                        n_neighbors=n_neighbors,
                    )

                    self.assertAlmostEqual(
                        result.cost,
                        qubo_value_from_lists(expected, q_matrix),
                    )

    def test_rejects_rows_wider_than_the_neighborhood(self):
        with self.assertRaisesRegex(ValueError, "representable with n_neighbors"):
            solver_dynamic_programming3(
                [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]],
                [0.0, 0.0, 0.0],
                dits=2,
                n_neighbors=1,
            )


if __name__ == "__main__":
    unittest.main()
