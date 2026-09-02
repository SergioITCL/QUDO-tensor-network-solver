import itertools
import random
import unittest

from qudo_solver.auxiliar_functions import qudo_value
from qudo_solver.solvers.transfer_matrix import solver_transfer_matrix


def _brute_force(q_matrix, q_row, dits, require_nonzero):
    assignments = itertools.product(range(dits), repeat=len(q_matrix))
    if require_nonzero:
        assignments = (assignment for assignment in assignments if any(assignment))
    return min(
        assignments,
        key=lambda assignment: qudo_value(assignment, q_matrix, q_row),
    )


class TransferMatrixSolverTests(unittest.TestCase):
    def test_enforces_nonzero_constraint_by_default(self):
        result = solver_transfer_matrix(
            [[1.0], [0.0, 2.0], [0.0, 0.0, 3.0]],
            [0.0, 0.0, 0.0],
            dits=2,
            n_neighbors=2,
        )

        self.assertEqual(result.solution_list, [1, 0, 0])
        self.assertEqual(result.cost, 1.0)

    def test_supports_the_unconstrained_problem_and_linear_terms(self):
        result = solver_transfer_matrix(
            [[2.0], [0.5, 3.0]],
            [-5.0, 1.0],
            dits=3,
            n_neighbors=1,
            require_nonzero=False,
        )

        self.assertEqual(result.solution_list, [1, 0])
        self.assertEqual(result.cost, -3.0)

    def test_matches_brute_force_on_random_instances(self):
        rng = random.Random(161803)

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
                    q_row = [rng.uniform(-2.0, 2.0) for _ in q_matrix]

                    for require_nonzero in (False, True):
                        expected = _brute_force(
                            q_matrix, q_row, dits, require_nonzero
                        )
                        result = solver_transfer_matrix(
                            q_matrix,
                            q_row,
                            dits=dits,
                            n_neighbors=n_neighbors,
                            require_nonzero=require_nonzero,
                        )
                        self.assertAlmostEqual(
                            result.cost,
                            qudo_value(expected, q_matrix, q_row),
                        )

    def test_rejects_rows_wider_than_the_transfer_state(self):
        with self.assertRaisesRegex(ValueError, "can represent"):
            solver_transfer_matrix(
                [[1.0], [2.0, 3.0], [4.0, 5.0, 6.0]],
                [0.0, 0.0, 0.0],
                dits=2,
                n_neighbors=1,
            )


if __name__ == "__main__":
    unittest.main()
