"""Common solution model returned by every solver."""

from typing import List

from pydantic import BaseModel

from qudo_solver.auxiliar_functions import qudo_value


class SolutionClass(BaseModel):
    solution_list: List[int]
    dits: int
    cost: float
    execution_time: float

    @classmethod
    def from_solution_list(
        cls,
        qudo_instance_matrix: List[List[float]],
        qudo_instance_row: List[float],
        solution_list: List[int],
        dits: int,
        execution_time: float,
    ) -> "SolutionClass":
        """Build a result and evaluate its complete quadratic-linear cost."""
        return cls(
            solution_list=solution_list,
            dits=dits,
            cost=qudo_value(
                solution_list,
                qudo_instance_matrix,
                qudo_instance_row,
            ),
            execution_time=execution_time,
        )
