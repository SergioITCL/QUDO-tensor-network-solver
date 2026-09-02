"""General-purpose QUDO solver using SCIP through PySCIPOpt.

This module only translates the compact QUDO representation into a bounded
integer nonlinear model.  Branch-and-bound, relaxations, presolve, cuts,
heuristics, and the treatment of the quadratic expressions are all left to
SCIP; no finite-neighborhood dynamic programming or QUBO encoding is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from numbers import Integral
from time import perf_counter
from typing import Any, Sequence

try:
    from pyscipopt import Model, SCIP_EVENTTYPE, quicksum
except ImportError as error:  # pragma: no cover - exercised without the extra dependency
    Model = None  # type: ignore[assignment]
    SCIP_EVENTTYPE = None  # type: ignore[assignment]
    quicksum = None  # type: ignore[assignment]
    _PYSCIPOPT_IMPORT_ERROR: ImportError | None = error
else:
    _PYSCIPOPT_IMPORT_ERROR = None

from qudo_solver.qudo_solver_core.solution import SolutionClass


# Change this locally when SCIP's detailed solving log is useful for debugging.
SCIP_OUTPUT_ENABLED = False
_RANDOM_SEED_PARAMETER = "randomization/randomseedshift"


@dataclass(frozen=True)
class SCIPMetadata:
    """Statistics reported by SCIP for an optimization run.

    ``solving_time`` is SCIP's internal optimization time.  In contrast,
    ``SolutionClass.execution_time`` includes validation, model construction,
    optimization, solution extraction, and result construction.
    """

    status: str
    solving_time: float
    nodes: int
    objective: float | None
    best_bound: float | None
    gap: float | None


@dataclass
class SCIPTargetResult:
    """Outcome of one continuous SCIP run searching for a target cost."""

    reached: bool
    target_cost: float
    time_to_target: float | None
    solution: SolutionClass | None
    best_cost: float | None
    total_execution_time: float
    metadata: SCIPMetadata
    incumbent_history: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class _TargetTracker:
    target_cost: float
    tolerance: float
    history: list[tuple[float, float]] = field(default_factory=list)
    best_cost: float | None = None
    best_values: list[int] | None = None
    target_values: list[int] | None = None
    reached: bool = False
    time_to_target: float | None = None
    callback_error: Exception | None = None


@dataclass
class _SCIPRun:
    solution: SolutionClass | None
    metadata: SCIPMetadata
    tracker: _TargetTracker | None


def _require_pyscipopt() -> None:
    if Model is None or quicksum is None:
        raise ImportError(
            "solver_scip requires PySCIPOpt. Install the project dependencies "
            "with `poetry install`, or install it directly with "
            "`pip install pyscipopt`."
        ) from _PYSCIPOPT_IMPORT_ERROR


def _validate_problem(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
) -> None:
    """Validate dimensions and coefficients of a compact QUDO instance."""
    if len(q_matrix) == 0:
        raise ValueError("q_matrix must contain at least one variable")
    if len(q_matrix) != len(q_row):
        raise ValueError("q_matrix and q_row must have the same length")
    if not isinstance(dits, Integral) or isinstance(dits, bool) or dits < 2:
        raise ValueError("dits must be an integer of at least 2")
    if (
        not isinstance(n_neighbors, Integral)
        or isinstance(n_neighbors, bool)
        or n_neighbors < 0
    ):
        raise ValueError("n_neighbors must be a non-negative integer")

    for i, (row, linear_coefficient) in enumerate(zip(q_matrix, q_row)):
        if len(row) == 0:
            raise ValueError(f"q_matrix[{i}] must not be empty")
        maximum_length = min(i, n_neighbors) + 1
        if len(row) > maximum_length:
            raise ValueError(
                f"q_matrix[{i}] contains {len(row) - 1} interactions, "
                f"but at most {maximum_length - 1} are allowed for "
                f"n_neighbors={n_neighbors}"
            )

        for coefficient in row:
            _validate_coefficient(coefficient, "q_matrix coefficients")
        _validate_coefficient(linear_coefficient, "q_row coefficients")


def _validate_coefficient(coefficient: float, description: str) -> None:
    try:
        finite = isfinite(float(coefficient))
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{description} must be numeric") from error
    if not finite:
        raise ValueError(f"{description} must be finite")


def _objective_energy(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    solution: Sequence[int],
) -> float:
    """Evaluate the complete compact QUDO objective externally."""
    if len(solution) != len(q_matrix) or len(q_row) != len(q_matrix):
        raise ValueError("q_matrix, q_row, and solution must have equal length")

    energy = 0.0
    for i, row in enumerate(q_matrix):
        value = solution[i]
        first_previous = i - len(row) + 1
        energy += float(row[-1]) * value * value
        energy += float(q_row[i]) * value
        energy += sum(
            float(coefficient) * solution[first_previous + offset] * value
            for offset, coefficient in enumerate(row[:-1])
        )
    return float(energy)


def _build_quadratic_expression(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    variables: Sequence[Any],
) -> Any:
    """Build C(x), including every compact coefficient exactly once."""
    terms: list[Any] = []
    for i, row in enumerate(q_matrix):
        diagonal = float(row[-1])
        if diagonal != 0.0:
            terms.append(diagonal * variables[i] * variables[i])

        linear = float(q_row[i])
        if linear != 0.0:
            terms.append(linear * variables[i])

        first_previous = i - len(row) + 1
        for offset, coefficient in enumerate(row[:-1]):
            coefficient = float(coefficient)
            if coefficient != 0.0:
                previous = first_previous + offset
                terms.append(coefficient * variables[previous] * variables[i])

    return quicksum(terms)


def _build_model(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    require_nonzero: bool,
) -> tuple[Any, list[Any]]:
    """Create the direct bounded-integer SCIP formulation."""
    model = Model("QUDO")
    if not SCIP_OUTPUT_ENABLED:
        model.hideOutput()

    variables = [
        model.addVar(vtype="I", lb=0, ub=dits - 1, name=f"x_{i}")
        for i in range(len(q_matrix))
    ]
    quadratic_expression = _build_quadratic_expression(
        q_matrix, q_row, variables
    )

    # PySCIPOpt's linear objective API is fed an auxiliary variable.  The
    # quadratic/nonlinear epigraph remains SCIP's responsibility.
    objective_var = model.addVar(
        vtype="C", lb=None, ub=None, name="objective"
    )
    model.addCons(
        objective_var >= quadratic_expression,
        name="quadratic_objective",
    )
    model.setObjective(objective_var, sense="minimize")

    if require_nonzero:
        model.addCons(quicksum(variables) >= 1, name="require_nonzero")

    return model, variables


def _validate_runtime_options(
    time_budget: float | None,
    seed: int | None,
    target_cost: float | None = None,
    target_tolerance: float | None = None,
) -> None:
    if time_budget is not None:
        try:
            numeric_time_budget = float(time_budget)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("time budget must be a positive finite number") from error
        if not isfinite(numeric_time_budget) or numeric_time_budget <= 0.0:
            raise ValueError("time budget must be a positive finite number")

    if seed is not None and (
        not isinstance(seed, Integral)
        or isinstance(seed, bool)
        or seed < 0
        or seed > 2_147_483_647
    ):
        raise ValueError("seed must be an integer between 0 and 2147483647")

    if target_cost is not None:
        _validate_coefficient(target_cost, "target_cost")
    if target_tolerance is not None:
        _validate_coefficient(target_tolerance, "target_tolerance")
        if float(target_tolerance) < 0.0:
            raise ValueError("target_tolerance must be non-negative")


def _metadata(model: Any, best_solution: Any | None) -> SCIPMetadata:
    objective = (
        float(model.getSolObjVal(best_solution))
        if best_solution is not None
        else None
    )
    return SCIPMetadata(
        status=str(model.getStatus()),
        solving_time=float(model.getSolvingTime()),
        nodes=int(model.getNTotalNodes()),
        objective=objective,
        best_bound=float(model.getDualbound()),
        gap=float(model.getGap()),
    )


def _budget_exhausted_metadata() -> SCIPMetadata:
    """Represent a call whose wall-clock budget expired before optimize()."""
    return SCIPMetadata(
        status="budget_exhausted_before_optimize",
        solving_time=0.0,
        nodes=0,
        objective=None,
        best_bound=None,
        gap=None,
    )


def _extract_values(
    model: Any,
    scip_solution: Any,
    variables: Sequence[Any],
    dits: int,
    require_nonzero: bool,
) -> list[int]:
    """Round and validate one SCIP solution in the original integer domain."""
    values: list[int] = []
    for i, variable in enumerate(variables):
        value = int(round(float(model.getSolVal(scip_solution, variable))))
        if not 0 <= value < dits:
            raise RuntimeError(
                f"SCIP returned x_{i}={value}, outside [0, {dits - 1}]"
            )
        values.append(value)

    if require_nonzero and not any(values):
        raise RuntimeError("SCIP returned a solution violating sum(x) >= 1")
    return values


def _make_solution(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    values: list[int],
    dits: int,
) -> SolutionClass:
    return SolutionClass(
        solution_list=values,
        dits=dits,
        cost=_objective_energy(q_matrix, q_row, values),
        execution_time=0.0,
    )


def _attach_target_handler(
    model: Any,
    variables: Sequence[Any],
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    require_nonzero: bool,
    started_at: float,
    tracker: _TargetTracker,
) -> None:
    """Record externally improving incumbents and interrupt at the target."""

    def on_best_solution_found(event_model: Any, event: Any) -> None:
        del event
        try:
            incumbent = event_model.getBestSol()
            if incumbent is None:
                return
            values = _extract_values(
                event_model, incumbent, variables, dits, require_nonzero
            )
            cost = _objective_energy(q_matrix, q_row, values)

            # BESTSOLFOUND concerns SCIP's auxiliary z objective.  Keep only
            # strict improvements under the externally evaluated QUDO cost.
            if tracker.best_cost is not None and cost >= tracker.best_cost:
                return

            elapsed = perf_counter() - started_at
            tracker.best_cost = cost
            tracker.best_values = values
            tracker.history.append((elapsed, cost))

            if cost <= tracker.target_cost + tracker.tolerance:
                tracker.reached = True
                tracker.time_to_target = elapsed
                tracker.target_values = values
                event_model.interruptSolve()
        except Exception as error:  # callbacks cannot reliably propagate errors
            tracker.callback_error = error
            event_model.interruptSolve()

    model.attachEventHandlerCallback(
        on_best_solution_found,
        [SCIP_EVENTTYPE.BESTSOLFOUND],
        name="qudo_target_recorder",
        description="Record improving QUDO incumbents and stop at target",
    )


def _solve_scip(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
    time_budget: float | None,
    require_nonzero: bool,
    seed: int | None,
    started_at: float,
    target_cost: float | None = None,
    target_tolerance: float | None = None,
) -> _SCIPRun:
    _require_pyscipopt()
    _validate_problem(q_matrix, q_row, dits, n_neighbors)
    _validate_runtime_options(
        time_budget, seed, target_cost, target_tolerance
    )

    # Model construction time is part of total time, but not SCIP's internal
    # solving time reported by model.getSolvingTime().
    model, variables = _build_model(q_matrix, q_row, int(dits), require_nonzero)
    if seed is not None:
        if _RANDOM_SEED_PARAMETER not in model.getParams():
            raise RuntimeError(
                "This SCIP version does not expose the official "
                f"{_RANDOM_SEED_PARAMETER!r} parameter"
            )
        model.setParam(_RANDOM_SEED_PARAMETER, int(seed))

    tracker: _TargetTracker | None = None
    if target_cost is not None:
        if target_tolerance is None:
            raise RuntimeError("target_tolerance is required in target mode")
        tracker = _TargetTracker(
            target_cost=float(target_cost),
            tolerance=float(target_tolerance),
        )
        _attach_target_handler(
            model,
            variables,
            q_matrix,
            q_row,
            int(dits),
            require_nonzero,
            started_at,
            tracker,
        )

    # The public budget covers validation and model building too.  SCIP gets
    # only what remains; stopping and extraction add a small unavoidable
    # wall-clock overhead after its internal timer expires.
    if time_budget is None:
        model.optimize()
    else:
        remaining_time = float(time_budget) - (perf_counter() - started_at)
        if remaining_time <= 0.0:
            return _SCIPRun(
                solution=None,
                metadata=_budget_exhausted_metadata(),
                tracker=tracker,
            )
        model.setParam("limits/time", remaining_time)
        model.optimize()

    if tracker is not None and tracker.callback_error is not None:
        raise RuntimeError("SCIP incumbent callback failed") from tracker.callback_error

    # A time-limit status can still have a valid incumbent.
    best_solution = model.getBestSol()
    run_metadata = _metadata(model, best_solution)
    if tracker is not None and tracker.reached:
        values = tracker.target_values
    elif tracker is not None and tracker.best_values is not None:
        # The auxiliary z incumbent and the externally best QUDO assignment can
        # differ while z still has slack, so preserve the latter in target mode.
        values = tracker.best_values
    elif best_solution is not None:
        values = _extract_values(
            model, best_solution, variables, int(dits), require_nonzero
        )
    else:
        values = None

    result = (
        _make_solution(q_matrix, q_row, values, int(dits))
        if values is not None
        else None
    )
    return _SCIPRun(solution=result, metadata=run_metadata, tracker=tracker)


def solver_scip(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
    time_limit: float | None = None,
    require_nonzero: bool = False,
    seed: int | None = None,
) -> SolutionClass:
    """Return SCIP's best solution within an approximate total wall budget.

    Model construction is subtracted from ``time_limit`` before configuring
    SCIP.  Solution extraction can add a small unavoidable stopping overhead.
    """
    # Start before validation and model construction for cross-solver timing.
    started_at = perf_counter()
    run = _solve_scip(
        q_matrix,
        q_row,
        dits,
        n_neighbors,
        time_limit,
        require_nonzero,
        seed,
        started_at,
    )
    if run.solution is None:
        raise RuntimeError(
            "SCIP finished with status "
            f"{run.metadata.status!r} without finding a feasible solution"
        )
    run.solution.execution_time = perf_counter() - started_at
    return run.solution


def solver_scip_with_metadata(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
    time_limit: float | None = None,
    require_nonzero: bool = False,
    seed: int | None = None,
) -> tuple[SolutionClass, SCIPMetadata]:
    """Solve QUDO and also return SCIP statistics useful for experiments."""
    started_at = perf_counter()
    run = _solve_scip(
        q_matrix,
        q_row,
        dits,
        n_neighbors,
        time_limit,
        require_nonzero,
        seed,
        started_at,
    )
    if run.solution is None:
        raise RuntimeError(
            "SCIP finished with status "
            f"{run.metadata.status!r} without finding a feasible solution"
        )
    run.solution.execution_time = perf_counter() - started_at
    return run.solution, run.metadata


def solver_scip_time_to_target(
    q_matrix: Sequence[Sequence[float]],
    q_row: Sequence[float],
    dits: int,
    n_neighbors: int,
    target_cost: float,
    max_time: float = 60.0,
    require_nonzero: bool = False,
    seed: int | None = None,
    target_tolerance: float = 1e-9,
) -> SCIPTargetResult:
    """Measure wall time until SCIP first reaches an external QUDO target.

    One continuous SCIP run is monitored with ``BESTSOLFOUND``.  A target is
    reached when the externally recomputed cost is at most
    ``target_cost + target_tolerance``.
    """
    started_at = perf_counter()
    run = _solve_scip(
        q_matrix,
        q_row,
        dits,
        n_neighbors,
        max_time,
        require_nonzero,
        seed,
        started_at,
        target_cost=target_cost,
        target_tolerance=target_tolerance,
    )
    tracker = run.tracker
    if tracker is None:  # Internal invariant: target_cost always creates it.
        raise RuntimeError("SCIP target tracker was not initialized")

    result = SCIPTargetResult(
        reached=tracker.reached,
        target_cost=float(target_cost),
        time_to_target=tracker.time_to_target,
        solution=run.solution,
        best_cost=run.solution.cost if run.solution is not None else None,
        total_execution_time=0.0,
        metadata=run.metadata,
        incumbent_history=list(tracker.history),
    )
    total_execution_time = perf_counter() - started_at
    result.total_execution_time = total_execution_time
    if result.solution is not None:
        result.solution.execution_time = total_execution_time
    return result
