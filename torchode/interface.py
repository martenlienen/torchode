from typing import Any, Callable, Optional, Tuple, Union

import torch

from .adjoints import AutoDiffAdjoint
from .problems import InitialValueProblem
from .single_step_methods import SingleStepMethod
from .solution import Solution
from .step_size_controllers import (
    FixedStepController,
    PIDController,
    StepSizeController,
)
from .terms import ODETerm
from .typing import *

METHODS = {}


def register_method(name: str, constructor: Callable[[ODETerm], SingleStepMethod]):
    METHODS[name] = constructor


def solve_ivp(
    f: Union[ODETerm, Callable[[TimeTensor, DataTensor], DataTensor]],
    y0: DataTensor,
    t_eval: EvaluationTimesTensor,
    *,
    t_span: Optional[Tuple[TimeTensor, TimeTensor]] = None,
    method: Union[str, SingleStepMethod] = "tsit5",
    max_steps: Optional[int] = None,
    controller: Optional[StepSizeController] = None,
    dt0: Optional[TimeTensor] = None,
    args: Any = None,
) -> Solution:
    """Solve an initial value problem

    Arguments
    =========
    f
        The dynamics to solve
    y0
        Initial conditions
    t_eval
        Time points to evaluate the solution at
    t_span
        Start and end times of the integration. By default, integrate from the first to
        the last evaluation point.
    method
        Either the name of a registered stepping method, e.g. `"tsit5"`, or a stepping
        method object
    max_steps
        Stop the solver after this many steps
    controller
        Step size controller for the integration. By default a PID controller with
        `atol=1e-7, rtol=1e-7, pcoeff=0.2, icoeff=0.5, dcoeff=0.0` will be
        constructed.
    dt0
        An optional initial time step
    args
        Extra arguments to be passed to the integration term
    """

    if isinstance(f, ODETerm):
        term = f
    else:
        if args is not None:
            term = ODETerm(f, with_args=True)
        else:
            term = ODETerm(f)

    # TODO: Automatically reshape y0 into [batch, features] and back into its original
    # shape

    if not isinstance(method, SingleStepMethod):
        method = METHODS[method](term=term)

    if controller is None:
        controller = PIDController(
            term=term, atol=1e-7, rtol=1e-7, pcoeff=0.2, icoeff=0.5, dcoeff=0.0
        )

    adjoint = AutoDiffAdjoint(method, controller, max_steps=max_steps)

    batch_size = y0.shape[0]
    if t_eval is not None and t_eval.ndim == 1:
        t_eval = t_eval.expand((batch_size, -1))
    if t_span is not None:
        t_start, t_end = t_span
    else:
        t_start, t_end = t_eval[:, 0], t_eval[:, -1]
    if t_start.ndim == 0:
        t_start = t_start.expand(batch_size)
    if t_end.ndim == 0:
        t_end = t_end.expand(batch_size)
    problem = InitialValueProblem(y0, t_start, t_end, t_eval)
    return adjoint.solve(problem, term, dt0=dt0, args=args)
