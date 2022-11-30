from typing import Any, Dict, Generic, NamedTuple, Optional, Tuple, TypeVar

import torch.nn as nn

from ..interpolation import LocalInterpolation
from ..problems import InitialValueProblem
from ..terms import ODETerm
from ..typing import *


class StepResult(NamedTuple):
    y: DataTensor
    error_estimate: Optional[DataTensor]


MethodState = TypeVar("MethodState")
InterpolationData = TypeVar("InterpolationData")


class SingleStepMethod(nn.Module, Generic[MethodState, InterpolationData]):
    def init(
        self,
        term: Optional[ODETerm],
        problem: InitialValueProblem,
        f0: Optional[DataTensor],
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> MethodState:
        raise NotImplementedError()

    def step(
        self,
        term: Optional[ODETerm],
        running: AcceptTensor,
        y0: DataTensor,
        t0: TimeTensor,
        dt: TimeTensor,
        state: MethodState,
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> Tuple[StepResult, InterpolationData, MethodState, Optional[StatusTensor]]:
        """Advance the solution from `y0` to `y0+dt`.

        Arguments
        ---------
        running
            Marks the instances in the batch that are actually still running. This is
            important for solvers with variable computation time such as implicit methods
            that use this information to short-circuit the evaluation of finished
            instances.
        y0
            Features at `t0`
        t0
            Initial point in time
        dt
            Step size of the step to make
        state
            Current state of the stepping method
        stats
            Tracked statistics for the current solve
        args
            Additional arguments for the ODE term

        Returns
        -------
        result
            Features `y1` at `t1 = t0 + dt` with an error estimate
        interpolation_data
            Additional information for interpolation between `t0` and `t1`
        state
            Updated state of the stepping method
        status
            Status to signify if integration should be stopped early (or None for
            all successes)
        """
        raise NotImplementedError()

    def merge_states(
        self, accept: AcceptTensor, current: MethodState, previous: MethodState
    ) -> MethodState:
        raise NotImplementedError()

    def build_interpolation(self, data: InterpolationData) -> LocalInterpolation:
        raise NotImplementedError()

    def convergence_order(self) -> int:
        raise NotImplementedError()
