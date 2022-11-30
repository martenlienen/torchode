from math import sqrt
from typing import Any, Callable, Dict, Generic, NamedTuple, Optional, Tuple, TypeVar

import torch
import torch.nn as nn

from .problems import InitialValueProblem
from .single_step_methods import StepResult
from .status_codes import Status
from .terms import ODETerm
from .typing import *

ControllerState = TypeVar("ControllerState")


class StepSizeController(nn.Module, Generic[ControllerState]):
    """A step size controller determines the size of integration steps."""

    def init(
        self,
        term: Optional[ODETerm],
        problem: InitialValueProblem,
        method_order: int,
        dt0: Optional[TimeTensor],
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> Tuple[TimeTensor, ControllerState, Optional[DataTensor]]:
        """Find the initial step size and initialize the controller state

        If the user suggests an initial step size, the controller should go with that
        one.

        If the controller evaluates the vector field at the initial step, for example
        to determine the initial step size, the evaluation at `t0` can be returned to
        save an evaluation in FSAL step methods.

        Arguments
        ---------
        term
            The integration term
        problem
            The problem to solve
        method_order
            Convergence order of the stepping method
        dt0
            An initial step size suggested by the user
        stats
            Container that the controller can initialize new statistics to track in
        args
            Static arguments for calls to the ODE term
        """
        raise NotImplementedError()

    def adapt_step_size(
        self,
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        step_result: StepResult,
        state: ControllerState,
        stats: Dict[str, Any],
    ) -> Tuple[AcceptTensor, TimeTensor, ControllerState, Optional[StatusTensor]]:
        """Adapt the integration step size based on the step just taken

        Arguments
        ---------
        t0
            Start time of the step
        dt
            Current step size
        y0
            State before the previous step
        step_result
            Result of the step just taken
        state
            Current controller state
        stats
            Tracked statistics for the current solve which can be updated in-place

        Returns
        -------
        accept
            Should the step be accepted or rejected?
        dt
            Next step size, either for the next step or to retry the current step
            if it was rejected
        state
            Next controller state
        status
            Status to signify if integration should be stopped early (or None for
            all sucesses)
        """
        raise NotImplementedError()

    def merge_states(
        self, running: AcceptTensor, current: ControllerState, previous: ControllerState
    ) -> ControllerState:
        """Merge two controller states

        Any batch-specific state should be updated so that it updates only for the still
        running instances and stays constant for finished instances.

        Arguments
        ---------
        running
            Marks the instances in the batch that are still being solved
        current
            The controller state at the end of the current iteration
        previous
            The previous controller state
        """
        raise NotImplementedError()


class FixedStepState(NamedTuple):
    accept_all: AcceptTensor
    dt0: TimeTensor


class FixedStepController(StepSizeController[FixedStepState]):
    """A fixed-step step size controller.

    Does not actually control anything. Just accepts any result and keeps the step size
    fixed.
    """

    @torch.jit.export
    def init(
        self,
        term: Optional[ODETerm],
        problem: InitialValueProblem,
        method_order: int,
        dt0: Optional[TimeTensor],
        *,
        stats: Dict[str, Any],
        args: Any,
    ):
        assert dt0 is not None, "Fixed step size solving requires you to configure dt0"
        return (
            dt0,
            FixedStepState(
                accept_all=torch.ones(
                    problem.batch_size, device=problem.device, dtype=torch.bool
                ),
                dt0=dt0,
            ),
            None,
        )

    @torch.jit.export
    def adapt_step_size(
        self,
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        step_result: StepResult,
        state: FixedStepState,
        stats: Dict[str, Any],
    ) -> Tuple[AcceptTensor, TimeTensor, FixedStepState, Optional[StatusTensor]]:
        return state.accept_all, state.dt0, state, None

    @torch.jit.export
    def merge_states(
        self, running: AcceptTensor, current: FixedStepState, previous: FixedStepState
    ) -> FixedStepState:
        return current


def rms_norm(y: DataTensor) -> NormTensor:
    """Root mean squared error norm.

    As suggested in [1], Equation (4.11).

    References
    ----------
    .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential
           Equations I: Nonstiff Problems", Sec. II.4, 2nd edition.
    """
    # `vector_norm` autmatically deals with complex vectors correctly
    return torch.linalg.vector_norm(y / sqrt(y.shape[1]), ord=2, dim=1)


def max_norm(y: DataTensor) -> NormTensor:
    """Maximums norm."""
    return torch.linalg.vector_norm(y, dim=1, ord=torch.inf)


class AdaptiveStepSizeController(
    StepSizeController[ControllerState], Generic[ControllerState]
):
    def initial_state(
        self,
        method_order: int,
        problem: InitialValueProblem,
        dt_min: Optional[TimeTensor],
        dt_max: Optional[TimeTensor],
    ) -> ControllerState:
        raise NotImplementedError()

    def update_state(
        self,
        state: ControllerState,
        y0: DataTensor,
        dt: TimeTensor,
        error_ratio: Optional[NormTensor],
        accept: Optional[AcceptTensor],
    ) -> ControllerState:
        raise NotImplementedError()

    def dt_factor(self, state: ControllerState, error_ratio: NormTensor):
        raise NotImplementedError()


class IntegralState:
    def __init__(
        self,
        method_order: int,
        almost_zero: torch.Tensor,
        dt_min: Optional[torch.Tensor] = None,
        dt_max: Optional[torch.Tensor] = None,
    ):
        self.method_order = method_order
        self.almost_zero = almost_zero
        self.dt_min = dt_min
        self.dt_max = dt_max

    def update_error_ratios(
        self, prev_error_ratio: NormTensor, prev_prev_error_ratio: NormTensor
    ):
        return self

    def __repr__(self):
        return (
            f"IState(method_order={self.method_order}, "
            f"almost_zero={self.almost_zero}, "
            f"dt_min={self.dt_min}), "
            f"dt_max={self.dt_max})"
        )

    @staticmethod
    def default(
        *,
        method_order: int,
        batch_size: int,
        dtype: torch.dtype,
        device: Optional[torch.device],
        dt_min: Optional[torch.Tensor],
        dt_max: Optional[torch.Tensor],
    ):
        # Pre-allocate a fixed, very small number as a lower bound for the error ratio
        if dtype == torch.float16:
            float_min = 1e-5
        else:
            float_min = 1e-38
        almost_zero = torch.tensor(float_min, dtype=dtype, device=device)
        return IntegralState(method_order, almost_zero, dt_min, dt_max)


class IntegralController(nn.Module):
    """The simplest controller that scales the step size proportional to the error."""

    def __init__(
        self,
        atol: float,
        rtol: float,
        *,
        term: Optional[ODETerm] = None,
        norm: Callable[[DataTensor], NormTensor] = rms_norm,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
        safety: float = 0.9,
        factor_min: float = 0.2,
        factor_max: float = 10.0,
    ):
        super().__init__()

        self.register_buffer("atol", torch.tensor(atol))
        self.register_buffer("rtol", torch.tensor(rtol))
        self.term = term
        self.norm = norm
        self.dt_min = dt_min
        self.dt_max = dt_max

        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max

    def dt_factor(self, state: IntegralState, error_ratio: NormTensor):
        """Compute the growth factor of the timestep."""

        k_I = 1.0 / state.method_order
        factor = self.safety * error_ratio ** (-k_I)
        return torch.clamp(factor, min=self.factor_min, max=self.factor_max)

    def initial_state(
        self,
        method_order: int,
        problem: InitialValueProblem,
        dt_min: Optional[TimeTensor],
        dt_max: Optional[TimeTensor],
    ) -> IntegralState:
        return IntegralState.default(
            method_order=method_order,
            batch_size=problem.batch_size,
            dtype=problem.data_dtype,
            device=problem.device,
            dt_min=dt_min,
            dt_max=dt_max,
        )

    @torch.jit.export
    def merge_states(
        self, running: AcceptTensor, current: IntegralState, previous: IntegralState
    ) -> IntegralState:
        return current

    def update_state(
        self,
        state: IntegralState,
        y0: DataTensor,
        dt: TimeTensor,
        error_ratio: Optional[NormTensor],
        accept: Optional[AcceptTensor],
    ) -> IntegralState:
        return state

    ################################################################################
    # The following methods should be on AdaptiveStepSizeController if TorchScript #
    # supports inheritance at some point                                           #
    ################################################################################

    @torch.jit.export
    def init(
        self,
        term: Optional[ODETerm],
        problem: InitialValueProblem,
        method_order: int,
        dt0: Optional[TimeTensor],
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> Tuple[TimeTensor, IntegralState, Optional[DataTensor]]:
        if dt0 is None:
            dt0, f0 = self._select_initial_step(
                term,
                problem.t_start,
                problem.y0,
                problem.time_direction,
                method_order,
                stats,
                args,
            )
        else:
            f0 = None
        dt_min = self.dt_min
        if dt_min is not None:
            dt_min = torch.tensor(
                dt_min, dtype=problem.time_dtype, device=problem.device
            )
        dt_max = self.dt_max
        if dt_max is not None:
            dt_max = torch.tensor(
                dt_max, dtype=problem.time_dtype, device=problem.device
            )
        return dt0, self.initial_state(method_order, problem, dt_min, dt_max), f0

    @torch.jit.export
    def adapt_step_size(
        self,
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        step_result: StepResult,
        state: IntegralState,
        stats: Dict[str, Any],
    ) -> Tuple[AcceptTensor, TimeTensor, IntegralState, Optional[StatusTensor]]:
        y1, error_estimate = step_result.y, step_result.error_estimate

        if error_estimate is None:
            # If the stepping method could not provide an error estimate, we interpret
            # this as an error estimate that gets the step accepted without changing the
            # step size, i.e. as an error ratio of 1 (disregarding the safety factor).
            return (
                torch.ones_like(dt, dtype=torch.bool),
                dt,
                self.update_state(state, y0, dt, None, None),
                None,
            )

        # Compute error ratio and decide on step acceptance
        error_bounds = torch.add(
            self.atol, torch.maximum(y0.abs(), y1.abs()), alpha=self.rtol
        )
        error = error_estimate.abs()
        # We lower-bound the error ratio by some small number to avoid division by 0 in
        # `dt_factor`.
        error_ratio = torch.maximum(self.norm(error / error_bounds), state.almost_zero)
        accept = error_ratio < 1.0

        # Adapt the step size
        dt_next = dt * self.dt_factor(state, error_ratio).to(dtype=dt.dtype)

        # Check for infinities and NaN
        status = torch.where(
            torch.isfinite(error_ratio),
            Status.SUCCESS.value,
            Status.INFINITE_NORM.value,
        )

        # Enforce the minimum and maximum step size
        dt_min = state.dt_min
        dt_max = state.dt_max
        if dt_min is not None or dt_max is not None:
            abs_dt_next = dt_next.abs()
            dt_next = torch.sign(dt_next) * torch.clamp(abs_dt_next, dt_min, dt_max)
            if dt_min is not None:
                status = torch.where(
                    abs_dt_next < dt_min, Status.REACHED_DT_MIN.value, status
                )

        return (
            accept,
            dt_next,
            self.update_state(state, y0, dt, error_ratio, accept),
            status,
        )

    def _select_initial_step(
        self,
        term: Optional[ODETerm],
        t0: TimeTensor,
        y0: DataTensor,
        direction: torch.Tensor,
        convergence_order: int,
        stats: Dict[str, Any],
        args: Any,
    ) -> Tuple[TimeTensor, DataTensor]:
        """Empirically select a good initial step.

        This is an adaptation of the algorithm described in [1]_. We changed it in such a
        way that the tolerances apply to the norms instead of the components of `y`.

        References
        ----------
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations
        I: Nonstiff Problems", Sec. II.4, 2nd edition.
        """

        if torch.jit.is_scripting() or term is None:
            assert term is None, "The integration term is fixed for JIT compilation"
            term = self.term
        assert term is not None

        norm = self.norm
        f0 = term.vf(t0, y0, stats, args)

        error_bounds = torch.add(self.atol, torch.abs(y0), alpha=self.rtol)
        inv_scale = torch.reciprocal(error_bounds)

        d0 = norm(y0 * inv_scale)
        d1 = norm(f0 * inv_scale)

        small_number = torch.tensor(1e-6, dtype=d0.dtype, device=d0.device)
        dt0 = torch.where((d0 < 1e-5) | (d1 < 1e-5), small_number, 0.01 * d0 / d1)

        y1 = torch.addcmul(y0, (direction.to(dtype=dt0.dtype) * dt0)[:, None], f0)
        f1 = term.vf(
            torch.addcmul(t0, direction.to(dtype=t0.dtype), dt0.to(dtype=t0.dtype)),
            y1,
            stats,
            args,
        )

        d2 = norm((f1 - f0) * inv_scale) / dt0

        maxd1d2 = torch.maximum(d1, d2)
        dt1 = torch.where(
            maxd1d2 <= 1e-15,
            torch.maximum(small_number, dt0 * 1e-3),
            (0.01 / maxd1d2) ** (1.0 / convergence_order),
        )

        return (direction * torch.minimum(100 * dt0, dt1)).to(dtype=t0.dtype), f0


class PIDState:
    def __init__(
        self,
        method_order: int,
        prev_error_ratio: NormTensor,
        prev_prev_error_ratio: NormTensor,
        almost_zero: torch.Tensor,
        dt_min: Optional[torch.Tensor] = None,
        dt_max: Optional[torch.Tensor] = None,
    ):
        self.method_order = method_order
        self.prev_error_ratio = prev_error_ratio
        self.prev_prev_error_ratio = prev_prev_error_ratio
        self.almost_zero = almost_zero
        self.dt_min = dt_min
        self.dt_max = dt_max

    def update_error_ratios(
        self, prev_error_ratio: NormTensor, prev_prev_error_ratio: NormTensor
    ):
        return PIDState(
            self.method_order,
            prev_error_ratio,
            prev_prev_error_ratio,
            self.almost_zero,
            self.dt_min,
            self.dt_max,
        )

    def __repr__(self):
        return (
            f"PIDState(method_order={self.method_order}, "
            f"prev_error_ratio={self.prev_error_ratio}), "
            f"prev_prev_error_ratio={self.prev_prev_error_ratio}, "
            f"almost_zero={self.almost_zero}, "
            f"dt_min={self.dt_min}), "
            f"dt_max={self.dt_max})"
        )

    @staticmethod
    def default(
        *,
        method_order: int,
        batch_size: int,
        dtype: torch.dtype,
        device: Optional[torch.device],
        dt_min: Optional[torch.Tensor],
        dt_max: Optional[torch.Tensor],
    ):
        default_ratio = torch.ones(batch_size, dtype=dtype, device=device)
        # Pre-allocate a fixed, very small number as a lower bound for the error ratio
        if dtype == torch.float16:
            float_min = 1e-5
        else:
            float_min = 1e-38
        almost_zero = torch.tensor(float_min, dtype=dtype, device=device)
        return PIDState(
            method_order, default_ratio, default_ratio, almost_zero, dt_min, dt_max
        )


class PIDController(nn.Module):
    """A PID step size controller.

    The formula for the dt scaling factor with PID control is taken from [1], Equation
    (34).

    References
    ----------
    [1] Söderlind, G. (2003). Digital Filters in Adaptive Time-Stepping. ACM
        Transactions on Mathematical Software, 29, 1–26.
    """

    def __init__(
        self,
        atol: float,
        rtol: float,
        pcoeff: float,
        icoeff: float,
        dcoeff: float,
        *,
        term: Optional[ODETerm] = None,
        norm: Callable[[DataTensor], NormTensor] = rms_norm,
        dt_min: Optional[float] = None,
        dt_max: Optional[float] = None,
        safety: float = 0.9,
        factor_min: float = 0.2,
        factor_max: float = 10.0,
    ):
        super().__init__()

        self.register_buffer("atol", torch.tensor(atol))
        self.register_buffer("rtol", torch.tensor(rtol))
        self.term = term
        self.norm = norm
        self.dt_min = dt_min
        self.dt_max = dt_max

        self.pcoeff = pcoeff
        self.icoeff = icoeff
        self.dcoeff = dcoeff
        self.safety = safety
        self.factor_min = factor_min
        self.factor_max = factor_max

    def dt_factor(self, state: PIDState, error_ratio: NormTensor):
        """Compute the growth factor of the timestep."""

        # This is an instantiation of Equation (34) in the Söderlind paper where we have
        # factored out the safety coefficient. I have not found a reference for dividing
        # the PID coefficients by the order of the solver but DifferentialEquations.jl
        # and diffrax both do it, so we do it too. Note that our error ratio is the
        # reciprocal of Söderlind's error ratio (except for the safety factor).
        # Therefore, the factor exponents have the opposite sign from the paper.
        #
        # Interesting thing from the introduction of that paper is that you work with p
        # if you want per-step-error-control and p+1 if you want
        # per-unit-step-error-control where p is the convergence order of the stepping
        # method.
        order = state.method_order
        k_I, k_P, k_D = self.icoeff / order, self.pcoeff / order, self.dcoeff / order

        factor1 = error_ratio ** (-(k_I + k_P + k_D))
        factor2 = state.prev_error_ratio ** (k_P + 2 * k_D)
        factor3 = state.prev_prev_error_ratio**-k_D
        factor = self.safety * factor1 * factor2 * factor3

        return torch.clamp(factor, min=self.factor_min, max=self.factor_max)

    def initial_state(
        self,
        method_order: int,
        problem: InitialValueProblem,
        dt_min: Optional[TimeTensor],
        dt_max: Optional[TimeTensor],
    ) -> PIDState:
        return PIDState.default(
            method_order=method_order,
            batch_size=problem.batch_size,
            dtype=problem.data_dtype,
            device=problem.device,
            dt_min=dt_min,
            dt_max=dt_max,
        )

    @torch.jit.export
    def merge_states(
        self, running: AcceptTensor, current: PIDState, previous: PIDState
    ) -> PIDState:
        return current.update_error_ratios(
            torch.where(running, current.prev_error_ratio, previous.prev_error_ratio),
            torch.where(
                running, current.prev_prev_error_ratio, previous.prev_prev_error_ratio
            ),
        )

    def update_state(
        self,
        state: PIDState,
        y0: DataTensor,
        dt: TimeTensor,
        error_ratio: Optional[NormTensor],
        accept: Optional[AcceptTensor],
    ) -> PIDState:
        if error_ratio is None:
            return state.update_error_ratios(
                prev_error_ratio=y0.new_ones(dt.shape),
                prev_prev_error_ratio=state.prev_error_ratio,
            )
        else:
            assert accept is not None
            return state.update_error_ratios(
                prev_error_ratio=torch.where(
                    accept, error_ratio, state.prev_error_ratio
                ),
                prev_prev_error_ratio=torch.where(
                    accept, state.prev_error_ratio, state.prev_prev_error_ratio
                ),
            )

    ################################################################################
    # The following methods should be on AdaptiveStepSizeController if TorchScript #
    # supports inheritance at some point                                           #
    ################################################################################

    @torch.jit.export
    def init(
        self,
        term: Optional[ODETerm],
        problem: InitialValueProblem,
        method_order: int,
        dt0: Optional[TimeTensor],
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> Tuple[TimeTensor, PIDState, Optional[DataTensor]]:
        if dt0 is None:
            dt0, f0 = self._select_initial_step(
                term,
                problem.t_start,
                problem.y0,
                problem.time_direction,
                method_order,
                stats,
                args,
            )
        else:
            f0 = None
        dt_min = self.dt_min
        if dt_min is not None:
            dt_min = torch.tensor(
                dt_min, dtype=problem.time_dtype, device=problem.device
            )
        dt_max = self.dt_max
        if dt_max is not None:
            dt_max = torch.tensor(
                dt_max, dtype=problem.time_dtype, device=problem.device
            )
        return dt0, self.initial_state(method_order, problem, dt_min, dt_max), f0

    @torch.jit.export
    def adapt_step_size(
        self,
        t0: TimeTensor,
        dt: TimeTensor,
        y0: DataTensor,
        step_result: StepResult,
        state: PIDState,
        stats: Dict[str, Any],
    ) -> Tuple[AcceptTensor, TimeTensor, PIDState, Optional[StatusTensor]]:
        y1, error_estimate = step_result.y, step_result.error_estimate

        if error_estimate is None:
            # If the stepping method could not provide an error estimate, we interpret
            # this as an error estimate that gets the step accepted without changing the
            # step size, i.e. as an error ratio of 1 (disregarding the safety factor).
            return (
                torch.ones_like(dt, dtype=torch.bool),
                dt,
                self.update_state(state, y0, dt, None, None),
                None,
            )

        # Compute error ratio and decide on step acceptance
        error_bounds = torch.add(
            self.atol, torch.maximum(y0.abs(), y1.abs()), alpha=self.rtol
        )
        error = error_estimate.abs()
        # We lower-bound the error ratio by some small number to avoid division by 0 in
        # `dt_factor`.
        error_ratio = torch.maximum(self.norm(error / error_bounds), state.almost_zero)
        accept = error_ratio < 1.0

        # Adapt the step size
        dt_next = dt * self.dt_factor(state, error_ratio).to(dtype=dt.dtype)

        # Check for infinities and NaN
        status = torch.where(
            torch.isfinite(error_ratio),
            Status.SUCCESS.value,
            Status.INFINITE_NORM.value,
        )

        # Enforce the minimum and maximum step size
        dt_min = state.dt_min
        dt_max = state.dt_max
        if dt_min is not None or dt_max is not None:
            abs_dt_next = dt_next.abs()
            dt_next = torch.sign(dt_next) * torch.clamp(abs_dt_next, dt_min, dt_max)
            if dt_min is not None:
                status = torch.where(
                    abs_dt_next < dt_min, Status.REACHED_DT_MIN.value, status
                )

        return (
            accept,
            dt_next,
            self.update_state(state, y0, dt, error_ratio, accept),
            status,
        )

    def _select_initial_step(
        self,
        term: Optional[ODETerm],
        t0: TimeTensor,
        y0: DataTensor,
        direction: torch.Tensor,
        convergence_order: int,
        stats: Dict[str, Any],
        args: Any,
    ) -> Tuple[TimeTensor, DataTensor]:
        """Empirically select a good initial step.

        This is an adaptation of the algorithm described in [1]_. We changed it in such a
        way that the tolerances apply to the norms instead of the components of `y`.

        References
        ----------
        .. [1] E. Hairer, S. P. Norsett G. Wanner, "Solving Ordinary Differential Equations
        I: Nonstiff Problems", Sec. II.4, 2nd edition.
        """

        if torch.jit.is_scripting() or term is None:
            assert term is None, "The integration term is fixed for JIT compilation"
            term = self.term
        assert term is not None

        norm = self.norm
        f0 = term.vf(t0, y0, stats, args)

        error_bounds = torch.add(self.atol, torch.abs(y0), alpha=self.rtol)
        inv_scale = torch.reciprocal(error_bounds)

        d0 = norm(y0 * inv_scale)
        d1 = norm(f0 * inv_scale)

        small_number = torch.tensor(1e-6, dtype=d0.dtype, device=d0.device)
        dt0 = torch.where((d0 < 1e-5) | (d1 < 1e-5), small_number, 0.01 * d0 / d1)

        y1 = torch.addcmul(y0, (direction.to(dtype=dt0.dtype) * dt0)[:, None], f0)
        f1 = term.vf(
            torch.addcmul(t0, direction.to(dtype=t0.dtype), dt0.to(dtype=t0.dtype)),
            y1,
            stats,
            args,
        )

        d2 = norm((f1 - f0) * inv_scale) / dt0

        maxd1d2 = torch.maximum(d1, d2)
        dt1 = torch.where(
            maxd1d2 <= 1e-15,
            torch.maximum(small_number, dt0 * 1e-3),
            (0.01 / maxd1d2) ** (1.0 / convergence_order),
        )

        return (direction * torch.minimum(100 * dt0, dt1)).to(dtype=t0.dtype), f0
