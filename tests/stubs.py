from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch

from torchode import ODETerm
from torchode.interpolation import LocalInterpolation
from torchode.single_step_methods import SingleStepMethod, StepResult
from torchode.step_size_controllers import StepSizeController


class NoInterpolation(LocalInterpolation):
    """Don't interpolate, just evaluate the solution."""

    def __init__(self, f):
        super().__init__()

        self.f = f

    def evaluate(self, t, idx):
        return self.f(t, idx)


class SSMState:
    def merge(self, accept, previous):
        return self


@dataclass
class SSMInterpolationData:
    f: Any


class StubStepMethod(SingleStepMethod[SSMState, SSMInterpolationData]):
    def __init__(self, f, status, error: Optional[Callable] = None):
        def dy(t, y):
            assert False, "this should never be called"

        super().__init__()

        self.term = ODETerm(dy)
        self.f = f
        self.status = status
        self.error = error

    def init(self, term, problem, f0, *, stats, args):
        shape = (problem.batch_size,)
        self.status = EnsureShape(self.status, shape=shape)
        return SSMState()

    def step(self, term, running, y, t, dt, state, *, stats, args):
        y1 = self.f(t)
        if self.error is None:
            error_estimate = torch.zeros_like(y1)
        else:
            error_estimate = self.error(t)
        return (
            StepResult(y1, error_estimate),
            SSMInterpolationData(self.f),
            state,
            self.status(t + dt, y),
        )

    def merge_states(self, accept, current, previous):
        pass

    def build_interpolation(self, data):
        return NoInterpolation(data.f)

    def convergence_order(self):
        return 3


class EnsureShape:
    def __init__(self, value_or_callable, *, shape):
        if callable(value_or_callable):
            self.callable = value_or_callable
        else:
            value = value_or_callable
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value)
            self.callable = lambda *args, **kwargs: value
        self.shape = shape

    def __call__(self, *args, **kwargs):
        value = self.callable(*args, **kwargs)
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value)
        return value.expand(self.shape)


class StubStepSizeController(StepSizeController):
    def __init__(self, dt0, dt, accept, status):
        super().__init__()

        if not isinstance(dt0, torch.Tensor):
            dt0 = torch.tensor(dt0)
        self.dt0 = dt0

        self.dt = dt
        self.accept = accept
        self.status = status

    def init(self, term, problem, convergence_order, dt0, *, stats, args):
        shape = (problem.batch_size,)
        self.dt = EnsureShape(self.dt, shape=shape)
        self.accept = EnsureShape(self.accept, shape=shape)
        self.status = EnsureShape(self.status, shape=shape)

        return self.dt0.expand(shape), {}, None

    def adapt_step_size(self, t0, dt, y0, step_result, state, stats):
        return (
            self.accept(t0, dt, y0, step_result),
            self.dt(t0, dt, y0, step_result),
            state,
            self.status(t0, dt, y0, step_result),
        )

    def merge_states(self, running, current, previous):
        return current
