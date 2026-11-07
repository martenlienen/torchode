from unittest.mock import Mock

import pytest
import torch

from torchode import (
    InitialValueProblem,
    IntegralController,
    ODETerm,
    PIDController,
    Status,
)
from torchode.single_step_methods import StepResult
from torchode.step_size_controllers import PIDState, max_norm, rms_norm


@pytest.mark.parametrize(
    "controller",
    [
        PIDController(atol=1, rtol=1, pcoeff=1, icoeff=1, dcoeff=1),
        IntegralController(atol=1.0, rtol=0.0),
    ],
)
def test_no_error_estimate_keeps_step_size_constant(controller):
    t0 = torch.tensor([0.0])
    dt = torch.tensor([1.0])
    y0 = torch.tensor([[1.0, 2.0, 3.0]])

    problem = InitialValueProblem(y0, t0, t0 + dt, t0.reshape((-1, 1)))
    state = controller.initial_state(
        method_order=5, problem=problem, dt_min=None, dt_max=None
    )
    result = StepResult(y0, None)
    accept, dt_next, state_next, status = controller.adapt_step_size(
        t0, dt, y0, result, state, stats={}
    )

    assert accept.tolist() == [True]
    assert dt.tolist() == dt_next.tolist()
    assert status is None


@pytest.mark.parametrize(
    "controller",
    [
        PIDController(atol=1, rtol=1, pcoeff=1, icoeff=1, dcoeff=1),
        IntegralController(atol=1.0, rtol=0.0),
    ],
)
def test_zero_error_produces_finite_step_size(controller):
    t0 = torch.tensor([0.0])
    dt = torch.tensor([1.0])
    y0 = torch.tensor([[1.0, 2.0, 3.0]])

    problem = InitialValueProblem(y0, t0, t0 + dt, t0.reshape((-1, 1)))
    state = controller.initial_state(
        method_order=5, problem=problem, dt_min=None, dt_max=None
    )
    result = StepResult(y0, torch.zeros_like(y0))
    _, dt_next, _, _ = controller.adapt_step_size(t0, dt, y0, result, state, stats={})

    assert torch.isfinite(dt_next).all()


@pytest.mark.parametrize(
    "controller",
    [
        PIDController(atol=1e-5, rtol=1e-5, pcoeff=0.2, icoeff=0.5, dcoeff=0.0),
        IntegralController(atol=1e-5, rtol=1e-5),
    ],
)
def test_zero_length_interval_gets_zero_initial_step(controller):
    t_start = torch.tensor([0.0, 1.0])
    t_end = torch.tensor([1.0, 1.0])
    y0 = torch.tensor([[1.0], [2.0]])
    term = ODETerm(lambda t, y: -y)
    problem = InitialValueProblem(y0, t_start, t_end)
    stats = {}
    term.init(problem, stats)
    dt0, _, _ = controller.init(term, problem, 5, None, stats=stats, args=None)

    assert dt0[0] > 0.0
    assert dt0[1] == 0.0


@pytest.mark.parametrize("make_controller", [PIDController, IntegralController])
@pytest.mark.parametrize(
    ["dt0", "dt_min", "dt_max", "expected"],
    [
        # Without limits, the selected initial step is about 0.0457
        (None, None, None, 0.0457),
        (None, 0.1, None, 0.1),
        (None, None, 0.01, 0.01),
        (1.0, None, 0.01, 0.01),
    ],
)
def test_initial_step_respects_step_size_limits(
    make_controller, dt0, dt_min, dt_max, expected
):
    if make_controller is PIDController:
        controller = PIDController(
            1e-5, 1e-5, 0.2, 0.5, 0.0, dt_min=dt_min, dt_max=dt_max
        )
    else:
        controller = IntegralController(1e-5, 1e-5, dt_min=dt_min, dt_max=dt_max)
    t_start = torch.tensor([0.0, 0.0])
    t_end = torch.tensor([10.0, -10.0])
    y0 = torch.tensor([[1.0], [1.0]])
    term = ODETerm(lambda t, y: -y)
    problem = InitialValueProblem(y0, t_start, t_end)
    stats = {}
    term.init(problem, stats)
    if dt0 is not None:
        dt0 = problem.time_direction * dt0
    dt0, _, _ = controller.init(term, problem, 5, dt0, stats=stats, args=None)

    torch.testing.assert_close(
        dt0, torch.tensor([expected, -expected]), atol=0.0, rtol=1e-3
    )

@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "controller",
    [
        PIDController(atol=1e-5, rtol=1e-5, pcoeff=0.2, icoeff=0.5, dcoeff=0.1),
        IntegralController(atol=1e-5, rtol=1e-5),
    ],
)
def test_initial_step_has_finite_gradients(controller, dtype):
    # A regular instance, one with zero state and one with a zero-length interval
    t_start = torch.tensor([0.0, 0.0, 1.0], dtype=dtype)
    t_end = torch.tensor([1.0, 1.0, 1.0], dtype=dtype)
    y0 = torch.tensor([[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]], dtype=dtype)
    y0.requires_grad_()
    A = torch.tensor([[-0.5, 0.1], [0.2, -0.3]], dtype=dtype, requires_grad=True)
    term = ODETerm(lambda t, y: y @ A)
    problem = InitialValueProblem(y0, t_start, t_end)
    stats = {}
    term.init(problem, stats)
    dt0, _, _ = controller.init(term, problem, 5, None, stats=stats, args=None)
    dt0.sum().backward()

    assert torch.isfinite(y0.grad).all()
    assert torch.isfinite(A.grad).all()


@pytest.mark.parametrize(
    "controller",
    [
        # Use RMS norm because rms_norm(max_float32) is inf
        PIDController(atol=1, rtol=1, pcoeff=1, icoeff=1, dcoeff=1, norm=rms_norm),
        IntegralController(atol=1.0, rtol=0.0, norm=rms_norm),
    ],
)
def test_infinite_error_norm_signals_error_status(controller):
    batch_size = 3
    t0 = torch.zeros(batch_size)
    dt = torch.ones(batch_size)
    y0 = torch.ones((batch_size, 2))

    problem = InitialValueProblem(y0, t0, t0 + dt, t0.reshape((-1, 1)))
    state = controller.initial_state(
        method_order=5, problem=problem, dt_min=None, dt_max=None
    )
    result = StepResult(
        y0,
        error_estimate=torch.tensor(
            [[1.0, 1.0], [torch.finfo(y0.dtype).max, 1.0], [0.5, 1.25]]
        ),
    )
    _, _, _, status = controller.adapt_step_size(t0, dt, y0, result, state, stats={})

    assert status.tolist() == [
        Status.SUCCESS.value,
        Status.INFINITE_NORM.value,
        Status.SUCCESS.value,
    ]


@pytest.mark.parametrize(
    "controller",
    [
        PIDController(atol=1, rtol=1, pcoeff=1, icoeff=1, dcoeff=1, norm=max_norm),
        IntegralController(atol=1.0, rtol=0.0, norm=max_norm),
    ],
)
def test_nan_y_signals_error_status(controller):
    batch_size = 2
    t0 = torch.zeros(batch_size)
    dt = torch.ones(batch_size)
    y0 = torch.tensor([[float("nan"), 1.0], [1.5, 1.0]])

    problem = InitialValueProblem(y0, t0, t0 + dt, t0.reshape((-1, 1)))
    state = controller.initial_state(
        method_order=5, problem=problem, dt_min=None, dt_max=None
    )

    result = StepResult(
        y0,
        error_estimate=torch.ones((batch_size, 2)),
    )
    _, _, _, status = controller.adapt_step_size(t0, dt, y0, result, state, stats={})

    assert status.tolist() == [Status.INFINITE_NORM.value, Status.SUCCESS.value]


@pytest.mark.parametrize(
    "controller",
    [
        PIDController(atol=1.0, rtol=0.0, pcoeff=0, icoeff=1, dcoeff=0, norm=max_norm),
        IntegralController(atol=1.0, rtol=0.0, norm=max_norm),
    ],
)
def test_accepts_step_if_error_small(controller):
    dt_min = torch.tensor(0.5)
    t0 = torch.tensor([0.0, 1.5])
    dt = torch.tensor([1.0, 0.5])
    y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    problem = InitialValueProblem(y0, t0, t0 + dt, t0.reshape((-1, 1)))
    state = controller.initial_state(
        method_order=5, problem=problem, dt_min=None, dt_max=None
    )
    result = StepResult(y0, torch.full_like(y0, 0.5))
    accept, dt_next, state_next, status = controller.adapt_step_size(
        t0, dt, y0, result, state, stats={}
    )

    assert accept.all()
    assert (dt_next > dt).all()
    assert (status == Status.SUCCESS.value).all()
    if isinstance(state, PIDState):
        assert (state_next.prev_error_ratio != 1.0).all()
        assert (state_next.prev_prev_error_ratio == state.prev_error_ratio).all()

    accept, dt_final, state_final, status = controller.adapt_step_size(
        t0 + dt, dt_next, y0, result, state_next, stats={}
    )

    assert accept.all()
    assert (dt_final > dt_next).all()
    assert (status == Status.SUCCESS.value).all()
    if isinstance(state, PIDState):
        assert (state_next.prev_error_ratio != 1.0).all()
        assert (state_final.prev_prev_error_ratio == state_final.prev_error_ratio).all()


@pytest.mark.parametrize(
    "controller",
    [
        PIDController(
            atol=1.0,
            rtol=0.0,
            pcoeff=0,
            icoeff=1,
            dcoeff=0,
            norm=max_norm,
            factor_min=0.0,
        ),
        IntegralController(atol=1.0, rtol=0.0, norm=max_norm, factor_min=0.0),
    ],
)
def test_sets_status_on_dt_min(controller):
    dt_min = torch.tensor(0.5)
    t0 = torch.tensor([0.0, 0.0])
    dt = torch.tensor([1.0, 1.0])
    y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    problem = InitialValueProblem(y0, t0, t0 + dt, t0.reshape((-1, 1)))
    state = controller.initial_state(2, problem, dt_min, dt_max=None)

    # Huge error for the second instance to get a significant decrease in dt
    error = torch.tensor([[0.9, 0.7], [0.0, 100.0]])
    result = StepResult(y0, error)
    accept, dt_next, state_next, status = controller.adapt_step_size(
        t0, dt, y0, result, state, stats={}
    )

    assert accept.tolist() == [True, False]
    assert dt_next[1] == dt_min
    assert status.tolist() == [Status.SUCCESS.value, Status.REACHED_DT_MIN.value]
    if isinstance(state, PIDState):
        assert (state_next.prev_error_ratio[1] == state.prev_error_ratio[1]).all()
        assert (state_next.prev_error_ratio[0] != state.prev_error_ratio[0]).all()
        assert (
            state_next.prev_prev_error_ratio[1] == state.prev_prev_error_ratio[1]
        ).all()
        assert (state_next.prev_prev_error_ratio[0] == state.prev_error_ratio[0]).all()


def test_pid_does_not_update_state_if_step_is_rejected():
    dt_min = torch.tensor(0.5)
    t0 = torch.tensor([0.0, 2.5])
    dt = torch.tensor([1.0, 1.0])
    y0 = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    controller = PIDController(
        atol=0.75,
        rtol=0,
        pcoeff=0,
        icoeff=1,
        dcoeff=0,
        norm=max_norm,
        safety=1.0,
    )
    problem = InitialValueProblem(y0, t0, t0 + dt, t0.reshape((-1, 1)))
    state = controller.initial_state(
        method_order=3, problem=problem, dt_min=None, dt_max=None
    )
    result = StepResult(y0, torch.tensor([[0.0, 0.8, 0.0], [0.0, 0.0, 0.0]]))

    state.prev_error_ratio = torch.rand((2,))
    state.prev_prev_error_ratio = torch.rand((2,))
    accept, dt_next, state_next, status = controller.adapt_step_size(
        t0, dt, y0, result, state, stats={}
    )

    assert list(accept) == [False, True]
    assert dt_next[0] < dt[0]
    assert dt_next[1] > dt[1]
    assert list(status) == [Status.SUCCESS.value, Status.SUCCESS.value]
    assert (state_next.prev_error_ratio[0] == state.prev_error_ratio[0]).all()
    assert (state_next.prev_prev_error_ratio[0] == state.prev_prev_error_ratio[0]).all()
    assert (state_next.prev_prev_error_ratio[1] == state.prev_error_ratio[1]).all()


@pytest.mark.parametrize(
    "controller",
    [
        PIDController(atol=1.0, rtol=0.0, pcoeff=0, icoeff=1, dcoeff=0, norm=max_norm),
        IntegralController(atol=1.0, rtol=0.0, norm=max_norm),
    ],
)
def test_limit_step_size(controller):
    dt_max = 0.4
    t0 = torch.tensor([0.0, 1.5])
    dt = torch.tensor([1.0, 0.5])
    y0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    problem = InitialValueProblem(y0, t0, t0 + dt, t0.reshape((-1, 1)))
    result = StepResult(y0, torch.full_like(y0, 0.5))

    state = controller.initial_state(
        method_order=5, problem=problem, dt_min=None, dt_max=None
    )
    _, dt_next_no_limit, _, _ = controller.adapt_step_size(t0, dt, y0, result, state, stats={})
    assert (dt_next_no_limit > dt_max).any()

    state = controller.initial_state(
        method_order=5, problem=problem, dt_min=None, dt_max=dt_max
    )
    _, dt_next_limited, _, _ = controller.adapt_step_size(t0, dt, y0, result, state, stats={})
    assert (dt_next_limited <= dt_max).all()
