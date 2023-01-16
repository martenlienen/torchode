from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from problems import create_ivp, get_problem
from pytest import approx
from stubs import StubStepMethod, StubStepSizeController

from torchode import (
    AutoDiffAdjoint,
    BacksolveAdjoint,
    Dopri5,
    Euler,
    FixedStepController,
    InitialValueProblem,
    IntegralController,
    JointBacksolveAdjoint,
    ODETerm,
    PIDController,
    Status,
)


def test_evaluates_solution_at_evaluation_points():
    y, _, problem = get_problem("sine", [[0.0, 2.0], [2.0, 4.0], [0.5, 2.5]])
    step_method = StubStepMethod(y, Status.SUCCESS.value)
    step_size_controller = StubStepSizeController(0.3, 0.3, True, Status.SUCCESS.value)

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert (solution.status == Status.SUCCESS.value).all()
    assert solution.stats["n_steps"].tolist() == [7, 7, 7]
    assert solution.stats["n_accepted"].tolist() == [7, 7, 7]
    assert solution.stats["n_initialized"].tolist() == [2, 2, 2]
    assert (solution.ts == problem.t_eval).all()
    assert (solution.ys == y(problem.t_eval)).all()


def test_odes_step_independently():
    y, _, problem = get_problem("sine", [[0.0, 0.15, 1.0], [1.0, 1.9, 2.0]])
    step_method = StubStepMethod(y, Status.SUCCESS.value)
    step_size_controller = StubStepSizeController(
        [0.1, 0.3], [0.5, 0.125], True, Status.SUCCESS.value
    )

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert (solution.status == Status.SUCCESS.value).all()
    assert solution.stats["n_steps"].tolist() == [3, 7]
    assert solution.stats["n_accepted"].tolist() == [3, 7]
    assert solution.stats["n_initialized"].tolist() == [3, 3]
    assert (solution.ts == problem.t_eval).all()
    assert (solution.ys == y(problem.t_eval)).all()


def test_multiple_evaluation_points_in_single_step():
    y, _, problem = get_problem("sine", [[0.0, 0.25, 0.33, 1.0]])
    step_method = StubStepMethod(y, Status.SUCCESS.value)
    step_size_controller = StubStepSizeController(0.1, 0.5, True, Status.SUCCESS.value)

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert (solution.status == Status.SUCCESS.value).all()
    assert solution.stats["n_steps"].tolist() == [3]
    assert solution.stats["n_accepted"].tolist() == [3]
    assert solution.stats["n_initialized"].tolist() == [4]
    assert (solution.ts == problem.t_eval).all()
    assert (solution.ys == y(problem.t_eval)).all()


def test_multiple_evals_on_the_last_step():
    # The first ODE will be evaluated twice in the first iteration and another two times
    # in the second iteration due to the third ODE
    y, _, problem = get_problem(
        "sine", [[0.0, 1.0], [0.9, 1.0], [0.9, 1.0]], t_start=[0.0] * 3, t_end=[1.0] * 3
    )
    step_method = StubStepMethod(y, Status.SUCCESS.value)
    step_size_controller = StubStepSizeController(
        [1.1, 1.1, 0.6], 0.6, True, Status.SUCCESS.value
    )

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert (solution.status == Status.SUCCESS.value).all()
    assert solution.stats["n_steps"].tolist() == [1, 1, 2]
    assert solution.stats["n_accepted"].tolist() == [1, 1, 2]
    assert solution.stats["n_initialized"].tolist() == [2, 2, 2]
    assert (solution.ts == problem.t_eval).all()
    assert (solution.ys == y(problem.t_eval)).all()


def test_no_t_eval_evaluates_at_t_end():
    dy = lambda t, y: torch.ones_like(y)
    y0 = [[0.0], [1.0], [2.0]]
    t_start = [0.0, 5.0, 2.0]
    t_end = [10.0, 9.0, 4.5]
    term, problem = create_ivp(dy, y0, t_eval=None, t_start=t_start, t_end=t_end)
    step_method = Euler(term)
    dt = [0.5001, 0.2501, 1.0]
    step_size_controller = StubStepSizeController(dt, dt, True, Status.SUCCESS.value)

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert (solution.status == Status.SUCCESS.value).all()
    assert solution.stats["n_steps"].tolist() == [20, 16, 3]
    assert solution.stats["n_accepted"].tolist() == [20, 16, 3]
    assert solution.stats["n_initialized"].tolist() == [1, 1, 1]
    assert solution.ts[:, 0].tolist() == t_end
    assert solution.ys[:, 0].numpy() == approx(
        (problem.y0 + (problem.t_end - problem.t_start)[:, None]).numpy()
    )


def test_terminates_after_max_steps():
    y0 = torch.zeros((2, 1))
    t_eval = torch.tensor([[1.0, 4.9], [2.0, 13.0]])
    problem = InitialValueProblem(y0, t_eval[:, 0], t_eval[:, -1], t_eval)
    y = lambda t, *args: y0[args[0]] if len(args) > 0 else y0
    step_method = StubStepMethod(y, Status.SUCCESS.value)
    adjoint = AutoDiffAdjoint(step_method, FixedStepController(), max_steps=7)
    solution = adjoint.solve(problem, dt0=torch.ones(2))

    assert solution.status.tolist() == [
        Status.SUCCESS.value,
        Status.REACHED_MAX_STEPS.value,
    ]
    assert solution.stats["n_steps"].tolist() == [4, 7]
    assert solution.stats["n_initialized"].tolist() == [2, 1]


def test_rejected_steps_continue_at_same_place():
    y, _, problem = get_problem("sine", [[0.0, 1.0], [1.0, 2.0]])
    step_method = StubStepMethod(y, Status.SUCCESS.value)
    step_method.step = Mock(side_effect=step_method.step)
    accept = Mock(side_effect=[True, [True, False], [True, False], True, True])
    step_size_controller = StubStepSizeController(
        0.1, 0.5, accept, Status.SUCCESS.value
    )

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    step_times = torch.stack([args[0][3] for args in step_method.step.call_args_list])
    expected_step_times = [
        [0.0, 1.0],
        [0.1, 1.1],
        [0.6, 1.1],
        [1.0, 1.1],
        [1.0, 1.6],
    ]
    assert step_times.numpy() == approx(np.array(expected_step_times))
    assert (solution.status == Status.SUCCESS.value).all()
    assert solution.stats["n_steps"].tolist() == [3, 5]
    assert solution.stats["n_accepted"].tolist() == [3, 3]
    assert solution.stats["n_initialized"].tolist() == [2, 2]
    assert (solution.ts == problem.t_eval).all()
    assert (solution.ys == y(problem.t_eval)).all()


def test_value_is_only_saved_when_step_is_accepted():
    y, _, problem = get_problem("sine", [[0.6]] * 3, t_start=[0.0] * 3, t_end=[1.0] * 3)
    f_eval = Mock(side_effect=[torch.tensor([[float(i)]] * 3) for i in range(3)])
    f_mock = lambda t: f_eval(t) if torch.all(t == 0.6) else y(t)
    f_idx_wrap = lambda t, *args: f_mock(t) if len(args) == 0 else f_mock(t)[args[0]]
    step_method = StubStepMethod(f_idx_wrap, Status.SUCCESS.value)
    accept = [[False, True, True], [False, True, True], [True, True, True]]
    dt = [[1.1, 0.0, 0.5], [1.1, 0.0, 0.0], [0.0] * 3]
    step_size_controller = StubStepSizeController(
        [1.1, 1.1, 0.55],
        Mock(side_effect=dt),
        Mock(side_effect=accept),
        Status.SUCCESS.value,
    )

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert (solution.status == Status.SUCCESS.value).all()
    assert solution.stats["n_steps"].tolist() == [3, 1, 2]
    assert solution.stats["n_accepted"].tolist() == [1, 1, 2]
    assert solution.stats["n_initialized"].tolist() == [1, 1, 1]
    assert (solution.ts == problem.t_eval).all()
    assert (solution.ys == torch.tensor([[[2.0]], [[0.0]], [[1.0]]])).all()


def test_rejection_of_initial_step_does_not_skip_evaluation_at_t_start():
    y0 = torch.tensor([[1.3], [2.5]])
    y = lambda t, *args: y0[args[0]] if len(args) > 0 else y0
    f = lambda t, y: torch.tensor([[0.0], [0.0]])
    _, problem = create_ivp(f, y0, [[0.0], [0.0]], t_start=[0.0, 0.0], t_end=[1.0, 2.0])
    step_method = StubStepMethod(y, Status.SUCCESS.value)
    accept = [[False, False], True, True, True, True, True]
    step_size_controller = StubStepSizeController(
        0.6, 0.6, Mock(side_effect=accept), Status.SUCCESS.value
    )

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert (solution.status == Status.SUCCESS.value).all()
    assert solution.stats["n_steps"].tolist() == [3, 5]
    assert solution.stats["n_accepted"].tolist() == [2, 4]
    assert solution.stats["n_initialized"].tolist() == [1, 1]
    assert (solution.ts == problem.t_eval).all()
    assert (solution.ys == y0[:, None]).all()


def test_stops_on_non_successful_step_method():
    y, _, problem = get_problem("sine", [[0.0, 2.0], [2.0, 4.0]])
    step_method = StubStepMethod(
        y,
        Mock(
            side_effect=[
                Status.SUCCESS.value,
                Status.SUCCESS.value,
                Status.SUCCESS.value,
                [Status.SUCCESS.value, Status.GENERAL_ERROR.value],
            ]
        ),
    )
    step_size_controller = StubStepSizeController(0.2, 0.2, True, Status.SUCCESS.value)

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert solution.status.tolist() == [
        Status.SUCCESS.value,
        Status.GENERAL_ERROR.value,
    ]
    assert solution.stats["n_steps"].tolist() == [4, 4]
    assert solution.stats["n_accepted"].tolist() == [4, 4]
    assert solution.stats["n_initialized"].tolist() == [1, 1]


def test_finished_solves_do_not_update_dt():
    # Here the time step for the second instance would normally grow with every step
    # because the error is tiny and we use an adaptive step size controller. However,
    # the second instance is finished after a single step, so dt should never change.
    y0 = torch.zeros((2, 1))
    f = lambda t, *args: y0[args[0]] if len(args) > 0 else y0
    f_err = lambda t: torch.tensor([[0.9], [1e-10]])
    step_method = StubStepMethod(f, Status.SUCCESS.value, error=f_err)
    step = Mock(side_effect=step_method.step)
    step_method.step = step
    term, problem = create_ivp(f, f(0.0), [[0.0, 5.0], [0.0, 0.9]])
    adjoint = AutoDiffAdjoint(
        step_method,
        IntegralController(atol=1.0, rtol=0.0, term=term),
    )
    solution = adjoint.solve(problem, dt0=torch.ones(2))

    assert solution.status.tolist() == [Status.SUCCESS.value, Status.SUCCESS.value]
    dts = torch.stack([args[0][2] for args in step.call_args_list])
    assert (dts[:, 1] < 1.1).all()


def test_solution_is_only_evaluated_inside_of_integration_range():
    t_eval = torch.tensor([[0.1, 2.0], [0.3, 0.3 + 1e-3], [1.0, 1.0 - 5e-1]])
    t_start, t_end = t_eval.T
    y, term, problem = get_problem("sine", t_eval)
    error = lambda y: 0.01 * torch.ones_like(y)
    step_method = StubStepMethod(y, Status.SUCCESS.value, error=error)
    step_method.step = Mock(side_effect=step_method.step)
    step_size_controller = IntegralController(atol=1.0, rtol=0.0, term=term)

    solution = AutoDiffAdjoint(step_method, step_size_controller).solve(problem)
    step_times = torch.stack([args[0][3] for args in step_method.step.call_args_list])

    t_min, t_max = torch.minimum(t_start, t_end), torch.maximum(t_start, t_end)
    assert (step_times >= t_min).all()
    assert (step_times <= t_max).all()

    assert solution.status.tolist() == [Status.SUCCESS.value] * 3
    assert solution.ts.tolist() == problem.t_eval.tolist()


def test_stops_on_non_successful_adapt_step_size():
    y, _, problem = get_problem("sine", [[0.0, 2.0], [2.0, 4.0]])
    step_method = StubStepMethod(y, Status.SUCCESS.value)
    step_size_controller = StubStepSizeController(
        0.2,
        0.2,
        True,
        Mock(
            side_effect=[
                *([Status.SUCCESS.value] * 5),
                [Status.GENERAL_ERROR.value, Status.SUCCESS.value],
            ]
        ),
    )

    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    assert solution.status.tolist() == [
        Status.GENERAL_ERROR.value,
        Status.SUCCESS.value,
    ]
    assert solution.stats["n_steps"].tolist() == [6, 6]
    assert solution.stats["n_accepted"].tolist() == [6, 6]
    assert solution.stats["n_initialized"].tolist() == [1, 1]


def test_inverting_time_inverts_dynamics():
    y, term, problem = get_problem("sine", [[0.3, 2.0], [1.5, 6.3]])
    step_method = Dopri5(term)
    step_size_controller = IntegralController(1e-8, 1e-8, term=term)
    adjoint = AutoDiffAdjoint(step_method, step_size_controller)
    solution = adjoint.solve(problem)

    problem_inv = InitialValueProblem(
        solution.ys[:, -1],
        t_start=problem.t_end,
        t_end=problem.t_start,
        t_eval=torch.flip(problem.t_eval, dims=(1,)),
    )
    solution_inv = adjoint.solve(problem_inv)

    assert solution.status.tolist() == [
        Status.SUCCESS.value,
        Status.SUCCESS.value,
    ]
    assert solution_inv.status.tolist() == [
        Status.SUCCESS.value,
        Status.SUCCESS.value,
    ]
    assert solution_inv.ys[:, -1].numpy() == approx(
        problem.y0.numpy(), abs=1e-4, rel=1e-4
    )


@pytest.mark.parametrize(
    "controller",
    [
        PIDController(atol=1e-5, rtol=1e-5, pcoeff=0.0, icoeff=1.0, dcoeff=0.0),
        IntegralController(atol=1e-5, rtol=1e-5),
    ],
)
def test_solving_in_opposite_directions_at_the_same_time(controller):
    term, problem = create_ivp(
        lambda t, y: 2 * torch.ones_like(y),
        y0=[[1.0], [9.0]],
        t_eval=[[0.0, 3.0, 5.0], [2.0, 0.5, -8.0]],
    )
    step_method = Dopri5()
    adjoint = AutoDiffAdjoint(step_method, controller)
    solution = adjoint.solve(problem, term=term)

    expected = [[1.0, 7.0, 11.0], [9.0, 6.0, -11.0]]
    assert solution.ys.squeeze(dim=-1).numpy() == approx(np.array(expected))


class LinearDynamics(nn.Module):
    def __init__(self, A):
        super().__init__()

        self.register_parameter("A", A)

    def forward(self, t, y):
        return y @ self.A


@pytest.mark.parametrize("adjoint", ["autodiff", "backsolve", "joint-backsolve"])
def test_gradients_with_t_eval(adjoint):
    def f(y0, params):
        term = ODETerm(LinearDynamics(params))
        if adjoint == "joint-backsolve":
            # JointBacksolveAdjoint is only applicable if the evaluation points are
            # "aligned"
            t_eval = A.new_tensor([[0.0, 1.125, 3.0], [2.0, 3.5, 6.0]])
        else:
            t_eval = A.new_tensor([[0.0, 1.77, 3.0], [2.0, 3.5, 6.0]])
        problem = InitialValueProblem(
            y0, t_start=t_eval[:, 0], t_end=t_eval[:, -1], t_eval=t_eval
        )
        step_method = Dopri5()
        step_size_controller = PIDController(1e-6, 0.0, 0.3, 0.6, 0.0)
        if adjoint == "autodiff":
            adjoint_ = AutoDiffAdjoint(step_method, step_size_controller)
        elif adjoint == "backsolve":
            adjoint_ = BacksolveAdjoint(term, step_method, step_size_controller)
        elif adjoint == "joint-backsolve":
            adjoint_ = JointBacksolveAdjoint(term, step_method, step_size_controller)
        solution = adjoint_.solve(problem, term=term)

        return solution.ys.mean()

    # I tried a few random matrices and with this one it is reasonably fast and also has
    # a few rejected steps
    y0 = torch.tensor(
        [[0.33, 1.2, -0.4], [0.55, -0.8, 1.23]], dtype=torch.double, requires_grad=True
    )
    A = nn.Parameter(
        torch.tensor(
            [
                [-1.4534, -0.3726, -0.7916],
                [-1.0553, 0.9511, 0.1316],
                [-1.8320, -0.7707, -1.0867],
            ],
            dtype=torch.double,
            requires_grad=True,
        )
    )

    assert torch.autograd.gradcheck(f, (y0, A))


@pytest.mark.parametrize("adjoint", ["autodiff", "backsolve", "joint-backsolve"])
def test_gradients_without_t_eval(adjoint):
    def f(y0, params):
        term = ODETerm(LinearDynamics(params))
        t_start = A.new_tensor([0.0, 2.0])
        t_end = A.new_tensor([3.0, 6.0])
        problem = InitialValueProblem(y0, t_start=t_start, t_end=t_end, t_eval=None)
        step_method = Dopri5()
        step_size_controller = PIDController(1e-6, 0.0, 0.3, 0.6, 0.0)
        if adjoint == "autodiff":
            adjoint_ = AutoDiffAdjoint(
                step_method,
                step_size_controller,
                # Curiously, this test only passes if we take the gradients through the
                # time steps into account
                backprop_through_step_size_control=True,
            )
        elif adjoint == "backsolve":
            adjoint_ = BacksolveAdjoint(term, step_method, step_size_controller)
        elif adjoint == "joint-backsolve":
            adjoint_ = JointBacksolveAdjoint(term, step_method, step_size_controller)
        solution = adjoint_.solve(problem, term=term)

        return solution.ys.mean()

    # I tried a few random matrices and with this one it is reasonably fast and also has
    # a few rejected steps
    y0 = torch.tensor(
        [[0.33, 1.2, -0.4], [0.55, -0.8, 1.23]], dtype=torch.double, requires_grad=True
    )
    A = nn.Parameter(
        torch.tensor(
            [
                [-1.4534, -0.3726, -0.7916],
                [-1.0553, 0.9511, 0.1316],
                [-1.8320, -0.7707, -1.0867],
            ],
            dtype=torch.double,
            requires_grad=True,
        )
    )

    if adjoint == "autodiff":
        atol = 1e-5
    elif adjoint == "backsolve":
        # BacksolveAdjoint has no equivalent of backprop_through_step_size_control for
        # now, so we accept a somewhat larger error in the gradient for now.
        atol = 1e-2
    elif adjoint == "joint-backsolve":
        # BacksolveAdjoint has not equivalent of backprop_through_step_size_control for
        # now, so we accept a somewhat larger error in the gradient for now.
        atol = 1e-5

    assert torch.autograd.gradcheck(f, (y0, A), atol=atol)
