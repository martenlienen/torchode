import pytest
from pytest import approx

from problems import STEP_METHODS, get_problem
from torchode import Euler, solve_ivp


def test_simple_interface_works():
    y, term, problem = get_problem("sine", [[0.1, 0.15, 1.0], [1.0, 1.9, 2.0]])

    sol = solve_ivp(term, problem.y0, problem.t_eval)

    assert sol.ys.numpy() == approx(y(sol.ts).numpy(), rel=1e-5)


@pytest.mark.parametrize("step_method", STEP_METHODS)
def test_accepts_step_method_object(step_method):
    y, term, problem = get_problem("sine", [[0.1, 0.15, 1.0], [1.0, 1.9, 2.0]])

    dt0 = (
        problem.t_end.new_full(problem.t_end.shape, 1e-3)
        if step_method is Euler
        else None
    )
    sol = solve_ivp(term, problem.y0, problem.t_eval, method=step_method(), dt0=dt0)

    assert sol.ys.numpy() == approx(y(sol.ts).numpy(), rel=1e-2)
