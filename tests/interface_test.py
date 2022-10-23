from problems import get_problem
from pytest import approx

from torchode import solve_ivp


def test_simple_interface_works():
    y, term, problem = get_problem("sine", [[0.1, 0.15, 1.0], [1.0, 1.9, 2.0]])

    sol = solve_ivp(term, problem.y0, problem.t_eval)

    assert sol.ys.numpy() == approx(y(sol.ts).numpy(), rel=1e-5)
