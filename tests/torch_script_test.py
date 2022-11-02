import pytest
import torch
from problems import get_problem
from pytest import approx

from torchode import AutoDiffAdjoint, Dopri5, Euler, Heun, IntegralController, Tsit5


@pytest.mark.parametrize("step_method", [Dopri5, Heun, Tsit5, Euler])
def test_can_be_jitted_with_torch_script(step_method):
    _, term, problem = get_problem("sine", [[0.1, 0.15, 1.0], [1.0, 1.9, 2.0]])
    step_size_controller = IntegralController(1e-3, 1e-3, term=term)
    adjoint = AutoDiffAdjoint(step_method(term), step_size_controller)
    jitted = torch.jit.script(adjoint)

    dt0 = torch.tensor([0.01, 0.01]) if step_method is Euler else None
    solution = adjoint.solve(problem, dt0=dt0)
    solution_jit = jitted.solve(problem, dt0=dt0)

    assert solution.ts == approx(solution_jit.ts)
    assert solution.ys == approx(solution_jit.ys, abs=1e-3, rel=1e-3)


@pytest.mark.parametrize("step_method", [Dopri5, Heun, Tsit5, Euler])
def test_passing_term_dynamically_equals_fixed_term(step_method):
    _, term, problem = get_problem("sine", [[0.1, 0.15, 1.0], [1.0, 1.9, 2.0]])

    dt0 = torch.tensor([0.01, 0.01]) if step_method is Euler else None

    controller = IntegralController(1e-3, 1e-3)
    adjoint = AutoDiffAdjoint(step_method(None), controller)
    solution = adjoint.solve(problem, term, dt0=dt0)

    controller_jit = IntegralController(1e-3, 1e-3, term=term)
    adjoint_jit = AutoDiffAdjoint(step_method(term), controller_jit)
    solution_jit = torch.jit.script(adjoint_jit).solve(problem, dt0=dt0)

    assert solution.ts == approx(solution_jit.ts)
    assert solution.ys == approx(solution_jit.ys, abs=1e-3, rel=1e-3)
