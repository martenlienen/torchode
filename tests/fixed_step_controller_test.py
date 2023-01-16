from pytest import approx
import pytest
import torch
from problems import get_problem

from torchode import AutoDiffAdjoint, Dopri5, FixedStepController, Heun, Status, Tsit5


@pytest.mark.parametrize("step_method", [Dopri5, Heun, Tsit5])
def test_solve_with_fixed_steps(step_method):
    f, term, problem = get_problem("sine", [[0.1, 0.7, 2.0], [0.5, 1.2, 4.0]])
    controller = FixedStepController()
    adjoint = AutoDiffAdjoint(step_method(), controller)
    dt0 = torch.tensor([0.01, 0.005])
    solution = adjoint.solve(problem, term, dt0)

    assert solution.status.tolist() == [Status.SUCCESS.value] * 2
    assert (solution.ts == problem.t_eval).all()
    assert solution.ys.numpy() == approx(f(problem.t_eval), rel=1e-2)
