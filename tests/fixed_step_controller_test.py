import torch
from pytest import approx

from problems import get_problem
from torchode import AutoDiffAdjoint, FixedStepController, Status, Tsit5


def test_solve_with_fixed_steps():
    f, term, problem = get_problem("sine", [[0.1, 0.7, 2.0], [0.5, 1.2, 4.0]])
    controller = FixedStepController()
    adjoint = AutoDiffAdjoint(Tsit5(), controller)
    dt0 = torch.tensor([0.01, 0.005])
    solution = adjoint.solve(problem, term, dt0)

    assert solution.status.tolist() == [Status.SUCCESS.value] * 2
    assert (solution.ts == problem.t_eval).all()
    assert solution.ys.numpy() == approx(f(problem.t_eval), rel=1e-2)
