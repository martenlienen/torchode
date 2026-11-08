import pytest
import torch
import torch.nn as nn

from torchode import (
    AutoDiffAdjoint,
    BacksolveAdjoint,
    Dopri5,
    Heun,
    InitialValueProblem,
    JointBacksolveAdjoint,
    ODETerm,
    PIDController,
    Tsit5,
)


class DtypeCheckingDynamics(nn.Module):
    def __init__(self, time_dtype, data_dtype):
        super().__init__()

        self.time_dtype = time_dtype
        self.data_dtype = data_dtype

    def forward(self, t, y):
        assert t.dtype == self.time_dtype
        assert y.dtype == self.data_dtype

        return -y


@pytest.mark.parametrize(
    "time_dtype,data_dtype",
    [(torch.float64, torch.float32), (torch.float32, torch.float64)],
)
@pytest.mark.parametrize("step_method", [Dopri5, Heun, Tsit5])
@pytest.mark.parametrize(
    "adjoint", [AutoDiffAdjoint, BacksolveAdjoint, JointBacksolveAdjoint]
)
def test_data_and_time_dtypes_are_not_mixed(
    time_dtype, data_dtype, step_method, adjoint
):
    term = ODETerm(DtypeCheckingDynamics(time_dtype, data_dtype))
    y0 = torch.tensor([[0.33, 1.2, -0.4]], dtype=data_dtype, requires_grad=True)
    t_eval = torch.tensor([[0.0, 1.77, 3.0]], dtype=time_dtype, requires_grad=True)
    problem = InitialValueProblem(
        y0, t_start=t_eval[:, 0], t_end=t_eval[:, -1], t_eval=t_eval
    )
    step_size_controller = PIDController(1e-6, 0.0, 0.3, 0.6, 0.0)
    if adjoint is AutoDiffAdjoint:
        solver = AutoDiffAdjoint(step_method(), step_size_controller)
    else:
        solver = adjoint(term, step_method(), step_size_controller)
    solution = solver.solve(problem, term)

    # Ensure that we took at least a few steps, so that things "get the chance"
    # to be mixed up
    assert solution.stats["n_steps"] > 3

    assert solution.ts.dtype == time_dtype
    assert solution.ys.dtype == data_dtype

    # Make sure that the adjoint backwards also pass t and y to f with the right
    # dtypes
    grad_y0, grad_t_eval = torch.autograd.grad(solution.ys.sum(), (y0, t_eval))
    assert grad_y0.dtype == data_dtype
    assert grad_t_eval.dtype == time_dtype
