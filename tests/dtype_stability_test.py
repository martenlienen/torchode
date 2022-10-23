import pytest
import torch

from torchode import (
    AutoDiffAdjoint,
    Dopri5,
    Heun,
    InitialValueProblem,
    ODETerm,
    PIDController,
    Tsit5,
)


@pytest.mark.parametrize(
    "time_dtype,data_dtype",
    [(torch.float64, torch.float32), (torch.float32, torch.float64)],
)
@pytest.mark.parametrize("step_method", [Dopri5, Heun, Tsit5])
def test_data_and_time_dtypes_are_not_mixed(time_dtype, data_dtype, step_method):
    def f(t, y):
        assert t.dtype == time_dtype
        assert y.dtype == data_dtype

        return torch.zeros_like(y)

    term = ODETerm(f)
    y0 = torch.tensor([[0.33, 1.2, -0.4]], dtype=data_dtype)
    t_eval = torch.tensor([[0.0, 1.77, 3.0]], dtype=time_dtype)
    problem = InitialValueProblem(
        y0, t_start=t_eval[:, 0], t_end=t_eval[:, -1], t_eval=t_eval
    )
    step_size_controller = PIDController(1e-6, 0.0, 0.3, 0.6, 0.0)
    adjoint = AutoDiffAdjoint(step_method(), step_size_controller)
    solution = adjoint.solve(problem, term)

    # Ensure that we took at least a few steps, so that things "get the chance" to be
    # mixed up
    assert solution.stats["n_steps"] > 3

    assert solution.ts.dtype == time_dtype
    assert solution.ys.dtype == data_dtype
