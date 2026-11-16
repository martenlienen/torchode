import pytest
import torch

from torchode import FixedStepController, solve_ivp


@pytest.mark.parametrize(
    "device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
)
def test_third_order_convergence(device):
    y0 = torch.tensor(
        [[1.0], [2.0]], dtype=torch.float64, device=device, requires_grad=True
    )
    ts = y0.new_tensor([[0.0, 0.37, 1.0], [0.0, -0.43, -1.0]])
    errors = []
    for n in [8, 16]:
        sol = solve_ivp(
            lambda t, y: y,
            y0,
            ts,
            method="bosh3",
            controller=FixedStepController(),
            dt0=ts[:, -1] / n,
        )
        exact = y0[:, None] * ts.exp().unsqueeze(-1)
        errors.append((sol.ys[:, -1] - exact[:, -1]).abs())
    ratio = errors[0] / errors[1]
    assert ((7 < ratio) & (ratio < 9)).all()
    torch.testing.assert_close(sol.ys, exact, atol=5e-5, rtol=1e-5)
    (grad,) = torch.autograd.grad(sol.ys.sum(), y0)
    torch.testing.assert_close(
        grad, ts.exp().sum(1, keepdim=True), atol=5e-5, rtol=1e-5
    )
