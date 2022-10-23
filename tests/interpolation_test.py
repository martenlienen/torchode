import numpy as np
import torch
from numpy.polynomial import Polynomial

from torchode import Tsit5
from torchode.interpolation import (
    FourthOrderPolynomialInterpolation,
    ThirdOrderPolynomialInterpolation,
)
from torchode.single_step_methods.runge_kutta import ERKInterpolationData


def test_third_order_evaluation_batch():
    p = Polynomial([1.1, 0.2, -0.7, -2.8], domain=(0, 1), window=(0, 1))
    t0 = torch.tensor([2.5, 1.0, 5.0])
    t1 = torch.tensor([5.0, 3.0, 6.0])
    batch_size = 3
    n_features = 5
    coefficients = [torch.tensor([[c] * n_features] * batch_size) for c in p.coef]
    interp = ThirdOrderPolynomialInterpolation(t0, t1, coefficients)

    t = (t0 + t1) * 0.5
    idx = torch.arange(batch_size)
    assert np.allclose(interp.evaluate(t, idx).numpy(), p(0.5))


def test_third_order_coefficients_from_k():
    t0 = np.array([0.3])
    dt = np.array([1.9])
    p = Polynomial([1.1, 0.2, -0.7, -2.8], domain=(t0[0], (t0 + dt)[0]), window=(0, 1))
    y0 = torch.from_numpy(p(t0))[:, None].float()
    y1 = torch.from_numpy(p(t0 + dt))[:, None].float()
    p_deriv = p.deriv()
    k = torch.tensor(np.stack([p_deriv(t0), p_deriv(t0 + dt)]))[..., None].float()
    t0, dt = torch.tensor(t0), torch.tensor(dt)
    interp = ThirdOrderPolynomialInterpolation.from_k(t0, dt, y0, y1, k)

    assert interp.t0.allclose(t0)
    assert interp.t1.allclose(t0 + dt)
    for i in range(4):
        assert np.allclose(interp.coefficients[i].numpy(), p.coef[i])


def test_fourth_order_evaluation_batch():
    p = Polynomial([1, -2, 5, 2.5, 3.7], domain=(0, 1), window=(0, 1))
    t0 = torch.tensor([2.5, 1.0, 5.0])
    t1 = torch.tensor([5.0, 3.0, 6.0])
    batch_size = 3
    n_features = 5
    coefficients = [torch.tensor([[c] * n_features] * batch_size) for c in p.coef]
    interp = FourthOrderPolynomialInterpolation(t0, t1, coefficients)

    t = (t0 + t1) * 0.5
    idx = torch.arange(batch_size)
    assert np.allclose(interp.evaluate(t, idx).numpy(), p(0.5))


def test_fourth_order_coefficients_from_k():
    t0 = np.array([1.0])
    dt = np.array([1.5])
    p = Polynomial(
        [1, -0.5, 1.2, 3.3, 0.5], domain=(t0[0], (t0 + dt)[0]), window=(0, 1)
    )
    y0 = torch.from_numpy(p(t0))[:, None].float()
    y1 = torch.from_numpy(p(t0 + dt))[:, None].float()
    p_deriv = p.deriv()
    k = torch.tensor(
        np.stack([p_deriv(t0), (p(t0 + dt / 2) - p(t0)) / dt, p_deriv(t0 + dt)])
    )[..., None].float()
    b_mid = torch.tensor([0.0, 1.0, 0.0])
    t0, dt = torch.tensor(t0), torch.tensor(dt)
    interp = FourthOrderPolynomialInterpolation.from_k(t0, dt, y0, y1, k, b_mid)

    assert interp.t0.allclose(t0)
    assert interp.t1.allclose(t0 + dt)
    for i in range(5):
        assert np.allclose(interp.coefficients[i].numpy(), p.coef[i])


def test_tsit5_recovers_coefficients_of_4th_order_polynomial():
    t0 = np.array([1.0])
    dt = np.array([1.5])
    p = Polynomial(
        [1, -0.5, 1.2, 3.3, 0.5], domain=(t0[0], (t0 + dt)[0]), window=(0, 1)
    )
    y0 = torch.from_numpy(p(t0))[:, None].float()
    y1 = torch.from_numpy(p(t0 + dt))[:, None].float()
    p_deriv = p.deriv()
    tableau = Tsit5.TABLEAU.to(y0.device, y0.dtype, y0.dtype)
    k = torch.tensor(
        np.stack(
            [p_deriv(t0 + tableau.c[i].numpy() * dt) for i in range(len(tableau.c))]
        )
    )[..., None].float()
    t0, dt = torch.tensor(t0), torch.tensor(dt)
    interp = Tsit5.build_interpolation(
        None, ERKInterpolationData(tableau, t0, dt, y0, y1, k)
    )

    assert interp.t0.allclose(t0)
    assert interp.t1.allclose(t0 + dt)
    for i in range(5):
        assert np.allclose(interp.coefficients[i].numpy(), p.coef[i], atol=1e-4)
