from unittest.mock import Mock

import pytest
import torch
from problems import create_ivp

from torchode import AutoDiffAdjoint, Dopri5, Heun, ODETerm, PIDController, Tsit5


@pytest.mark.parametrize("step_method", [Dopri5, Heun, Tsit5])
def test_extra_args_are_passed_to_dynamics(step_method):
    f = Mock(return_value=torch.ones((1, 3)))
    term = ODETerm(f, with_args=True)
    step_size_controller = PIDController(1e-3, 1e-3, 0.0, 1.0, 0.0)
    adjoint = AutoDiffAdjoint(step_method(term), step_size_controller)
    static_args = (0, True, "hello", torch.tensor([0.0, -1.0]))
    _, problem = create_ivp(f, [[0.0, 1.0, 2.0]], [[0.0, 0.2, 0.6, 1.0]])
    solution = adjoint.solve(problem, term, args=static_args)

    assert f.call_count > 0
    args = [args[0][2] for args in f.call_args_list]
    assert all(arg is static_args for arg in args)
