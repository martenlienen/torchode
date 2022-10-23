from unittest.mock import Mock

import torch
from problems import create_ivp

from torchode.single_step_methods.runge_kutta import ButcherTableau, ExplicitRungeKutta
from torchode.status_codes import Status


class FixedTableauExplicitRungeKutta(ExplicitRungeKutta):
    TABLEAU = ButcherTableau.from_lists(
        c=[0, 0.5, 1.0],
        a=[[], [1], [0.5, 0.5]],
        b=[1 / 3, 1 / 3, 1 / 3],
        b_low_order=[1 / 2, 0, 1 / 2],
    )

    def __init__(self, term):
        super().__init__(term, FixedTableauExplicitRungeKutta.TABLEAU)


def test_evaluates_term_at_correct_nodes():
    f = Mock(side_effect=lambda t, y: y)
    term, problem = create_ivp(f, [[1.0], [2.0]], [[0.0], [0.0]])
    method = FixedTableauExplicitRungeKutta(term)
    stats = {}
    term.init(problem, stats)
    state = method.init(None, problem, None, stats=stats, args=None)
    dt = torch.tensor([1.0, 1.0])
    running = torch.ones(2, dtype=torch.bool)
    result, _, _, status = method.step(
        None, running, problem.y0, problem.t_start, dt, state, stats=stats, args=None
    )

    f_t = torch.stack([args[0][0] for args in f.call_args_list])
    f_y = torch.stack([args[0][1] for args in f.call_args_list])

    assert status is None
    # 3-stage tableau, so we expect 3 calls
    assert len(f_t) == 3
    assert torch.allclose(f_t[0], problem.t_start)
    assert torch.allclose(f_t[1], problem.t_start + 0.5 * dt)
    assert torch.allclose(f_t[2], problem.t_start + 1.0 * dt)

    y0 = problem.y0
    assert torch.allclose(f_y[0], y0)
    assert torch.allclose(f_y[1], y0 + dt[:, None] * y0)
    assert torch.allclose(f_y[2], y0 + dt[:, None] * (y0 + f_y[1]) / 2)

    assert torch.allclose(result.y, y0 + dt[:, None] * f_y.sum(dim=0) / 3)

    y_err = result.error_estimate
    assert torch.allclose(y_err, dt[:, None] * (-f_y[0] / 6 + f_y[1] / 3 - f_y[2] / 6))
