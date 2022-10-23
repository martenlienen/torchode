from typing import Optional

import torch

from ..interpolation import ThirdOrderPolynomialInterpolation
from ..terms import ODETerm
from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class Heun(ExplicitRungeKutta):
    TABLEAU = ButcherTableau.from_lists(
        c=[0.0, 1.0], a=[[], [1.0]], b=[1 / 2, 1 / 2], b_low_order=[1.0, 0.0]
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, Heun.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return 2

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(
            data.t0, data.dt, data.y0, data.y1, data.k
        )
