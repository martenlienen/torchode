import torch

from ..interpolation import ThirdOrderPolynomialInterpolation
from ..terms import ODETerm
from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class Bosh3(ExplicitRungeKutta):
    TABLEAU = ButcherTableau.from_lists(
        c=[0.0, 1 / 2, 3 / 4, 1.0],
        a=[[], [1 / 2], [0.0, 3 / 4], [2 / 9, 1 / 3, 4 / 9]],
        b=[2 / 9, 1 / 3, 4 / 9, 0.0],
        b_low_order=[7 / 24, 1 / 4, 1 / 3, 1 / 8],
    )

    def __init__(self, term: ODETerm | None = None):
        super().__init__(term, Bosh3.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return 3

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        return ThirdOrderPolynomialInterpolation.from_k(
            data.t0, data.dt, data.y0, data.y1, data.k
        )
