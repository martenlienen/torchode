from typing import Optional

import torch

from ..interpolation import FourthOrderPolynomialInterpolation
from ..terms import ODETerm
from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class Dopri5(ExplicitRungeKutta):
    TABLEAU = ButcherTableau.from_lists(
        c=[0.0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1.0, 1.0],
        a=[
            [],
            [1 / 5],
            [3 / 40, 9 / 40],
            [44 / 45, -56 / 15, 32 / 9],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ],
        b=[35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0.0],
        b_low_order=[
            1951 / 21600,
            0,
            22642 / 50085,
            451 / 720,
            -12231 / 42400,
            649 / 6300,
            1 / 60,
        ],
        b_other=[
            # Coefficients for y at the mid point
            [
                6025192743 / 30085553152 / 2,
                0,
                51252292925 / 65400821598 / 2,
                -2691868925 / 45128329728 / 2,
                187940372067 / 1594534317056 / 2,
                -1776094331 / 19743644256 / 2,
                11237099 / 235043384 / 2,
            ]
        ],
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, Dopri5.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return 5

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        b_other = data.tableau.b_other
        assert b_other is not None
        b_mid = b_other[0]
        return FourthOrderPolynomialInterpolation.from_k(
            data.t0, data.dt, data.y0, data.y1, data.k, b_mid
        )
