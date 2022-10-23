from typing import Optional

import sympy as sp
import torch

from ..interpolation import FourthOrderPolynomialInterpolation
from ..terms import ODETerm
from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


def compute_interpolation_weights():
    """Compute the interpolation weights for the Tsit5 interpolation coefficients of
    2nd, 3rd and 4th order.

    The original Tsit5 paper builds the 4th order interpolation polynomial as a linear
    combination of 7 polynomials. This function computes weights that give us the
    coefficients a, b, c in the standard polynomial form `a*x^4 + b*x^3 + ...` directly.
    This way, we can evaluate the interpolant more efficiently.
    """

    t, y0, dt, f1, f2, f3, f4, f5, f6, f7 = sp.symbols(
        "t y_0 dt f_1 f_2 f_3 f_4 f_5 f_6 f_7"
    )
    f = [f1, f2, f3, f4, f5, f6, f7]
    # fmt: off
    # The 7 basis functions of the interpolant
    b = [
        -1.0530884977290216 * t * (t - 1.3299890189751412) * (t**2 - 1.4364028541716351 * t + 0.7139816917074209),
        0.1017 * t**2 * (t**2 - 2.1966568338249754 * t + 1.2949852507374631),
        2.490627285651252793 * t**2 * (t**2 - 2.38535645472061657 * t + 1.57803468208092486),
        -16.54810288924490272 * (t - 1.21712927295533244) * (t - 0.61620406037800089) * t**2,
        47.37952196281928122 * (t - 1.203071208372362603) * (t - 0.658047292653547382) * t**2,
        -34.87065786149660974 * (t - 1.2) * (t - 0.666666666666666667) * t**2,
        2.5 * (t - 1) * (t - 0.6) * t**2
    ]
    # fmt: on
    interpolant = y0 + dt * sum(f_i * b_i for f_i, b_i in zip(f, b))
    # Fully expand the polynomial and collect the powers of t
    form = sp.collect(sp.expand(interpolant, t), t)
    # The coefficients of t^2, t^3 and t^4 are of the form
    #
    #     dt * \sum_i x_i f_i
    #
    # and we collect the x_i here in a matrix. Then the coefficients of the interpolant
    # can be found by a matrix multiplication between this and the k vector of the RK
    # stages.
    return [
        [float(form.coeff(t, i).coeff(f[j]).coeff(dt)) for j in range(len(f))]
        for i in range(2, 5)
    ]


class Tsit5(ExplicitRungeKutta):
    """The 5th order Runge-Kutta method by Tsitouras

    References
    ----------

    ```bibtex
    @article{tsitouras2011runge,
      title={Runge--Kutta pairs of order 5(4) satisfying only the first column
             simplifying assumption},
      author={Tsitouras, Charalampos},
      journal={Computers \\& Mathematics with Applications},
      volume={62},
      number={2},
      pages={770--775},
      year={2011},
      publisher={Elsevier}
    }
    ```
    """

    TABLEAU = ButcherTableau.from_lists(
        c=[0.0, 0.161, 0.327, 0.9, 0.9800255409045097, 1.0, 1.0],
        a=[
            # fmt: off
            [],
            [0.161],
            [-0.008480655492356989, 0.335480655492357],
            [2.8971530571054935, -6.359448489975075, 4.3622954328695815],
            [5.325864828439257, -11.748883564062828, 7.4955393428898365, -0.09249506636175525],
            [5.86145544294642, -12.92096931784711, 8.159367898576159, -0.071584973281401, -0.02826905039406838],
            [0.09646076681806523, 0.01, 0.4798896504144996, 1.379008574103742, -3.290069515436081, 2.324710524099774],
            # fmt: on
        ],
        b=[
            0.09646076681806523,
            0.01,
            0.4798896504144996,
            1.379008574103742,
            -3.290069515436081,
            2.324710524099774,
            0.0,
        ],
        # The paper introduces b-tilde as the weights of the lower-order interpolant but
        # the weights they give in the end are actually directly the weights for the
        # error estimate, see [1].
        #
        # [1] https://github.com/patrick-kidger/diffrax/issues/98
        b_err=[
            0.00178001105222577714,
            0.0008164344596567469,
            -0.007880878010261995,
            0.1447110071732629,
            -0.5823571654525552,
            0.45808210592918697,
            # The original paper has the sign of this coefficient wrong, see [1]
            #
            # [1] https://github.com/SciML/OrdinaryDiffEq.jl/issues/1654
            -1 / 66,
        ],
        b_other=compute_interpolation_weights(),
    )

    def __init__(self, term: Optional[ODETerm] = None):
        super().__init__(term, Tsit5.TABLEAU)

    @torch.jit.export
    def convergence_order(self):
        return 5

    @torch.jit.export
    def build_interpolation(self, data: ERKInterpolationData):
        y0 = data.y0
        dt = data.dt.to(dtype=y0.dtype)
        f0 = data.k[0]
        b = data.tableau.b_other
        assert b is not None

        B = torch.einsum("b, cs, sbf -> cbf", dt, b, data.k)
        c, b, a = B[0], B[1], B[2]
        d = dt[:, None] * f0
        e = y0

        coefficients = (e, d, c, b, a)
        return FourthOrderPolynomialInterpolation(
            data.t0, data.t0 + data.dt, coefficients
        )
