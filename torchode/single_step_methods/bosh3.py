import torch

from ..interpolation import ThirdOrderPolynomialInterpolation
from ..terms import ODETerm
from .runge_kutta import ButcherTableau, ERKInterpolationData, ExplicitRungeKutta


class Bosh3(ExplicitRungeKutta):
    """The 3rd order Runge-Kutta method by Bogacki and Shampine

    References
    ----------

    ```bibtex
    @article{bogacki1989pair,
      title={A 3(2) pair of Runge--Kutta formulas},
      author={Bogacki, Przemyslaw and Shampine, Lawrence F.},
      journal={Applied Mathematics Letters},
      volume={2},
      number={4},
      pages={321--325},
      year={1989},
      publisher={Elsevier}
    }
    ```
    """

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
