from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
from torchtyping import TensorType

from ..interpolation import LocalInterpolation
from ..problems import InitialValueProblem
from ..status_codes import Status
from ..terms import ODETerm
from ..typing import *
from .base import StepResult


class CoefficientVector(TensorType["nodes"]):
    pass


class RungeKuttaMatrix(TensorType["nodes", "weights"]):
    pass


class WeightVector(TensorType["weights"]):
    pass


class WeightMatrix(TensorType["rows", "weights"]):
    pass


class ButcherTableau:
    def __init__(
        self,
        # Coefficients for the evaluation nodes in time
        c: CoefficientVector,
        # Runge-Kutta matrix
        a: RungeKuttaMatrix,
        # Coefficients for the high-order solution estimate
        b: WeightVector,
        # Coefficients for the error estimate
        b_err: WeightVector,
        # Additional additional rows of the b matrix
        b_other: Optional[WeightMatrix] = None,
        fsal: Optional[bool] = None,
        ssal: Optional[bool] = None,
    ):
        self.c = c
        self.a = a
        self.b = b
        self.b_err = b_err
        self.b_other = b_other

        if fsal is None:
            fsal = self.is_fsal()
        self.fsal = fsal
        if ssal is None:
            ssal = self.is_ssal()
        self.ssal = ssal

    @staticmethod
    def from_lists(
        *,
        c: List[float],
        a: List[List[float]],
        b: List[float],
        b_err: Optional[List[float]] = None,
        b_low_order: Optional[List[float]] = None,
        b_other: Optional[List[List[float]]] = None,
    ):
        assert b_err is not None or b_low_order is not None, (
            "You have to provide either the weights for the error approximation"
            " or the weights of an embedded lower-order method"
        )

        n_nodes = len(c)
        n_weights = len(b)
        assert n_nodes == n_weights
        assert len(a) == n_nodes

        # Fill a up into a full square matrix
        a_full = [row + [0.0] * (n_weights - len(row)) for row in a]

        b_coeffs = torch.tensor(b, dtype=torch.float64)
        if b_err is None:
            assert b_low_order is not None
            assert len(b_low_order) == n_weights
            b_low_coeffs = torch.tensor(b_low_order, dtype=torch.float64)
            b_err_coeffs = b_coeffs - b_low_coeffs
        else:
            b_err_coeffs = torch.tensor(b_err, dtype=torch.float64)

        if b_other is None:
            b_other_coeffs = None
        else:
            b_other_coeffs = torch.tensor(b_other, dtype=torch.float64)
            assert b_other_coeffs.ndim == 2
            assert b_other_coeffs.shape[1] == n_weights

        return ButcherTableau(
            c=torch.tensor(c, dtype=torch.float64),
            a=torch.tensor(a_full, dtype=torch.float64),
            b=b_coeffs,
            b_err=b_err_coeffs,
            b_other=b_other_coeffs,
        )

    def to(
        self, device: torch.device, time_dtype: torch.dtype, data_dtype: torch.dtype
    ) -> "ButcherTableau":
        b_other = self.b_other
        if b_other is not None:
            b_other = b_other.to(device, data_dtype)
        return ButcherTableau(
            c=self.c.to(device, time_dtype),
            a=self.a.to(device, data_dtype),
            b=self.b.to(device, data_dtype),
            b_err=self.b_err.to(device, data_dtype),
            b_other=b_other,
            fsal=self.fsal,
            ssal=self.ssal,
        )

    @property
    def n_stages(self):
        return self.c.shape[0]

    def is_fsal(self):
        """Is `f(y0)` equal to `f(y1)` from the previous step?

        If that is the case, we can reuse the result from the previous step.
        """
        is_lower_triangular = (torch.triu(self.a, diagonal=1) == 0.0).all().item()
        first_node_is_t0 = (self.c[0] == 0.0).item()
        last_node_is_t1 = (self.c[-1] == 1.0).item()
        first_stage_explicit = (self.a[0, 0] == 0.0).item()
        return (
            is_lower_triangular
            and (self.b == self.a[-1]).all().item()
            and first_node_is_t0
            and last_node_is_t1
            and first_stage_explicit
        )

    def is_ssal(self):
        """Is the solution equal to the last stage result?

        If that is the case, we can avoid the final computation of the solution and
        return the last stage result instead.
        """
        is_lower_triangular = (torch.triu(self.a, diagonal=1) == 0.0).all().item()
        last_node_is_t1 = (self.c[-1] == 1.0).item()
        last_stage_explicit = (self.a[-1, -1] == 0.0).item()
        return (
            is_lower_triangular
            and (self.b == self.a[-1]).all().item()
            and last_node_is_t1
            and last_stage_explicit
        )


class ERKInterpolationData(NamedTuple):
    tableau: ButcherTableau
    t0: TimeTensor
    dt: TimeTensor
    y0: DataTensor
    y1: DataTensor
    k: torch.Tensor


class ERKState(NamedTuple):
    tableau: ButcherTableau
    prev_vf1: Optional[DataTensor]


class ExplicitRungeKutta(nn.Module):
    def __init__(self, term: Optional[ODETerm], tableau: ButcherTableau):
        super().__init__()

        self.term = term
        self.tableau = tableau

    @torch.jit.export
    def init(
        self,
        term: Optional[ODETerm],
        problem: InitialValueProblem,
        f0: Optional[DataTensor],
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> ERKState:
        if self.tableau.fsal:
            term_ = term
            if torch.jit.is_scripting() or term_ is None:
                assert term is None, "The integration term is fixed for JIT compilation"
                term_ = self.term
            assert term_ is not None

            if f0 is None:
                prev_vf1 = term_.vf(problem.t_start, problem.y0, stats, args)
            else:
                prev_vf1 = f0
        else:
            prev_vf1 = None

        return ERKState(
            tableau=self.tableau.to(
                device=problem.device,
                data_dtype=problem.data_dtype,
                time_dtype=problem.time_dtype,
            ),
            prev_vf1=prev_vf1,
        )

    @torch.jit.export
    def merge_states(self, accept: AcceptTensor, current: ERKState, previous: ERKState):
        prev_vf1 = previous.prev_vf1
        current_vf1 = current.prev_vf1
        if current_vf1 is None or prev_vf1 is None:
            return current
        else:
            return ERKState(
                current.tableau, torch.where(accept[:, None], current_vf1, prev_vf1)
            )

    @torch.jit.export
    def step(
        self,
        term: Optional[ODETerm],
        running: AcceptTensor,
        y0: DataTensor,
        t0: TimeTensor,
        dt: TimeTensor,
        state: ERKState,
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> Tuple[StepResult, ERKInterpolationData, ERKState, Optional[StatusTensor]]:
        term_ = term
        if torch.jit.is_scripting() or term_ is None:
            assert term is None, "The integration term is fixed for JIT compilation"
            term_ = self.term
        assert term_ is not None
        tableau = state.tableau

        # Convert dt into the data dtype for dtype stability
        dt_data = dt.to(dtype=y0.dtype)

        prev_vf1 = state.prev_vf1
        vf0 = (
            prev_vf1
            if tableau.fsal and prev_vf1 is not None
            else term_.vf(t0, y0, stats, args)
        )
        y_i = y0
        k = vf0.new_empty((tableau.n_stages, vf0.shape[0], vf0.shape[1]))
        k[0] = vf0
        a = tableau.a
        t_nodes = torch.addcmul(t0, tableau.c[:, None], dt)
        for i in range(1, tableau.n_stages):
            y_i = torch.einsum("j, jbf -> bf", a[i, :i], k[:i])
            y_i = torch.addcmul(y0, dt_data[:, None], y_i)
            k[i] = term_.vf(t_nodes[i], y_i, stats, args)

        if tableau.ssal:
            y1 = y_i
        else:
            y1 = y0 + torch.einsum("b, s, sbf -> bf", dt_data, tableau.b, k)
        error_estimate = torch.einsum("b, s, sbf -> bf", dt_data, tableau.b_err, k)

        if tableau.fsal:
            state = ERKState(state.tableau, k[-1])

        return (
            StepResult(y1, error_estimate),
            ERKInterpolationData(tableau, t0, dt, y0, y1, k),
            state,
            None,
        )

    def build_interpolation(self, data: ERKInterpolationData) -> LocalInterpolation:
        raise NotImplementedError()
