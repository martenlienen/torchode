from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn

from ..interpolation import LinearInterpolation
from ..problems import InitialValueProblem
from ..status_codes import Status
from ..terms import ODETerm
from ..typing import *
from .base import StepResult

NoneType = type(None)


class LinearInterpolationData(NamedTuple):
    t0: TimeTensor
    dt: TimeTensor
    y0: DataTensor
    y1: DataTensor


class Euler(nn.Module):
    def __init__(self, term: Optional[ODETerm]):
        super().__init__()

        self.term = term

    @torch.jit.export
    def init(
        self,
        term: Optional[ODETerm],
        problem: InitialValueProblem,
        f0: Optional[DataTensor],
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> NoneType:
        return None

    @torch.jit.export
    def step(
        self,
        term: Optional[ODETerm],
        running: AcceptTensor,
        y0: DataTensor,
        t0: TimeTensor,
        dt: TimeTensor,
        state: NoneType,
        *,
        stats: Dict[str, Any],
        args: Any,
    ) -> Tuple[StepResult, LinearInterpolationData, NoneType, Optional[StatusTensor]]:
        term_ = term
        if torch.jit.is_scripting() or term_ is None:
            assert term is None, "The integration term is fixed for JIT compilation"
            term_ = self.term
        assert term_ is not None

        # Convert dt into the data dtype for dtype stability
        dt_data = dt.to(dtype=y0.dtype)

        y1 = torch.addcmul(y0, dt_data[:, None], term_.vf(t0, y0, stats, args))

        return (
            StepResult(y1, None),
            LinearInterpolationData(t0, dt, y0, y1),
            state,
            None,
        )

    @torch.jit.export
    def merge_states(
        self, accept: AcceptTensor, current: NoneType, previous: NoneType
    ) -> NoneType:
        return None

    @torch.jit.export
    def convergence_order(self):
        return 1

    @torch.jit.export
    def build_interpolation(self, data: LinearInterpolationData):
        return LinearInterpolation(data.t0, data.dt, data.y0, data.y1)
