from typing import Optional

from .typing import *


class InitialValueProblem:
    """An initial value problem.

    You can have different data types in the data and time domains. If your start, end
    and evaluation points are in double precision, all time computations will be done in
    double precision, while all data computations will happen in the data dtype. Note,
    that this requires your term to deal with inputs of different data types and still
    return the correct dtype (the one of `y0`).

    If you do not set explicit evaluation points, the solution will only be evaluated at
    the end points `t_end`. This improves performance of the solver loop in cases where
    we only care about the solution at the end point such as continuous normalizing
    flows.
    """

    def __init__(
        self,
        y0: DataTensor,
        t_start: Optional[TimeTensor] = None,
        t_end: Optional[TimeTensor] = None,
        t_eval: Optional[EvaluationTimesTensor] = None,
    ):
        self.y0 = y0
        self.t_eval = t_eval

        if t_start is None:
            assert t_eval is not None
            t_start = t_eval[:, 0]
        self.t_start = t_start

        if t_end is None:
            assert t_eval is not None
            t_end = t_eval[:, -1]
        self.t_end = t_end


        self.time_direction = torch.where(self.t_end > self.t_start, 1, -1)

        if not torch.jit.is_scripting():
            assert y0.ndim == 2
            assert self.t_start.ndim == 1
            assert self.t_end.ndim == 1
            assert same_dtype(self.t_start, self.t_end)
            assert same_shape(y0, self.t_start, self.t_end, dim=0)
            assert same_device(y0, self.t_start, self.t_end)

            if t_eval is not None:
                assert t_eval.ndim == 2
                assert same_dtype(self.t_start, t_eval)
                assert same_shape(self.t_start, t_eval, dim=0)
                assert same_device(self.t_start, t_eval)

    @property
    def data_dtype(self):
        return self.y0.dtype

    @property
    def time_dtype(self):
        return self.t_start.dtype

    @property
    def device(self):
        return self.y0.device

    @property
    def batch_size(self):
        return self.y0.shape[0]

    @property
    def n_features(self):
        return self.y0.shape[1]

    @property
    def n_evaluation_points(self):
        t_eval = self.t_eval
        if t_eval is None:
            return 0
        else:
            return t_eval.shape[1]

    def __repr__(self):
        return (
            f"InitialValueProblem(y0={self.y0}, t_start={self.t_start}, "
            f"t_end={self.t_end}, t_eval={self.t_eval})"
        )
