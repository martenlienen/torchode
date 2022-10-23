import torch

from torchode import InitialValueProblem, ODETerm
from torchode.typing import *


def make_time_tensor(t, dims):
    if not isinstance(t, torch.Tensor):
        t = torch.tensor(t)
    for _ in range(dims - len(t.shape)):
        t = t.unsqueeze(dim=0)
    return t


def create_ivp(f, y0, t_eval, t_start=None, t_end=None):
    term = ODETerm(f)
    if not isinstance(y0, torch.Tensor):
        y0 = torch.tensor(y0)
    if t_eval is None:
        assert t_start is not None
        assert t_end is not None
    else:
        t_eval = make_time_tensor(t_eval, 2)
        if t_start is None:
            t_start = t_eval[:, 0]
        if t_end is None:
            t_end = t_eval[:, -1]
    t_start = make_time_tensor(t_start, 1)
    t_end = make_time_tensor(t_end, 1)
    return term, InitialValueProblem(y0, t_start, t_end, t_eval)


def sine_solution(t: TimeTensor, *args):
    return (
        -0.5 * t**4 * torch.cos(2 * t)
        + 0.5 * t**3 * torch.sin(2 * t)
        + 0.25 * t**2 * torch.cos(2 * t)
        - t**3
        + 2 * t**4
        + (torch.pi - 0.25) * t**2
    )[..., None]


def sine_dynamics(t: TimeTensor, y: DataTensor):
    return (2 * y[:, 0] / t + t**4 * torch.sin(2 * t) - t**2 + 4 * t**3)[
        ..., None
    ]


KNOWN_SOLUTIONS = {"sine": (sine_solution, sine_dynamics)}


def get_problem(name: str, t_eval, t_start=None, t_end=None):
    if t_eval is None:
        assert t_start is not None
        assert t_end is not None
    else:
        t_eval = make_time_tensor(t_eval, 2)
        if t_start is None:
            t_start = t_eval[:, 0]
        if t_end is None:
            t_end = t_eval[:, -1]
    t_start = make_time_tensor(t_start, 1)
    t_end = make_time_tensor(t_end, 1)
    y, dy = KNOWN_SOLUTIONS[name]
    y0 = y(t_start)
    term, problem = create_ivp(dy, y0, t_eval, t_start, t_end)
    return y, term, problem
