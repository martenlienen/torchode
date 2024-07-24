from typing import Optional

import torch


def same_dtype(*tensors: torch.Tensor):
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:]):
        if a.dtype != b.dtype:
            return False
    return True


def same_device(*tensors: torch.Tensor):
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:]):
        if a.device != b.device:
            return False
    return True


def same_shape(*tensors: torch.Tensor, dim: Optional[int] = None):
    if len(tensors) <= 1:
        return True
    for a, b in zip(tensors[:-1], tensors[1:]):
        if dim is None:
            if a.shape != b.shape:
                return False
        else:
            if a.shape[dim] != b.shape[dim]:
                return False
    return True


################
# Tensor Types #
################


class DataTensor(torch.Tensor):
    """
    Data tensor.

    TensorType["batch", "feature", is_float]
    """

    pass


class NormTensor(torch.Tensor):
    """
    Norm tensor.

    TensorType["batch", is_float]
    """

    pass


class SolutionDataTensor(torch.Tensor):
    """
    Solution data tensor.

    TensorType["batch", "time", "feature", is_float]
    """

    pass


class TimeTensor(torch.Tensor):
    """
    Time tensor.

    TensorType["batch", is_float]
    """

    pass


class EvaluationTimesTensor(torch.Tensor):
    """
    Evaluation times tensor.

    TensorType["batch", "time", is_float]
    """

    pass


class AcceptTensor(torch.Tensor):
    """
    Accept tensor.

    TensorType["batch", torch.bool]
    """

    pass


class StatusTensor(torch.Tensor):
    """
    Status tensor.

    TensorType["batch", torch.long]
    """

    pass


class InterpTimeTensor(torch.Tensor):
    """
    Interpolation time tensor.

    TensorType["interp-points", is_float]
    """

    pass


class InterpDataTensor(torch.Tensor):
    """
    Interpolation data tensor.

    TensorType["interp-points", "feature", is_float]
    """

    pass


class SampleIndexTensor(torch.Tensor):
    """
    Sample index tensor.

    TensorType["interp-points", torch.long]
    """

    pass
