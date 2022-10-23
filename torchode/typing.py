from typing import Optional

import torch
from torchtyping import TensorType, is_float


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

# These are subclasses of TensorType instead of using TensorType annotations directly,
# because TorchScript does not support custom type constructors. In this way, we can
# continue to document the shapes and types of tensors while being TorchScript
# compatible, see [1].
#
# [1] https://github.com/patrick-kidger/torchtyping/issues/13


class DataTensor(TensorType["batch", "feature", is_float]):
    pass


class NormTensor(TensorType["batch", is_float]):
    pass


class SolutionDataTensor(TensorType["batch", "time", "feature", is_float]):
    pass


class TimeTensor(TensorType["batch", is_float]):
    pass


class EvaluationTimesTensor(TensorType["batch", "time", is_float]):
    pass


class AcceptTensor(TensorType["batch", torch.bool]):
    pass


class StatusTensor(TensorType["batch", torch.long]):
    pass


class InterpTimeTensor(TensorType["interp-points", is_float]):
    pass


class InterpDataTensor(TensorType["interp-points", "feature", is_float]):
    pass


class SampleIndexTensor(TensorType["interp-points", torch.long]):
    pass
