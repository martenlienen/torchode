from itertools import pairwise

import torch
from torchtyping import TensorType, is_float


def same_dtype(*tensors: torch.Tensor):
    return all(a.dtype == b.dtype for a, b in pairwise(tensors))


def same_device(*tensors: torch.Tensor):
    return all(a.device == b.device for a, b in pairwise(tensors))


def same_shape(*tensors: torch.Tensor, dim: int | None = None):
    if dim is None:
        return all(a.shape == b.shape for a, b in pairwise(tensors))
    return all(a.shape[dim] == b.shape[dim] for a, b in pairwise(tensors))


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
