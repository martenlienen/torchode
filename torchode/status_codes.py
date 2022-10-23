from enum import Enum


class Status(Enum):
    """A joint collection of all possible solver and step size controller status codes.

    Any status greater than 0, i.e. `Status.SUCCESS`, signifies some type of abnormal
    condition.

    This is an Enum instead of an IntEnum, because TorchScript only accepts Enums as of
    pytorch 1.11.
    """

    SUCCESS = 0
    GENERAL_ERROR = 1
    REACHED_DT_MIN = 2
    REACHED_MAX_STEPS = 3

    # The norm of the error ratio turned out infinite which points towards some bad
    # problems such as infinite or NaN y and the solving should be cancelled.
    INFINITE_NORM = 4
