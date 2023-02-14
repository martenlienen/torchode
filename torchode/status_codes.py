from enum import Enum

# Despite TorchScript's purported support of enums, the JIT compiler raises some
# unintelligible errors when confronted with them, because it tries to compile some Enum
# method that it does not support. To be save, we avoid the enum below in any internal
# code that might be JIT compiled and work with these constant integer values directly.
# The enum remains as a more user-friendly alternative that can, for example, give you
# the name of an error code.
SUCCESS = 0
GENERAL_ERROR = 1
REACHED_DT_MIN = 2
REACHED_MAX_STEPS = 3
INFINITE_NORM = 4


class Status(Enum):
    """A joint collection of all possible solver and step size controller status codes.

    Any status greater than 0, i.e. `Status.SUCCESS`, signifies some type of abnormal
    condition.

    This is an Enum instead of an IntEnum, because TorchScript only accepts Enums as of
    pytorch 1.11.
    """

    SUCCESS = SUCCESS
    GENERAL_ERROR = GENERAL_ERROR
    REACHED_DT_MIN = REACHED_DT_MIN
    REACHED_MAX_STEPS = REACHED_MAX_STEPS

    # The norm of the error ratio turned out infinite which points towards some bad
    # problems such as infinite or NaN y and the solving should be cancelled.
    INFINITE_NORM = INFINITE_NORM
