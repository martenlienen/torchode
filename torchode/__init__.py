"""A parallel ODE solver for PyTorch"""

__version__ = "1.1.0"

from .adjoints import AutoDiffAdjoint, BacksolveAdjoint, JointBacksolveAdjoint
from .interface import register_method, solve_ivp
from .problems import InitialValueProblem
from .single_step_methods import Bosh3, Dopri5, Euler, Heun, Tsit5
from .solution import Solution
from .status_codes import Status
from .step_size_controllers import (
    FixedStepController,
    IntegralController,
    PIDController,
)
from .terms import ODETerm

register_method("euler", Euler)
register_method("bosh3", Bosh3)
register_method("heun", Heun)
register_method("dopri5", Dopri5)
register_method("tsit5", Tsit5)
