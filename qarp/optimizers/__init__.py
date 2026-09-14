"""Classical optimizers for variational loops.  Public depth: flat.  Submodules
are private.
"""

from ._optimizer import Optimizer
from ._scipy_optimizer import ScipyOptimizer
from ._rotosolve_optimizer import RotosolveOptimizer

from ._gradient_descent_optimizer import (
    GradientDescentOptimizer,
    SGDOptimizer,
    SPSAOptimizer,
    RMSPropOptimizer,
    AdamOptimizer,
    AdaGradOptimizer,
    AdamaxOptimizer,
    NadamOptimizer,
    EarlyStopper,
    compute_fd_gradients,
    compute_spsa_gradients,
)

__all__ = [
    "AdaGradOptimizer",
    "AdamOptimizer",
    "AdamaxOptimizer",
    "EarlyStopper",
    "GradientDescentOptimizer",
    "NadamOptimizer",
    "Optimizer",
    "RMSPropOptimizer",
    "RotosolveOptimizer",
    "SGDOptimizer",
    "SPSAOptimizer",
    "ScipyOptimizer",
    "compute_fd_gradients",
    "compute_spsa_gradients",
]
