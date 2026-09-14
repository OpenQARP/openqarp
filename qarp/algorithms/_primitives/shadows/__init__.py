"""Classical shadows — randomized-measurement estimation.

``PauliShadow`` is a ``PrimitiveAlgorithm`` that returns ``<operator>`` and, from
the same campaign, exposes a reusable :class:`ShadowDataset` for estimating many
further observables via :class:`ShadowEstimator`.
"""

from .base import ShadowProtocol
from .dataset import ShadowDataset
from .estimator import ShadowEstimate, ShadowEstimator
from .kernels import PauliKernel, ShadowKernel
from .pauli import PauliShadow

__all__ = [
    "ShadowProtocol",
    "PauliShadow",
    "ShadowDataset",
    "ShadowEstimator",
    "ShadowEstimate",
    "ShadowKernel",
    "PauliKernel",
]
