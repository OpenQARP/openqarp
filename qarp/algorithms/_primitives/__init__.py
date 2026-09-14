"""Primitive algorithms — one estimand per instance, three orthogonal axes.

1. **Target** (*what quantity*): ``Target.SAMPLING`` / ``EXPECTATION_VALUE`` /
   ``OVERLAP`` / ``TRANSITION_AMPLITUDE``.  Each primitive declares its
   ``supported_targets``; input-driven primitives infer the target from their
   ``{ket, bra, operator}`` inputs via ``infer_target()``.
2. **Estimation strategy** (*how*): the class.  Protocol primitives
   (``Sampler``, ``PauliAveraging``, ``HadamardTest``, ``SWAPTest``,
   ``MirrorTest``, ``InterferometricTest``, ``Termwise*``,
   ``CuttingPrimitive``) build measurement circuits and consume **counts**
   (``consumes = Consumes.COUNTS``).  ``StateVector`` consumes **amplitudes**
   directly (``Consumes.AMPLITUDES``) — no measurement circuits, always
   exact, the only strategy eligible for adjoint backprop.
3. **Readout mode** (*finite shots vs the ∞-shot limit*): the ``n_shots``
   knob on protocol primitives.  ``qarp.EXACT`` feeds the same estimator the
   exact Born probabilities instead of sampled counts — any protocol, zero
   shot noise (requires an engine with amplitude access: noiseless
   simulation).  ``None`` defers to the engine default.

Rule of thumb: *want the quantity* → ``StateVector`` (cheapest, exact);
*study the protocol* → the protocol primitive, sampled or ``EXACT``.
Exact-protocol and ``StateVector`` agree numerically on scalar targets but
exist for different reasons: validation vs evaluation.
"""

from .target import Target

from .primitive_algorithm import PrimitiveAlgorithm
from .sampler import Sampler
from .state_vector import StateVector
from .swap_test import SWAPTest
from .hadamard_test import HadamardTest
from .mirror_test import MirrorTest
from .interferometric_test import InterferometricTest
from .termwise_hadamard_test import TermwiseHadamardTest
from .grouped_transition_hadamard_test import GroupedTransitionHadamardTest
from .termwise_swap_test import TermwiseSWAPTest
from .pauli_averaging import PauliAveraging
from .basis_rotation_averaging import BasisRotationAveraging
from .cutting_primitive import CuttingPrimitive
from .shadows import (
    ShadowProtocol,
    PauliShadow,
    ShadowDataset,
    ShadowEstimator,
    ShadowEstimate,
    ShadowKernel,
    PauliKernel,
)
