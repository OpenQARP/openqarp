"""Resource metrics: compiled circuit → stage-explicit resource vectors.

    from qarp.resources import estimate
    report = estimate(block, gateset=qx.clifford_t_rz_gateset(), device=dev)
    report[Stage.ROUTED].swap_count

The frozen, ``SCHEMA_VERSION``-pinned wire format (``ResourceVector``,
``Stage``, ``Provenance``), its producer (``ResourceEstimator`` / ``estimate``),
the ``ResourceModeler`` extension point (no in-tree implementer) and
ε-approximate Rz → Clifford+T synthesis (``synthesize_clifford_t``, optional
``pygridsynth`` dependency).  Public depth: flat.  Submodules are private.
"""

from ._counting import count_resources
from ._estimator import ResourceEstimator, ResourceReport, estimate
from ._modelers import ResourceModeler
from ._synthesis import synthesize_clifford_t
from ._vector import SCHEMA_VERSION, Provenance, ResourceVector, Stage

__all__ = [
    "Provenance",
    "ResourceEstimator",
    "ResourceModeler",
    "ResourceReport",
    "ResourceVector",
    "SCHEMA_VERSION",
    "Stage",
    "count_resources",
    "estimate",
    "synthesize_clifford_t",
]
