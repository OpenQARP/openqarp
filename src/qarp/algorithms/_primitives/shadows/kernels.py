"""Stateless estimation kernels — the one place the shadow inverse channel lives.

A :class:`ShadowKernel` is a small, picklable value object carrying only the
information needed to invert measurement outcomes into observable estimates:
``n_qubits``, ensemble parameters, and the pure inversion functions.  The
collector delegates to it, the :class:`~.dataset.ShadowDataset` carries it, and
the :class:`~.estimator.ShadowEstimator` uses it — so the math is defined once.

The kernel holds **no** qarpx objects and no circuits, so a dataset carrying one
is inert and serializable.  :class:`PauliKernel` ships today; matchgate /
global-Clifford kernels are later subclasses registered the same way — a new
ensemble is a registry entry, not a schema change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Setting axis encoding: a random-Pauli setting is one axis per qubit.
_AXIS_CODE = {"X": 0, "Y": 1, "Z": 2}


class ShadowKernel(ABC):
    """Stateless inversion kernel for one measurement ensemble.

    Attributes:
        ensemble: registry tag used by :meth:`ShadowDataset.from_dict`.
        n_qubits: register width the kernel was built for.
        capabilities: estimator features this ensemble supports (e.g.
            ``{"expval"}``; matchgate would add ``"rdm"``).  The estimator raises
            :class:`CapabilityError` for anything outside this set.
    """

    ensemble: str
    capabilities: frozenset[str]

    def __init__(self, n_qubits: int):
        self.n_qubits = n_qubits

    @abstractmethod
    def snapshot_estimate(self, setting: np.ndarray, outcome: int, term) -> float:
        """Single-snapshot estimate of one Pauli ``term`` under one ``(setting,
        outcome)``.  ``term`` is a tuple of ``(qubit, axis_char)`` pairs (empty
        for identity — never passed here; the estimator handles the constant)."""

    # --- serialization seam (kernel owns its setting encoding) -----------------

    def params(self) -> dict:
        """Ensemble parameters for the serialized descriptor (besides
        ``n_qubits``/``ensemble``).  Default: none."""
        return {}

    @classmethod
    def from_params(cls, n_qubits: int, params: dict) -> "ShadowKernel":
        return cls(n_qubits)


class PauliKernel(ShadowKernel):
    """Random-Pauli (local-Clifford) inverse channel.

    A setting is one axis per qubit (``0/1/2`` = ``X/Y/Z``).  The inverse channel
    factorizes per qubit: for a Pauli term ``P`` a snapshot contributes
    ``prod_{q in supp(P)} 3 * (-1)^{b_q}`` iff the measured axis on every qubit of
    ``supp(P)`` matches ``P``'s axis there, and ``0`` otherwise.
    """

    ensemble = "pauli"
    capabilities = frozenset({"expval"})

    def snapshot_estimate(self, setting: np.ndarray, outcome: int, term) -> float:
        val = 1.0
        for qubit, axis in term:
            if setting[qubit] != _AXIS_CODE[axis]:
                return 0.0
            bit = (outcome >> qubit) & 1
            val *= 3.0 * (1.0 - 2.0 * bit)  # 3 * (+1 if b==0 else -1)
        return val


# Registry keyed by ensemble tag — ShadowDataset.from_dict rebuilds through this,
# so a new ensemble is an added entry, never a schema-version bump.
KERNELS: dict[str, type[ShadowKernel]] = {PauliKernel.ensemble: PauliKernel}
