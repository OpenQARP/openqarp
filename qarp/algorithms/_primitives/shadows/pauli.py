"""``PauliShadow`` — the random-Pauli (local-Clifford) shadow ensemble.

Each setting draws an independent measurement axis per qubit; the per-setting
circuit is ``ket -> (per-qubit basis rotation) -> measure-all``.  The inversion
lives in :class:`~.kernels.PauliKernel`.
"""

from __future__ import annotations

import numpy as np

from ....blocks import AnyBlock, SimpleBlock
from ....blocks._block import CompositeBlockBase
from .base import ShadowProtocol
from .kernels import PauliKernel


class PauliShadow(ShadowProtocol):
    """Random-Pauli classical shadows (Huang–Kueng–Preskill, 2020)."""

    def _sample_settings(self, rng: np.random.Generator, n_settings: int, n_qubits: int):
        # 0/1/2 = X/Y/Z, one axis per qubit per setting.
        return rng.integers(0, 3, size=(n_settings, n_qubits), dtype=np.int8)

    def _kernel(self, n_qubits: int) -> PauliKernel:
        return PauliKernel(n_qubits)

    def _basis_change_block(self, setting) -> SimpleBlock:
        """Per-qubit rotation into the sampled measurement basis (measure Z after):
        ``X -> H`` ; ``Y -> S† then H`` ; ``Z -> nothing``.

        Its unitary ``R`` obeys ``R† Z R = P_axis`` on each qubit, so measuring Z
        in the rotated frame is measuring the axis the :class:`PauliKernel` then
        inverts.  That identity is the seam between circuit and kernel and is
        pinned exactly in ``test_pauli.py`` (a sign error here is otherwise
        invisible to sampled ⟨Z⟩/⟨X⟩ checks).
        """
        n = self.n_qubits
        basis = SimpleBlock(n, name="basis_change")
        for qubit in range(n):
            axis = int(setting[qubit])
            if axis == 0:  # X
                basis.h(qubit)
            elif axis == 1:  # Y
                basis.sdg(qubit)
                basis.h(qubit)
            # axis == 2 (Z): identity
        basis.target_qubits = list(range(n))
        basis.mark_built()
        return basis

    def _setting_block(self, setting) -> AnyBlock:
        n = self.n_qubits
        full = list(range(n))
        composite = CompositeBlockBase(n_qubits=n, name="PauliShadowSetting")

        ket_built = self.ket.build()
        ket_built.target_qubits = full
        composite.add_child(ket_built)

        composite.add_child(self._basis_change_block(setting))

        measure = SimpleBlock(n, name="measure")
        measure.measure([(qubit, qubit) for qubit in range(n)])
        measure.target_qubits = full
        composite.add_wired_child(measure)

        composite.build()
        return composite
