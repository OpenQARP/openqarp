from typing import Dict, List, Optional, Tuple

import numpy as np

from ...operators import JordanWigner, Mapping
from .. import SimpleBlock
from .._prepares_known_state import prepares_known_state
from .sparse_state_block import _prepare_sparse_circuit


@prepares_known_state
class MultiONVStateBlock(SimpleBlock):
    def __init__(
        self,
        onv_coefficients: Dict[Tuple[int, ...], complex],
        mapping: Optional[Mapping] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "MultiONV",
    ):
        """Prepare a superposition of occupation-number vectors under a
        fermion-to-qubit mapping — a CI-vector reference (CASCI/CISD) rather
        than the single determinant ``MappedONVStateBlock`` loads.

        Each ONV is mapped independently through ``mapping.encode_state``
        (a fixed linear bit transform for JW/BK/Parity — ``qarp/operators/
        _mappings.py`` — carrying no relative phase of its own, the same
        convention ``MappedONVStateBlock`` already relies on for a single
        determinant), and the resulting mapped bitstring/coefficient pairs
        are handed to the same ancilla-free sparse construction
        ``SparseStateBlock`` uses, so this block stays ancilla-free and
        scales with the number of determinants rather than ``2**n_qubits``.

        Determinant sign convention: the coefficient of ONV ``b`` multiplies
        ``a†_{p₁} a†_{p₂} … a†_{pₖ} |vac⟩`` with ``p₁ < p₂ < … < pₖ`` the
        occupied spin-orbital indices in abab order (``2·spatial + 0`` for α,
        ``+1`` for β).  This is qarp's Jordan–Wigner convention (Z-string on
        the lower indices, as in openfermion), so it is exactly what products
        of ``FermionOperator`` creation operators give.  pyscf CI vectors use
        an αα…ββ… string order that needs a per-determinant sign to reach it —
        convert them with ``qarp.operators.pyscf.onv_coefficients_from_civec``.

        Under Bravyi–Kitaev pass ``BravyiKitaev(n_qubits=len(onv))`` so the
        operator transform and ``encode_state`` agree on the register width
        (``encode_state`` sizes its β matrix by ``len(onv)``).

        Args:
            onv_coefficients: dict mapping occupation-number vectors — as
                tuples of 0/1, abab order (``qarp/operators/onv.py``) — to
                complex CI coefficients. All vectors must have equal length.
                Normalized internally.
            mapping: fermion-to-qubit mapping (default ``JordanWigner()``).
            target_qubits, name: standard Block kwargs.
        """
        if not onv_coefficients:
            raise ValueError("onv_coefficients dictionary cannot be empty.")
        lengths = {len(onv) for onv in onv_coefficients}
        if len(lengths) != 1:
            raise ValueError("All occupation-number vectors must have the same length.")
        n_qubits = lengths.pop()
        for onv in onv_coefficients:
            if not all(bit in (0, 1) for bit in onv):
                raise ValueError(f"onv {onv} contains invalid values; only 0 and 1 are allowed.")

        if mapping is None:
            mapping = JordanWigner()
        self.mapping = mapping
        self.onv_coefficients: Dict[Tuple[int, ...], complex] = dict(onv_coefficients)

        mapped: Dict[Tuple[int, ...], complex] = {}
        for onv, coeff in self.onv_coefficients.items():
            basis = tuple(mapping.encode_state(list(onv)))
            if basis in mapped:
                raise ValueError(
                    f"ONV {onv} maps to the same basis state {basis} as another "
                    f"entry: the two are distinct determinants, but "
                    f"{type(mapping).__name__}.encode_state is not injective on them."
                )
            mapped[basis] = complex(coeff)
        norm = sum(abs(amp) ** 2 for amp in mapped.values()) ** 0.5
        if norm < 1e-15:
            raise ValueError("Amplitudes cannot all be zero.")
        self._mapped_amplitudes: Dict[Tuple[int, ...], complex] = {
            basis: amp / norm for basis, amp in mapped.items()
        }

        super().__init__(n_qubits, target_qubits=target_qubits, name=name)

    def build_vanilla(self) -> None:
        _prepare_sparse_circuit(self, self._mapped_amplitudes)

    def target_statevector(self) -> np.ndarray:
        """The normalized, mapping-encoded CI coefficients, LSB-first (§1)."""
        psi = np.zeros(2**self.n_qubits, dtype=complex)
        for basis, amp in self._mapped_amplitudes.items():
            idx = sum(bit << i for i, bit in enumerate(basis))
            psi[idx] = amp
        return psi
