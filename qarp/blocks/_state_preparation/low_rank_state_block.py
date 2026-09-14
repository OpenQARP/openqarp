from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .._block import CompositeBlockBase, SimpleBlock
from .._prepares_known_state import prepares_known_state
from .._primitives.synthesized_unitary_block import SynthesizedUnitaryBlock
from ._linalg import _complex_gram_schmidt_complete
from .sparse_state_block import SparseStateBlock

_NORM_TOL = 1e-12


@prepares_known_state
class LowRankStateBlock(CompositeBlockBase):
    def __init__(
        self,
        n_qubits: int,
        amplitudes: Union[List[complex], Dict[Tuple[int, ...], complex]],
        cut: Optional[int] = None,
        max_schmidt_rank: Optional[int] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "LowRankState",
    ):
        r"""Approximate an arbitrary dense state by truncating its Schmidt
        rank across a chosen qubit bipartition (Araujo et al., *Sci. Rep.*
        **11**, 6329 (2021)) — an ε-dial for ``SynthesizedStateBlock``:
        exact when ``max_schmidt_rank`` is left at the full rank, an
        approximation with a known ``error_bound`` when it is lowered.

        **Cost (measured, CNOTs after decomposition on
        ``clifford_t_rz_gateset``, ``O0``).**  Truncation currently buys
        *no* circuit saving: a full ``2^cut × 2^cut`` unitary is synthesised
        per side (QSD, O(4^cut) = O(2^n) CNOTs at the middle cut), so at
        ``n = 8, cut = 4`` rank 1 costs 336, rank 2 costs 337 and full rank
        16 costs 568 against 508 for the dense ``SynthesizedStateBlock``; at
        ``n = 6`` rank 1 costs 72, full rank 111, dense 124.  Araujo's saving
        comes from synthesising only the ``rank`` constrained columns (an
        isometry), and rank 1 should be two plain state preparations
        (≈ 30 CNOTs at ``n = 8``); both are declared follow-ups.  Until they land,
        use this block for the ε-dial, not for cost.

        Bipartitions the register into ``A = qubits[0, cut)`` and
        ``B = qubits[cut, n)``, computes the Schmidt decomposition
        (SVD of the amplitude tensor reshaped ``(2^cut, 2^(n-cut))``) and
        keeps the top ``max_schmidt_rank`` terms ``λ_i |u_i⟩_A |v_i⟩_B``.
        The circuit is exact for the *truncated* state and needs no ancilla:
        ``⌈log₂(max_schmidt_rank)⌉`` of ``A``'s own qubits carry
        ``Σᵢ λᵢ|i⟩`` (``SparseStateBlock``), copied onto the matching
        ``B`` qubits by CNOT, then a single unitary on each side
        (``SynthesizedUnitaryBlock``, Gram-Schmidt-completed from the kept
        singular vectors) turns ``|i⟩_A|i⟩_B`` into ``|uᵢ⟩_A|vᵢ⟩_B`` —
        deterministic, ``ancilla_postselection = None``.

        ``error_bound`` is the discarded Schmidt weight
        ``1 - Σᵢ_{<rank} σᵢ²``, which *is* the infidelity to the untruncated
        target exactly (not just an upper bound) — the kept amplitudes are
        renormalized, so ``|⟨target|prepared⟩| = √(Σ σᵢ²)``.  ``is_exact``
        is ``True`` exactly when ``max_schmidt_rank`` equals the full rank.

        Args:
            n_qubits: number of qubits.
            amplitudes: target statevector, as a list or a dict of
                basis-state tuples (LSB-first, §1) — same convention as
                ``SynthesizedStateBlock``. Normalized internally.
            cut: qubit index splitting ``A = [0, cut)`` from
                ``B = [cut, n)``. Defaults to ``n_qubits // 2``.
            max_schmidt_rank: number of Schmidt terms to keep. Defaults to
                the full rank (``min(2**cut, 2**(n_qubits - cut))`` — exact,
                ``error_bound = 0``). Must be at most that full rank.
            target_qubits, name: standard Block kwargs.
        """
        if cut is None:
            cut = n_qubits // 2
        if not (0 < cut < n_qubits):
            raise ValueError(f"cut ({cut}) must be strictly between 0 and n_qubits ({n_qubits}).")

        dim_a, dim_b = 2**cut, 2 ** (n_qubits - cut)
        dense = _to_dense_amplitudes(amplitudes, n_qubits)
        norm = np.linalg.norm(dense)
        if norm < _NORM_TOL:
            raise ValueError("Amplitudes cannot all be zero.")
        dense = dense / norm

        matrix = dense.reshape(dim_b, dim_a).T
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        full_rank = len(singular_values)
        rank = full_rank if max_schmidt_rank is None else max_schmidt_rank
        if not (1 <= rank <= full_rank):
            raise ValueError(f"max_schmidt_rank must be between 1 and {full_rank}, got {rank}.")

        self.error_bound_value = max(0.0, 1.0 - float(np.sum(singular_values[:rank] ** 2)))
        self._dense_target = dense
        self.schmidt_rank = rank
        self.full_schmidt_rank = full_rank

        lambdas = singular_values[:rank] / np.linalg.norm(singular_values[:rank])
        self._lambdas = lambdas
        self._unitary_a = _complex_gram_schmidt_complete(u[:, :rank])
        self._unitary_b = _complex_gram_schmidt_complete(vh[:rank, :].T)
        self._index_qubits = int(np.ceil(np.log2(rank))) if rank > 1 else 0
        self._cut = cut

        super().__init__(n_qubits=n_qubits, target_qubits=target_qubits, name=name)

    def build_vanilla(self) -> None:
        cut, index_qubits = self._cut, self._index_qubits
        if index_qubits > 0:
            index_data = {
                tuple((i >> b) & 1 for b in range(index_qubits)): complex(self._lambdas[i])
                for i in range(self.schmidt_rank)
            }
            index_block = SparseStateBlock(index_qubits, index_data)
            index_block.target_qubits = list(range(index_qubits))
            self.add_wired_child(index_block)

            copy_block = SimpleBlock(self.n_qubits, name="copy_index")
            for j in range(index_qubits):
                copy_block.cx(j, cut + j)
            self.add_wired_child(copy_block)

        unitary_a = SynthesizedUnitaryBlock(self._unitary_a)
        unitary_a.target_qubits = list(range(cut))
        self.add_wired_child(unitary_a)

        unitary_b = SynthesizedUnitaryBlock(self._unitary_b)
        unitary_b.target_qubits = list(range(cut, self.n_qubits))
        self.add_wired_child(unitary_b)

    @property
    def is_exact(self) -> bool:
        return self.schmidt_rank == self.full_schmidt_rank

    @property
    def error_bound(self) -> float:
        return self.error_bound_value

    def target_statevector(self) -> np.ndarray:
        """The un-truncated, normalized input amplitudes — the ideal state
        this block approximates, per :attr:`error_bound`."""
        return self._dense_target


def _to_dense_amplitudes(
    amplitudes: Union[List[complex], Dict[Tuple[int, ...], complex]], n_qubits: int
) -> np.ndarray:
    if isinstance(amplitudes, dict):
        dense = np.zeros(2**n_qubits, dtype=complex)
        for basis_tuple, amp in amplitudes.items():
            if len(basis_tuple) != n_qubits:
                raise ValueError(
                    f"basis tuple {basis_tuple} has length {len(basis_tuple)}, expected {n_qubits}."
                )
            idx = sum(bit << i for i, bit in enumerate(basis_tuple))
            dense[idx] = amp
        return dense
    dense = np.asarray(amplitudes, dtype=complex)
    if dense.shape != (2**n_qubits,):
        raise ValueError(f"amplitudes must have length {2**n_qubits}, got {dense.shape}.")
    return dense
