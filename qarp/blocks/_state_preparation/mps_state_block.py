from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from .._block import CompositeBlockBase
from .._prepares_known_state import prepares_known_state
from .._primitives.synthesized_unitary_block import SynthesizedUnitaryBlock
from ._linalg import _complex_gram_schmidt_complete


def _validate_mps(tensors: List[NDArray]) -> None:
    if not tensors:
        raise ValueError("tensors cannot be empty.")
    if tensors[0].shape[0] != 1:
        raise ValueError(
            f"open-boundary MPS needs a trivial left bond (1), got {tensors[0].shape[0]}."
        )
    if tensors[-1].shape[2] != 1:
        raise ValueError(
            f"open-boundary MPS needs a trivial right bond (1), got {tensors[-1].shape[2]}."
        )
    for k, t in enumerate(tensors):
        if t.ndim != 3 or t.shape[1] != 2:
            raise ValueError(f"tensor {k} must have shape (chi_left, 2, chi_right), got {t.shape}.")
        if k > 0 and tensors[k - 1].shape[2] != t.shape[0]:
            raise ValueError(
                f"bond mismatch between site {k - 1} (right={tensors[k - 1].shape[2]}) "
                f"and site {k} (left={t.shape[0]})."
            )


def _dense_from_mps(tensors: List[NDArray]) -> np.ndarray:
    """Contract an open-boundary MPS into a dense statevector, LSB-indexed
    (site 0 = qubit 0, §1) — the block's mathematical definition, evaluated
    directly from the tensors rather than from any circuit."""
    n = len(tensors)
    result = tensors[0].reshape(2, -1)
    for k in range(1, n):
        result = np.tensordot(result, tensors[k], axes=([-1], [0]))
    result = result.reshape([2] * n)
    result = np.transpose(result, axes=list(range(n))[::-1])
    return result.reshape(-1)


def _right_canonicalize(tensors: List[NDArray]) -> Tuple[List[NDArray], complex]:
    """Sweep right-to-left so every site, reshaped ``(chi_left, 2*chi_right)``,
    has orthonormal rows — the gauge :func:`_build_site_unitary` needs to
    complete each tensor into a genuine isometry. Preserves the represented
    state exactly (a gauge transformation); returns the leftover norm/phase
    separately rather than folding it back in, so site 0 stays unit-norm."""
    tensors = [np.asarray(t, dtype=complex) for t in tensors]
    n = len(tensors)
    for k in range(n - 1, 0, -1):
        chi_l, d, chi_r = tensors[k].shape
        matrix = tensors[k].reshape(chi_l, d * chi_r)
        u, s, vh = np.linalg.svd(matrix, full_matrices=False)
        rank = len(s)
        tensors[k] = vh.reshape(rank, d, chi_r)
        remainder = u * s
        tensors[k - 1] = np.einsum("asc,cb->asb", tensors[k - 1], remainder)
    norm = np.linalg.norm(tensors[0])
    tensors[0] = tensors[0] / norm
    return tensors, norm


def _build_site_unitary(tensor: NDArray, dim_bond: int) -> NDArray:
    """A ``(2*dim_bond) x (2*dim_bond)`` unitary ``U`` such that
    ``U|b⟩_bond|0⟩_phys = Σ_{s,b'} tensor[b,s,b'] |b'⟩_bond|s⟩_phys`` for
    every reachable input bond value ``b < chi_left`` — a Gram-Schmidt
    completion of the (right-canonical, hence isometric) tensor's columns,
    the same technique ``SlaterDeterminantBlock``/``LowRankStateBlock`` use.
    Bond qubits are the low bits, the physical qubit the high bit.
    """
    chi_l, d, chi_r = tensor.shape
    full_dim = dim_bond * d
    input_indices = list(range(chi_l))  # phys=0 slot: index = 0*dim_bond + b = b
    columns = np.zeros((full_dim, chi_l), dtype=complex)
    for b in range(chi_l):
        for s in range(d):
            for bp in range(chi_r):
                row = s * dim_bond + bp
                columns[row, b] = tensor[b, s, bp]
    completed = _complex_gram_schmidt_complete(columns)
    unitary = np.zeros((full_dim, full_dim), dtype=complex)
    for j, idx in enumerate(input_indices):
        unitary[:, idx] = completed[:, j]
    other_indices = [i for i in range(full_dim) if i not in input_indices]
    for j, idx in enumerate(other_indices):
        unitary[:, idx] = completed[:, chi_l + j]
    return unitary


@prepares_known_state
class MPSStateBlock(CompositeBlockBase):
    def __init__(
        self,
        tensors: List[NDArray],
        target_qubits: Optional[List[int]] = None,
        name: str = "MPSState",
    ):
        r"""Prepare a state given as an open-boundary matrix product state,
        exact — "I ran DMRG, give me a circuit" (Schön et al., PRL **95**,
        110503 (2005), "exact sequential"; see also Ran, PRA **101**, 032310
        (2020); Rudolph et al., arXiv:2209.00595). Unlike
        ``VUMPOBrickworkBlock`` (a variational ansatz, whose parameters are
        optimized against a target), this takes an already-known MPS —
        e.g. from ``quimb``'s DMRG output, or a truncated dense vector — and
        emits an exact circuit for whatever it represents.

        A bond register of ``⌈log₂(max bond dimension)⌉`` qubits threads
        through the chain: right-canonicalizing the tensors turns each site
        into an isometry (Gram-Schmidt-completed to a genuine unitary,
        matching ``SlaterDeterminantBlock``/``LowRankStateBlock``'s
        technique), applied sequentially, bond register first then that
        site's physical qubit. The trivial right boundary (the MPS's own
        ``chi_N = 1``) forces the bond register back to ``|0…0⟩``
        deterministically at the end — ``ancilla_postselection = None`` —
        so this needs no approximation dial: it is exact for whatever the
        input tensors represent (any inaccuracy from how *that* MPS was
        obtained — e.g. a DMRG bond-dimension truncation — is a property of
        the input, not of this block).

        **Cost (measured, CNOTs after decomposition on
        ``clifford_t_rz_gateset``, ``O0``).**  One ``(2χ) × (2χ)`` unitary
        per site, so linear in ``N``: for ``χ = 2`` it is 24 at ``N = 4``
        (dense ``SynthesizedStateBlock``: 28), 36 at ``N = 6`` (124) and 48
        at ``N = 8`` (508) — plus the ``⌈log₂χ⌉`` bond-register ancillas,
        which the ancilla-free sliding-window form (Ran 2020) removes; that
        rewrite, and synthesising each site as an isometry rather than a
        full unitary, are declared follow-ups.

        Args:
            tensors: open-boundary MPS as a list of ``N`` site tensors, each
                shape ``(chi_left, 2, chi_right)``; ``chi_left = 1`` for
                site 0, ``chi_right = 1`` for site ``N - 1``, and adjacent
                bond dimensions must match. Site ``k`` = qubit ``k``, LSB
                (§1). Need not already be in canonical form or have minimal
                bond dimension — canonicalized internally.
            target_qubits, name: standard Block kwargs.
        """
        _validate_mps(tensors)
        self._original_tensors = [np.asarray(t, dtype=complex) for t in tensors]
        self.n_sites = len(tensors)

        canonical, _ = _right_canonicalize(self._original_tensors)
        self._canonical_tensors = canonical
        max_bond = max(max(t.shape[0], t.shape[2]) for t in canonical)
        self.bond_qubits = int(np.ceil(np.log2(max_bond)))

        super().__init__(
            n_qubits=self.bond_qubits + self.n_sites, target_qubits=target_qubits, name=name
        )

    @property
    def state_qubits(self) -> Tuple[int, ...]:
        return tuple(range(self.bond_qubits, self.n_qubits))

    def build_vanilla(self) -> None:
        dim_bond = 2**self.bond_qubits
        for k, tensor in enumerate(self._canonical_tensors):
            unitary = _build_site_unitary(tensor, dim_bond)
            site_block = SynthesizedUnitaryBlock(unitary)
            site_block.target_qubits = list(range(self.bond_qubits)) + [self.bond_qubits + k]
            self.add_wired_child(site_block)

    def target_statevector(self) -> np.ndarray:
        """The state the *original* (pre-canonicalization) tensors
        represent, normalized — canonicalization is a gauge transformation
        of the internal circuit construction, not of what's being prepared."""
        dense = _dense_from_mps(self._original_tensors)
        return dense / np.linalg.norm(dense)
