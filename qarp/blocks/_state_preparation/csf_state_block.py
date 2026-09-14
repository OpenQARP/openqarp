from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from sympy import Rational
from sympy.physics.quantum.cg import CG

from ...operators import JordanWigner, Mapping
from .._block import CompositeBlockBase
from .._prepares_known_state import prepares_known_state
from .multi_onv_state_block import MultiONVStateBlock

SpinNumber = Union[int, float, Rational]


def _half_integer(value: SpinNumber) -> Rational:
    """Coerce a spin quantum number to an exact half-integer ``Rational``,
    guarding against float half-integers that aren't bit-exact."""
    doubled = round(float(value) * 2)
    if abs(doubled - float(value) * 2) > 1e-9:
        raise ValueError(f"{value} is not an integer or half-integer.")
    return Rational(doubled, 2)


def _canonical_coupling_path(n_open_shell: int, S: Rational) -> List[Rational]:
    """``k/2 + S`` up-steps of ``1/2`` then ``k/2 − S`` down-steps to ``S``;
    the step counts are integers iff ``k − 2S`` is a non-negative even
    integer, which is exactly when ``S`` is reachable with ``k`` electrons."""
    excess = n_open_shell - 2 * S
    if S < 0 or excess < 0 or excess % 2 != 0:
        raise ValueError(
            f"S = {S} is not reachable with {n_open_shell} open-shell electrons: "
            "k - 2S must be a non-negative even integer."
        )
    n_up = int(n_open_shell / 2 + S)
    n_down = int(excess / 2)
    up = [Rational(i, 2) for i in range(1, n_up + 1)]
    down = [Rational(n_up - j, 2) for j in range(1, n_down + 1)]
    return up + down


def _genealogical_coupling(
    coupling_path: List[Rational], target_Ms: Rational
) -> Dict[Tuple[Rational, ...], float]:
    """Couple ``len(coupling_path)`` spin-1/2 electrons one at a time
    (Yamanouchi-Kotani / genealogical scheme) along the given sequence of
    intermediate total spins, via Clebsch-Gordan coefficients.

    Returns ``{(m_1, ..., m_k): amplitude}`` for the ``Ms = target_Ms``
    sector of the resulting ``S = coupling_path[-1]`` eigenstate — the
    individual-electron spin projections making up each determinant in the
    CSF.  Verified against textbook cases (2-electron singlet/triplet,
    3-electron doublet pair) before use here; see
    ``tests/test_blocks/test_state_preparation/test_csf_state_block.py``.
    """
    state: Dict[Tuple[Rational, ...], float] = {(): 1.0}
    s_prev = Rational(0)
    for s_next in coupling_path:
        new_state: Dict[Tuple[Rational, ...], float] = {}
        for spins, amplitude in state.items():
            m_prev = sum(spins, Rational(0))
            for m in (Rational(1, 2), Rational(-1, 2)):
                m_next = m_prev + m
                if abs(m_next) > s_next:
                    continue
                coefficient = float(CG(s_prev, m_prev, Rational(1, 2), m, s_next, m_next).doit())
                if abs(coefficient) < 1e-12:
                    continue
                key = spins + (m,)
                new_state[key] = new_state.get(key, 0.0) + amplitude * coefficient
        state = new_state
        s_prev = s_next
    return {spins: amp for spins, amp in state.items() if sum(spins, Rational(0)) == target_Ms}


@prepares_known_state
class CSFStateBlock(CompositeBlockBase):
    def __init__(
        self,
        n_spatial_orbitals: int,
        core_orbitals: List[int],
        open_shell_orbitals: List[int],
        coupling_path: Optional[List[SpinNumber]] = None,
        Ms: Optional[SpinNumber] = None,
        mapping: Optional[Mapping] = None,
        target_qubits: Optional[List[int]] = None,
        name: str = "CSF",
        *,
        S: Optional[SpinNumber] = None,
    ):
        r"""Prepare a spin-adapted configuration state function (CSF)
        directly, exact and ancilla-free — no prepare-then-project.

        A CSF couples the open-shell electrons' individual spins to a
        definite total spin via the genealogical (Yamanouchi-Kotani)
        scheme: ``coupling_path[k]`` is the intermediate total spin after
        coupling the first ``k + 1`` orbitals in ``open_shell_orbitals``
        (in that order), ending at ``coupling_path[-1] = S``. Unlike
        ``SzProjectorBlock`` / ``SpinSquaredProjectorBlock`` (LCU block
        encodings of the *projector*, succeeding only with probability
        ``1/λ`` — see "Projector Blocks" in ``blocks.rst``, citing Khinevich
        & Mizukami, arXiv:2601.08533), this constructs the target linear
        combination of determinants classically and hands it straight to
        ``MultiONVStateBlock`` — deterministic, no postselection.

        For 3+ open-shell electrons, more than one linearly independent CSF
        shares the same final ``S`` (e.g. two doublets for 3 open-shell
        electrons — the allyl-radical case) — ``coupling_path`` picks a
        specific one; it is not implied by ``S`` alone.  Giving ``S``
        instead selects the canonical path: couple upwards by ``1/2`` at
        every step until the largest intermediate spin needed, then
        downwards by ``1/2`` to ``S`` (``k/2 + S`` up-steps, ``k/2 − S``
        down-steps, so ``k − 2S`` must be a non-negative even integer) —
        ``[1/2, 0]`` for the two-electron singlet, ``[1/2, 1, 1/2]`` for the
        three-electron doublet, ``[1/2, 1, 3/2]`` for the quartet.

        **Sign convention.**  The textbook CSF is written in coupling order,
        ``|φ₁σ₁ φ₂σ₂ …|`` for ``open_shell_orbitals`` in the order given;
        the block emits every determinant with creation operators in
        ascending spin-orbital index (the ``MultiONVStateBlock`` convention).
        The two differ by a permutation that is the *same* for every
        determinant in the CSF — each open-shell orbital carries exactly one
        operator and the core pairs are even — so a textbook CSF may appear
        with an overall ``−1``; relative signs are unaffected.

        **Cost.**  The CSF expands into ``C(k, k/2 + Ms)`` determinants for
        ``k`` open-shell electrons — exponential in ``k`` — each riding the
        sparse-state construction; fine for ``k ≤ ~6``.  A direct CSF
        circuit (Sugisaki et al., JCTC 2019: sequential
        spin-coupling gates, no determinant expansion) is the declared
        follow-up.

        Args:
            n_spatial_orbitals: number of spatial orbitals; the qubit
                register is ``2 * n_spatial_orbitals`` spin orbitals (abab, §1).
            core_orbitals: spatial-orbital indices that are doubly occupied.
            open_shell_orbitals: spatial-orbital indices singly occupied,
                in genealogical coupling order.
            coupling_path: intermediate total spins ``S_1, ..., S_k``
                (``k = len(open_shell_orbitals)``); each step must satisfy
                ``|S_i - S_{i-1}| = 1/2`` (``S_0 = 0``).  Exactly one of
                ``coupling_path`` and ``S`` must be given.
            Ms: total spin projection; the core orbitals contribute 0, so
                this is both the open-shell and the total ``Ms``.  Required
                (defaulted only so ``coupling_path`` can be omitted).
            mapping: fermion-to-qubit mapping (default ``JordanWigner()``).
            target_qubits, name: standard Block kwargs.
            S: total spin; selects the canonical coupling path described
                above.  Keyword-only.
        """
        core = list(core_orbitals)
        open_shell = list(open_shell_orbitals)
        if Ms is None:
            raise ValueError("Ms is required.")
        if coupling_path is None:
            if S is None:
                raise ValueError("Give exactly one of coupling_path and S.")
            coupling_path = _canonical_coupling_path(len(open_shell), _half_integer(S))
        elif S is not None:
            raise ValueError("Give exactly one of coupling_path and S.")
        if len(coupling_path) != len(open_shell):
            raise ValueError("coupling_path must have one entry per open-shell orbital.")
        if set(core) & set(open_shell):
            raise ValueError("core_orbitals and open_shell_orbitals must be disjoint.")
        for p in core + open_shell:
            if not (0 <= p < n_spatial_orbitals):
                raise ValueError(f"orbital index {p} out of range for {n_spatial_orbitals}.")

        path = [_half_integer(s) for s in coupling_path]
        ms = _half_integer(Ms)
        s_prev = Rational(0)
        for s_next in path:
            if abs(s_next - s_prev) != Rational(1, 2) or s_next < 0:
                raise ValueError(
                    f"invalid coupling step {s_prev} -> {s_next}: coupling a spin-1/2 electron "
                    "must change the total spin by exactly 1/2."
                )
            s_prev = s_next
        if abs(ms) > s_prev:
            raise ValueError(f"|Ms| = {abs(ms)} exceeds the total spin S = {s_prev}.")

        spin_terms = _genealogical_coupling(path, ms)
        if not spin_terms:
            raise ValueError(f"Ms = {Ms} is not reachable for coupling_path = {coupling_path}.")

        n = n_spatial_orbitals
        onv_coefficients: Dict[Tuple[int, ...], complex] = {}
        for spins, amplitude in spin_terms.items():
            onv = [0] * (2 * n)
            for p in core:
                onv[2 * p] = 1
                onv[2 * p + 1] = 1
            for p, m in zip(open_shell, spins, strict=True):
                if m > 0:
                    onv[2 * p] = 1
                else:
                    onv[2 * p + 1] = 1
            onv_coefficients[tuple(onv)] = complex(amplitude)

        self.core_orbitals = core
        self.open_shell_orbitals = open_shell
        self.coupling_path = path
        self.Ms = ms
        self.mapping = mapping if mapping is not None else JordanWigner()
        self.onv_coefficients = onv_coefficients

        super().__init__(
            n_qubits=2 * n,
            target_qubits=target_qubits,
            name=name,
        )

    def build_vanilla(self) -> None:
        self.add_wired_child(MultiONVStateBlock(self.onv_coefficients, mapping=self.mapping))

    def target_statevector(self) -> np.ndarray:
        """The mapped, normalized CI coefficients — same convention as
        ``MultiONVStateBlock.target_statevector``, computed classically from
        ``onv_coefficients`` rather than read back from the built circuit."""
        psi = np.zeros(2**self.n_qubits, dtype=complex)
        norm_sq = sum(abs(amp) ** 2 for amp in self.onv_coefficients.values())
        for onv, amplitude in self.onv_coefficients.items():
            basis = self.mapping.encode_state(list(onv))
            idx = sum(bit << i for i, bit in enumerate(basis))
            psi[idx] = amplitude / norm_sq**0.5
        return psi
