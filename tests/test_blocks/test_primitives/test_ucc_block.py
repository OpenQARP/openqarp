"""Correctness tests for UCCBlock.

UCCBlock is a Pattern B (composite) wrapper around a single
``TrotterAnsatzBlock`` child that Trotterises the JW-encoded UCC excitation
generators.  Because the wrapper is structural — it only sets up the inner
block and forwards through ``add_child`` — these tests focus on the
end-to-end equivalence with a hand-built ``TrotterAnsatzBlock`` and on the
basic composite contract (build → set_symbols → flatten → unitary).
"""

import numpy as np
import pytest
from sympy import Symbol

import qarpx as qx
from qarp.blocks import TrotterAnsatzBlock, UCCBlock
from qarp.operators import FullyCommuting, NoGrouping, QubitWiseCommuting


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


@pytest.fixture
def onv4():
    return [1, 1, 0, 0]


# ── Construction + structural ────────────────────────────────────────────


def test_ucc_construction_collects_singles_and_doubles(onv4):
    """4 qubits, 2 occupied → spin-conserving SD has 2 singles + 1 double."""
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        spin_conserving=True,
        generalised=False,
    )
    assert ucc.n_qubits == 4
    assert len(ucc.symbols) == 3
    assert len(ucc.qubit_exponents) == 3


def test_ucc_doubles_only(onv4):
    ucc = UCCBlock(occupation_number_vector=onv4, singles=False, doubles=True, generalised=False)
    assert len(ucc.symbols) == 1


def test_ucc_singles_only(onv4):
    ucc = UCCBlock(occupation_number_vector=onv4, singles=True, doubles=False, generalised=False)
    assert len(ucc.symbols) == 2


# ── Equivalence with hand-built TrotterAnsatzBlock ───────────────────────


def test_ucc_unitary_matches_trotter_ansatz_block(onv4):
    """UCCBlock and a directly-constructed TrotterAnsatzBlock with the same
    operator pool / symbols / settings must produce bit-identical command
    streams and identical unitaries (the wrapper adds no gates of its own)."""
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        spin_conserving=True,
        generalised=False,
    )
    ucc.build()

    # Reference construction must use the pairing surface — the sorted
    # `symbols` registry deliberately does not carry symbol↔operator order.
    pair_symbols = [s for s, _ in ucc.symbol_qop_pairs]
    tab = TrotterAnsatzBlock(
        n_qubits=ucc.n_qubits,
        qubit_exponents=ucc.qubit_exponents,
        symbols=pair_symbols,
        steps=1,
        time=1.0,
        order=1,
        grouping=ucc.grouping,
        imaginary=True,
    )
    tab.build()

    # Symbolic equivalence: Rz parameter strings line up term-for-term.
    ucc_rzs = [str(c.params[0]) for c in ucc.flatten() if c.gate.name == "Rz"]
    tab_rzs = [str(c.params[0]) for c in tab.flatten() if c.gate.name == "Rz"]
    assert ucc_rzs == tab_rzs

    # Numeric equivalence after substitution (uses the C++ CompositeBlock
    # set_symbols recursive override — regression check for the bug fixed
    # in this same commit).
    vals = {s: 0.1 * (i + 1) for i, s in enumerate(ucc.symbols)}
    ucc_sub = ucc.set_symbols(vals)
    ucc_sub.build()
    tab_sub = tab.set_symbols(vals)
    tab_sub.build()

    U_ucc = _unitary(ucc_sub)
    U_tab = _unitary(tab_sub)
    assert np.linalg.norm(U_ucc - U_tab) < 1e-12


def test_ucc_unitary_is_unitary(onv4):
    """Sanity: the parametrised UCC ansatz evaluates to a unitary matrix."""
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        generalised=False,
    )
    ucc.build()
    sub = ucc.set_symbols({s: 0.137 * (i + 1) for i, s in enumerate(ucc.symbols)})
    sub.build()
    U = _unitary(sub)
    I = np.eye(2**ucc.n_qubits)
    assert np.linalg.norm(U @ U.conj().T - I) < 1e-10


def test_ucc_actually_evolves_state_with_theta(onv4):
    """Regression: ``set_symbols`` on UCCBlock must actually rotate the state
    away from |HF⟩ when θ ≠ 0.

    Previously, the generalised UCC enumeration
    the same antihermitised generator twice (with opposite signs), so setting
    every parameter to the same value gave exact cancellation and U → I (or
    U |HF⟩ → |HF⟩).  This test would have failed before the dedup fix."""
    from qarp.blocks import CompositeBlock, MappedONVStateBlock

    ucc = UCCBlock(occupation_number_vector=onv4, singles=True, doubles=True, generalised=True)
    wfn = CompositeBlock([MappedONVStateBlock(onv4), ucc]).build()
    theta = 0.3 * np.pi  # radians — large enough to visibly rotate |HF⟩
    sub = wfn.set_symbols({s: theta for s in wfn.symbols}).build()

    sim = qx.QarpSimulator()
    sv = np.array(sim.statevector(sub.flatten(), wfn.n_qubits))

    # |HF⟩ amplitude must drop substantially below 1.
    hf_idx = sum(b << q for q, b in enumerate(onv4))  # qarpx LSB
    assert abs(sv[hf_idx]) ** 2 < 0.5, (
        f"UCC ansatz at θ=0.3π left |HF⟩ amplitude at "
        f"{abs(sv[hf_idx]) ** 2:.4f} (≥ 0.5) — parameters not driving the "
        f"unitary.  See the generalised-UCC dedup bug (pytket_removal_plan §3.3 #14)."
    )


def test_ucc_refresh_symbols_preserves_unitary(onv4):
    """Regression: ``refresh_symbols`` must not change the parametric unitary
    when the new parameter values mirror the old.

    ``Block::replace_symbols`` must carry the linear coefficient across: rewriting
    ``Rx(0.5·θ)`` as ``Rx(φ)`` would break every parametric ansatz that goes
    through ``refresh_symbols`` (UCC / Trotter parameters carry a non-unity
    ``per_step_factor`` coefficient by design)."""
    from qarp.blocks import CompositeBlock, MappedONVStateBlock

    ucc = UCCBlock(occupation_number_vector=onv4, singles=True, doubles=True, generalised=True)
    wfn = CompositeBlock([MappedONVStateBlock(onv4), ucc]).build()
    refreshed = wfn.refresh_symbols("_run1").build()

    vals = {s: 0.3 for s in wfn.symbols}
    refreshed_vals = {Symbol(str(s) + "_run1"): v for s, v in vals.items()}

    U_orig = _unitary(wfn.set_symbols(vals).build())
    U_refreshed = _unitary(refreshed.set_symbols(refreshed_vals).build())
    assert np.linalg.norm(U_orig - U_refreshed) < 1e-12


# ── Independent reference: openfermion expm of antihermitised generators ─


def _ucc_reference_unitary(onv, params, *, generalised, n_qubits):
    """Build the exact (no-Trotter) UCC unitary via dense matrices + scipy.expm.

    Mirrors what ``UCCBlock`` constructs symbolically:

      * Build the antihermitised fermion excitation generators via
        ``qarp.operators.ucc`` using the same singles/doubles/generalised flags.
      * JW-encode each generator into a ``QubitOperator`` whose coefficients
        are purely imaginary (the operators are anti-Hermitian).
      * Convert each into a Hermitian matrix ``H_k`` by taking the imaginary
        part of every Pauli coefficient — exactly what ``TrotterAnsatzBlock``
        does in its ``imaginary=True`` branch.
      * Exponentiate the parameter-weighted sum: ``U_ref = expm(i Σ_k s_k H_k)``.

    The unit convention matches ``TrotterAnsatzBlock``'s radians convention
    (``per_step_factor = -2 · time / steps`` × ``exp(-i · angle/2 · P)``).

    Returns the unitary in qarpx LSB convention (``sparse_matrix``).
    """
    from scipy.linalg import expm

    from qarp.operators import JordanWigner, QubitOperator
    from qarp.operators.ucc import ucc_doubles, ucc_singles

    sops, ssyms = ucc_singles(onv, spin_conserving=True, generalised=generalised)
    dops, dsyms = ucc_doubles(onv, spin_conserving=True, generalised=generalised, paired=False)
    all_ops = sops + dops
    all_syms = ssyms + dsyms
    qexps = JordanWigner().encode_operator(all_ops)

    H_total = np.zeros((1 << n_qubits, 1 << n_qubits), dtype=complex)
    for sym, qop in zip(all_syms, qexps, strict=True):
        H_k = QubitOperator()
        for term, c in qop.terms.items():
            H_k += QubitOperator(term, complex(c).imag)
        H_k_mat = H_k.sparse_matrix(n_qubits).toarray()
        H_total += params[sym] * H_k_mat

    return expm(1j * H_total)


@pytest.mark.parametrize("generalised", [False, True])
def test_ucc_unitary_matches_openfermion_reference(onv4, generalised):
    """End-to-end correctness: UCC's compiled unitary matches an independent
    OpenFermion reference (``expm(i Σ_k s_k H_k)``) within the first-order
    Trotter error bound.

    This catches regressions in EITHER stage independently:
      * ``ucc_singles``/``ucc_doubles`` — wrong operator pool would shift the
        reference Hamiltonian and the test would fail.
      * ``TrotterAnsatzBlock`` — wrong angle convention or basis-change ladder
        would shift the qarpx unitary and the test would fail.
      * The C++ ``Block::replace_symbols`` (used downstream of this block) —
        any coefficient drop on rename would make the qarpx unitary differ.

    The earlier ``test_ucc_unitary_matches_trotter_ansatz_block`` only
    cross-checks UCC against TrotterAnsatzBlock; both share the encoding
    pipeline so a bug in either could go silent there.
    """
    n_qubits = 4
    # Small θ × many Trotter steps keeps the first-order Trotter error
    # below the chosen tolerance (~θ²/steps for n=1).
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        generalised=generalised,
        steps=10,
    )
    ucc.build()
    params = {s: 0.001 * (i + 1) for i, s in enumerate(ucc.symbols)}
    sub = ucc.set_symbols(params)
    sub.build()
    U_ucc = _unitary(sub)

    U_ref = _ucc_reference_unitary(onv4, params, generalised=generalised, n_qubits=n_qubits)
    err = np.linalg.norm(U_ucc - U_ref)
    assert err < 1e-5, (
        f"UCC unitary differs from openfermion reference by {err:.6e} "
        f"(generalised={generalised}); expected first-order Trotter error "
        f"< 1e-5 at θ_max=0.001 with steps=10."
    )


# ── Symbol lifecycle ─────────────────────────────────────────────────────


def test_ucc_refresh_symbols_appends_postfix(onv4):
    ucc = UCCBlock(occupation_number_vector=onv4, singles=True, doubles=True)
    ucc.build()
    refreshed = ucc.refresh_symbols("_run1")
    refreshed.build()
    for s in refreshed.symbols:
        assert str(s).endswith("_run1")


def test_ucc_symbol_postfix_at_construction(onv4):
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        symbol_postfix="_v0",
    )
    for s in ucc.symbols:
        assert str(s).endswith("_v0")


# ── grouping ────────────────────────────────────────────────────────────


def _generator_product_reference(ucc, vals):
    """``∏_k exp(+i s_k C_k)`` over the pairing surface, via scipy.

    Independent of the block's own emission: each generator's dense matrix
    comes from openfermion, exponentiated with ``scipy.expm``.
    """
    from scipy.linalg import expm

    from qarp.operators import QubitOperator

    dim = 2**ucc.n_qubits
    ref = np.eye(dim, dtype=complex)
    for symbol, qop in ucc.symbol_qop_pairs:
        A = np.zeros((dim, dim), dtype=complex)
        for term, coeff in qop.terms.items():
            if not term:
                continue
            P = np.array(QubitOperator(term, 1.0).sparse_matrix(ucc.n_qubits).todense())
            A = A + float(coeff.imag) * P
        ref = expm(1j * float(vals[symbol]) * A) @ ref
    return ref


@pytest.mark.parametrize("grouping", [None, NoGrouping(), FullyCommuting()])
def test_ucc_order_1_matches_scipy_generator_product(onv4, grouping):
    """At order 1 the UCC circuit is exactly ``∏_k exp(+i s_k C_k)`` over the
    generation-ordered pairing surface.  Grouping reorders commuting factors,
    which leaves the product invariant — asserted against scipy, not against
    another grouping's output."""
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        generalised=False,
        grouping=grouping,
    )
    ucc.build()
    vals = {s: 0.11 * (i + 1) for i, s in enumerate(ucc.symbols)}
    bound = ucc.set_symbols(vals)
    bound.build()
    assert np.linalg.norm(_unitary(bound) - _generator_product_reference(ucc, vals)) < 1e-10


def test_ucc_grouping_reaches_the_child_block(onv4):
    """``grouping=`` is forwarded to the inner TrotterAnsatzBlock — before the
    consolidation UCCBlock had no way to select a strategy at all."""
    for strategy in (NoGrouping(), QubitWiseCommuting(), FullyCommuting()):
        ucc = UCCBlock(occupation_number_vector=onv4, grouping=strategy)
        ucc.build()
        (child,) = ucc.children()
        assert child.grouping is strategy


def test_ucc_grouping_is_visible_in_circuit_cost(onv4):
    """The forwarded strategy is not decorative: grouping commuting generators
    shares one basis-change Clifford, so the default emits strictly fewer
    gates than the termwise partition while keeping one Rz per term."""

    def counts(strategy):
        ucc = UCCBlock(occupation_number_vector=onv4, singles=True, doubles=True, grouping=strategy)
        ucc.build()
        cmds = list(ucc.flatten())
        return len(cmds), sum(1 for c in cmds if c.gate.name == "Rz")

    n_grouped, rz_grouped = counts(FullyCommuting())
    n_termwise, rz_termwise = counts(NoGrouping())
    assert rz_grouped == rz_termwise
    assert n_grouped < n_termwise, f"grouping saved nothing: {n_grouped} vs {n_termwise}"


def test_group_commuting_kwarg_is_gone(onv4):
    """``group_commuting`` was a two-valued projection of ``grouping``; the
    hard cut means the old spelling fails loudly, not silently."""
    with pytest.raises(TypeError, match="group_commuting"):
        UCCBlock(occupation_number_vector=onv4, group_commuting=False)


# ── Higher-order Trotter path ───────────────────────────────────────────


@pytest.mark.parametrize("order", [2, 4])
def test_ucc_higher_order_unitary_is_unitary(onv4, order):
    """order ∈ {2, 4} forwards through ``TrotterAnsatzBlock`` Suzuki recursion;
    the block must still evaluate to a unitary."""
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        order=order,
    )
    ucc.build()
    sub = ucc.set_symbols({s: 0.05 * (i + 1) for i, s in enumerate(ucc.symbols)})
    sub.build()
    U = _unitary(sub)
    I = np.eye(2**ucc.n_qubits)
    assert np.linalg.norm(U @ U.conj().T - I) < 1e-10


def test_ucc_order_2_is_closer_to_reference_than_order_1(onv4):
    """Strang (order-2) Trotter has O(θ³) error per step vs O(θ²) for order-1.
    At the same θ and step count, order-2 must match the openfermion reference
    more tightly than order-1."""
    params_template = lambda u: {s: 0.05 * (i + 1) for i, s in enumerate(u.symbols)}
    n_qubits = 4

    ucc_o1 = UCCBlock(occupation_number_vector=onv4, singles=True, doubles=True, order=1, steps=1)
    ucc_o2 = UCCBlock(occupation_number_vector=onv4, singles=True, doubles=True, order=2, steps=1)
    ucc_o1.build()
    ucc_o2.build()
    params = params_template(ucc_o1)

    U_o1 = _unitary(ucc_o1.set_symbols(params).build())
    U_o2 = _unitary(ucc_o2.set_symbols(params).build())
    U_ref = _ucc_reference_unitary(onv4, params, generalised=False, n_qubits=n_qubits)

    err_o1 = np.linalg.norm(U_o1 - U_ref)
    err_o2 = np.linalg.norm(U_o2 - U_ref)
    assert err_o2 < err_o1, f"order=2 error {err_o2:.2e} not smaller than order=1 {err_o1:.2e}"


# ── Paired doubles ──────────────────────────────────────────────────────


def test_ucc_paired_doubles_reduces_operator_count(onv4):
    """``paired_doubles=True`` restricts the doubles pool to spatially-paired
    excitations, which on a 4-spin-orbital reference yields strictly fewer
    doubles than the full spin-conserving pool."""
    full = UCCBlock(
        occupation_number_vector=onv4,
        singles=False,
        doubles=True,
        paired_doubles=False,
    )
    paired = UCCBlock(
        occupation_number_vector=onv4,
        singles=False,
        doubles=True,
        paired_doubles=True,
    )
    assert len(paired.symbols) <= len(full.symbols)


def test_ucc_paired_doubles_is_unitary(onv4):
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        paired_doubles=True,
    )
    ucc.build()
    sub = ucc.set_symbols({s: 0.1 for s in ucc.symbols})
    sub.build()
    U = _unitary(sub)
    I = np.eye(2**ucc.n_qubits)
    assert np.linalg.norm(U @ U.conj().T - I) < 1e-10


# ── Alternative mappings ────────────────────────────────────────────────


@pytest.mark.parametrize("mapping_name", ["Parity", "BravyiKitaev"])
def test_ucc_alternative_mapping_is_unitary(onv4, mapping_name):
    """The mapping kwarg routes the JW-encoded operator pool through an
    alternative fermion→qubit encoding (Parity / BK).  The resulting ansatz
    must still be unitary.  Parametrized over plain names, constructed in the
    test body (module-scope qarpx objects leak — see the repo landmine)."""
    from qarp.operators import BravyiKitaev, Parity

    mapping = Parity(len(onv4)) if mapping_name == "Parity" else BravyiKitaev()
    ucc = UCCBlock(
        occupation_number_vector=onv4,
        singles=True,
        doubles=True,
        mapping=mapping,
    )
    ucc.build()
    sub = ucc.set_symbols({s: 0.1 for s in ucc.symbols})
    sub.build()
    U = _unitary(sub)
    I = np.eye(2**ucc.n_qubits)
    assert np.linalg.norm(U @ U.conj().T - I) < 1e-10
