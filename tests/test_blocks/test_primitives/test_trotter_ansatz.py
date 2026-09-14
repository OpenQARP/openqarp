"""Correctness tests for TrotterAnsatzBlock.

Covers the convention contract — each Pauli term P with effective
coefficient c contributes ``exp(+i s c P)`` per Trotter step, composed as a
first-order Trotter product across the (symbol, operator) pairs.
"""

import numpy as np
import pytest
from scipy.linalg import expm
from sympy import Symbol

import qarpx as qx
from qarp.blocks import TrotterAnsatzBlock
from qarp.operators import FullyCommuting, NoGrouping, QubitOperator, QubitWiseCommuting

# ── Pauli matrices for hand-written reference ───────────────────────────

X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)


def _kron_le(*ops):
    """Little-endian Kronecker product: ``_kron_le(A_q0, A_q1, ...)``.

    qarpx state vectors are indexed with q0 as the least-significant bit, so
    the operator on q0 must appear on the right in the Kronecker product.
    """
    result = np.array([[1.0]], complex)
    for op in reversed(ops):
        result = np.kron(result, op)
    return result


def _unitary(block):
    block.build()
    flat = block.flatten()
    sim = qx.QarpSimulator()
    return np.array(sim.unitary_matrix(flat, block.n_qubits))


# ── Structural smoke ────────────────────────────────────────────────────


def test_single_pauli_emits_basis_change_ladder_rz():
    """X0 Y1 should decompose to H + Sdg+H basis change, CX(0,1), Rz(q1), reverse."""
    b = TrotterAnsatzBlock(
        n_qubits=2,
        qubit_exponents=[QubitOperator("X0 Y1", -1j)],
        symbols=[Symbol("a")],
        steps=1,
        time=1.0,
        imaginary=True,
        grouping=NoGrouping(),
    ).build()
    gate_seq = [c.gate.name for c in b.commands()]
    # Basis-change → ladder-in → Rz → ladder-out → reverse-basis-change
    assert gate_seq == ["H", "Sdg", "H", "CX", "Rz", "CX", "H", "H", "S"]


def test_steps_repeat_term_block():
    """steps=N ⇒ each term emitted N times with angle 1/N as scalar."""
    a = Symbol("a")
    b = TrotterAnsatzBlock(
        n_qubits=1,
        qubit_exponents=[QubitOperator("Z0", -1j)],
        symbols=[a],
        steps=3,
        time=1.0,
        imaginary=True,
        grouping=NoGrouping(),
    ).build()
    rzs = [c for c in b.commands() if c.gate.name == "Rz"]
    assert len(rzs) == 3
    # Per-step factor: -2 · c / steps with c=-1 → +2/3 per step.
    expected = +2.0 / 3
    for rz in rzs:
        param = rz.params[0]
        assert abs(param.evaluate({"a": 1.0}) - expected) < 1e-10


# ── Numerical correctness vs exp(±is P) per term ───────────────────────


@pytest.mark.parametrize("s_val", [0.0, 0.137, 0.5, 1.0])
def test_double_excitation_unitary_matches_term_by_term(s_val):
    """X0Y1 + Y0X1 (UCC double generator) — first-order Trotter is exact for
    a single symbol with both terms sharing it.  Verify per-term exp(+isc·P)
    composition (term1 @ term2 in build order)."""
    op = QubitOperator("X0 Y1", -1j) + QubitOperator("Y0 X1", 1j)
    b = TrotterAnsatzBlock(
        n_qubits=2,
        qubit_exponents=[op],
        symbols=[Symbol("a")],
        steps=1,
        time=1.0,
        imaginary=True,
        grouping=NoGrouping(),
    ).build()
    U = _unitary(b.set_symbols({Symbol("a"): s_val}))

    # Little-endian: q0 on the right.  X0Y1 = Y(q1) ⊗ X(q0).
    XY = _kron_le(X, Y)  # X on q0, Y on q1
    YX = _kron_le(Y, X)  # Y on q0, X on q1
    # Radians convention: each term contributes exp(+i s c P).
    # term1 c=-1, term2 c=+1; build order is term1 then term2 so term2 acts last
    # in matrix multiplication: U = U_term2 @ U_term1.
    expected = expm(+1j * s_val * YX) @ expm(-1j * s_val * XY)
    assert np.linalg.norm(U - expected) < 1e-10


def test_two_symbols_commute_under_first_order():
    """Two independent symbols, two non-commuting operators, first-order
    Trotter should still evaluate term-by-term in sequence."""
    Q0 = QubitOperator("X0 Y1", -1j)
    Q1 = QubitOperator("Z0 Z1", -1j)
    a, c = Symbol("a"), Symbol("c")
    b = TrotterAnsatzBlock(
        n_qubits=2,
        qubit_exponents=[Q0, Q1],
        symbols=[a, c],
        steps=1,
        time=1.0,
        imaginary=True,
        grouping=NoGrouping(),
    ).build()
    U = _unitary(b.set_symbols({a: 0.2, c: 0.3}))

    # Little-endian: q0 on right.
    XY = _kron_le(X, Y)
    ZZ = _kron_le(Z, Z)
    # Per-term c=-1 for both; build order Q0 then Q1, so matrix product is
    # U = U_Q1 @ U_Q0 (Q1 = Z0Z1 acts last).
    expected = expm(-1j * 0.3 * ZZ) @ expm(-1j * 0.2 * XY)
    assert np.linalg.norm(U - expected) < 1e-10


# ── Symbol lifecycle ────────────────────────────────────────────────────


def test_replace_symbols_renames_pair():
    a, z = Symbol("a"), Symbol("z")
    b = TrotterAnsatzBlock(
        n_qubits=1,
        qubit_exponents=[QubitOperator("Z0", -1j)],
        symbols=[a],
        imaginary=True,
        grouping=NoGrouping(),
    ).build()
    b2 = b.replace_symbols({a: z})
    b2.build()
    # ``replace_symbols`` schedules a lazy rename that materialises in
    # ``flatten()`` (after Block.__deepcopy__ stopped silently rebuilding
    # the C++ command buffer — so the raw ``commands()`` view still carries
    # the original symbol while the flattened view reflects the rename).
    rz_param = next(c for c in b2.flatten() if c.gate.name == "Rz").params[0]
    assert "z" in str(rz_param)
    assert "a" not in str(rz_param)


def test_refresh_symbols_appends_postfix():
    a = Symbol("a")
    b = TrotterAnsatzBlock(
        n_qubits=1,
        qubit_exponents=[QubitOperator("Z0", -1j)],
        symbols=[a],
        imaginary=True,
        grouping=NoGrouping(),
    ).build()
    b2 = b.refresh_symbols("_run1")
    assert [str(s) for s in b2.symbols] == ["a_run1"]


# ── Order 2 (Suzuki symmetrisation) ─────────────────────────────────────


def test_order_2_symmetrizes_two_term_sequence():
    """With two ungrouped terms A, B at order=2, the per-step layout is
    [A/2, B, A/2] — 3 Rz rotations whose halves sum to one full A and one full B."""
    a, b = Symbol("a"), Symbol("b")
    blk = TrotterAnsatzBlock(
        n_qubits=1,
        qubit_exponents=[QubitOperator("Z0", -1j), QubitOperator("Z0", -1j)],
        symbols=[a, b],
        steps=1,
        time=1.0,
        order=2,
        grouping=NoGrouping(),
        imaginary=True,
    ).build()
    rzs = [c for c in blk.commands() if c.gate.name == "Rz"]
    assert len(rzs) == 3
    # Per-step factor with c=-1, steps=1, time=1: full angle = +2
    full = +2.0
    a_half = rzs[0].params[0].evaluate({"a": 1.0, "b": 0.0})
    b_full = rzs[1].params[0].evaluate({"a": 0.0, "b": 1.0})
    a_half_2 = rzs[2].params[0].evaluate({"a": 1.0, "b": 0.0})
    assert abs(a_half - full / 2) < 1e-10
    assert abs(b_full - full) < 1e-10
    assert abs(a_half_2 - full / 2) < 1e-10


@pytest.mark.parametrize("bad_order", [3, 5, 7, 0, -1, 1.5])
def test_invalid_order_rejected(bad_order):
    """Order must be 1 or an even integer >= 2; reject odd > 1, 0, negatives,
    and non-integers at construction time (matching TrotterBlock)."""
    with pytest.raises(ValueError, match="order must be 1 or an even integer"):
        TrotterAnsatzBlock(
            n_qubits=1,
            qubit_exponents=[QubitOperator("Z0", -1j)],
            symbols=[Symbol("a")],
            order=bad_order,
            imaginary=True,
            grouping=NoGrouping(),
        )


def test_order_8_emits_runtime_warning():
    """Suzuki recursion at order >= 8 unfolds into many sub-blocks; ansatz emits
    a RuntimeWarning so users notice the gate-count cliff (same threshold as
    TrotterBlock)."""
    with pytest.warns(RuntimeWarning, match="TrotterAnsatz order=8"):
        TrotterAnsatzBlock(
            n_qubits=1,
            qubit_exponents=[QubitOperator("Z0", -1j)],
            symbols=[Symbol("a")],
            order=8,
            imaginary=True,
            grouping=NoGrouping(),
        )


# ── Higher-order Suzuki recursion (order ∈ {4, 6}) ──────────────────────
#
# ``_suzuki_recurse`` threads the per-chunk time-scale factor into the
# symbol-bound Pauli coefficient ``c`` (mirrors the TrotterBlock recursion;
# see ``trotter_block.py::TrotterAnsatzBlock._suzuki_recurse``).  Tests
# below pin the per-step gate-count structure and the accuracy improvement
# vs the order-1/2 base cases.


def test_order_4_emits_5x_the_order_2_gate_count():
    """Per Trotter step, Suzuki order 4 = 5 × order 2.  For N=2 ungrouped
    terms, order 2 emits 3 Rz per step (pre[t0]/2, mid[t1], post[t0]/2);
    order 4 emits 5×3 = 15 Rz per step."""
    a, b = Symbol("a"), Symbol("b")
    qops = [QubitOperator("Z0", -1j), QubitOperator("X0", -1j)]
    common_kwargs = dict(
        n_qubits=1,
        qubit_exponents=qops,
        symbols=[a, b],
        steps=1,
        time=1.0,
        grouping=NoGrouping(),
        imaginary=True,
    )
    o2 = TrotterAnsatzBlock(order=2, **common_kwargs).build()
    o4 = TrotterAnsatzBlock(order=4, **common_kwargs).build()
    rzs_o2 = [c for c in o2.commands() if c.gate.name == "Rz"]
    rzs_o4 = [c for c in o4.commands() if c.gate.name == "Rz"]
    assert len(rzs_o2) == 3
    assert len(rzs_o4) == 5 * 3


def test_order_4_per_term_phase_preserved_on_commuting_operator():
    """For an all-commuting Q (single-pair, all Z), Trotter is exact at any
    order — order 4 must reproduce ``exp(+i s · c · P)`` per term, just
    like order 1.  Locks in that the Suzuki coefficient scaling sums to
    unity per Pauli term (no spurious phase drift)."""
    a = Symbol("a")
    # Q = -i·Z0 - 0.5·i·Z1 + -1.3·i·Z0Z1 — all mutually commuting (Z's).
    Q = QubitOperator("Z0", -1j) + QubitOperator("Z1", -0.5j) + QubitOperator("Z0 Z1", -1.3j)
    s_val = 0.27
    expected = expm(
        +1j
        * s_val
        * (-1 * _kron_le(Z, np.eye(2)) + -0.5 * _kron_le(np.eye(2), Z) + -1.3 * _kron_le(Z, Z))
    )
    for order in [1, 2, 4]:
        blk = TrotterAnsatzBlock(
            n_qubits=2,
            qubit_exponents=[Q],
            symbols=[a],
            steps=1,
            time=1.0,
            order=order,
            grouping=NoGrouping(),
            imaginary=True,
        ).build()
        U = _unitary(blk.set_symbols({a: s_val}))
        assert np.linalg.norm(U - expected) < 1e-10, (
            f"order={order}: err={np.linalg.norm(U - expected):.3e}"
        )


def test_order_4_higher_accuracy_than_order_2_on_noncommuting_q():
    """For non-commuting Q at small symbol value, order 4 Trotter error must
    be smaller than order 2 against the exact ``exp(+i·s·Q_matrix)`` target.

    Uses ``Q = X₀ + Z₀`` (single-qubit, anti-commuting Paulis) so the Trotter
    error is genuinely finite at every order — unlike the UCC ``X₀Y₁ + Y₀X₁``
    pair which mutually commutes (their product is ``Z⊗Z`` both ways).
    """
    a = Symbol("a")
    Q = QubitOperator("X0", -1j) + QubitOperator("Z0", -1j)
    Q_matrix = -1 * X + -1 * Z  # imag=True ⇒ c = coeff.imag = -1 per term.
    s_val = 0.13  # large enough that the Trotter error is well above noise
    expected = expm(+1j * s_val * Q_matrix)

    def err(order):
        blk = TrotterAnsatzBlock(
            n_qubits=1,
            qubit_exponents=[Q],
            symbols=[a],
            steps=1,
            time=1.0,
            order=order,
            grouping=NoGrouping(),
            imaginary=True,
        ).build()
        return np.linalg.norm(_unitary(blk.set_symbols({a: s_val})) - expected)

    e1, e2, e4 = err(1), err(2), err(4)
    assert e2 < e1, f"order-2 should beat order-1: e1={e1:.3e}, e2={e2:.3e}"
    assert e4 < e2, f"order-4 should beat order-2: e2={e2:.3e}, e4={e4:.3e}"


def test_order_6_higher_accuracy_than_order_4():
    """One recursion level deeper: order 6 < order 4 at small s."""
    a = Symbol("a")
    Q = QubitOperator("X0", -1j) + QubitOperator("Z0", -1j)
    Q_matrix = -1 * X + -1 * Z
    s_val = 0.13
    expected = expm(+1j * s_val * Q_matrix)

    def err(order):
        blk = TrotterAnsatzBlock(
            n_qubits=1,
            qubit_exponents=[Q],
            symbols=[a],
            steps=1,
            time=1.0,
            order=order,
            grouping=NoGrouping(),
            imaginary=True,
        ).build()
        return np.linalg.norm(_unitary(blk.set_symbols({a: s_val})) - expected)

    assert err(6) < err(4)


def test_order_4_matches_manual_suzuki_recursion():
    """Order-4 Suzuki composition: ``S_4(t) = S_2(u·t)² · S_2((1-4u)·t) · S_2(u·t)²``
    with ``u₄ = 1/(4 - 4^(1/3))``.  Build the composition by hand from
    order-2 TrotterAnsatzBlocks (scaling the inner Pauli coefficients by
    the Suzuki time-factor) and pin equality with the order=4 build.

    For a single (symbol, Q) pair, scaling time by ``u`` in the order-2
    sub-block corresponds to scaling the per-term coefficient ``c → u·c``
    — same scaling the recursion applies internally — so this is a
    direct, implementation-independent reference check.  Uses
    anti-commuting Paulis (``X₀ + Z₀``) so the test distinguishes a correct
    Suzuki structure from any composition that happens to land on the
    same unitary by virtue of mutually-commuting terms.
    """
    a = Symbol("a")
    Q = QubitOperator("X0", -1j) + QubitOperator("Z0", -1j)
    s_val = 0.13

    u4 = 1.0 / (4.0 - 4.0 ** (1.0 / 3.0))
    sub_factors = [u4, u4, 1.0 - 4.0 * u4, u4, u4]

    # Build each Suzuki chunk as an order-2 TrotterAnsatz whose qubit_exponents
    # are scaled by the chunk's time factor.  Composed in build order.
    U_manual = np.eye(2, dtype=complex)
    for factor in sub_factors:
        Q_scaled = QubitOperator("X0", -1j * factor) + QubitOperator("Z0", -1j * factor)
        sub = TrotterAnsatzBlock(
            n_qubits=1,
            qubit_exponents=[Q_scaled],
            symbols=[a],
            steps=1,
            time=1.0,
            order=2,
            grouping=NoGrouping(),
            imaginary=True,
        ).build()
        U_manual = _unitary(sub.set_symbols({a: s_val})) @ U_manual

    blk_o4 = TrotterAnsatzBlock(
        n_qubits=1,
        qubit_exponents=[Q],
        symbols=[a],
        steps=1,
        time=1.0,
        order=4,
        grouping=NoGrouping(),
        imaginary=True,
    ).build()
    U_block = _unitary(blk_o4.set_symbols({a: s_val}))
    assert np.linalg.norm(U_block - U_manual) < 1e-10


# ── Commuting-set grouping ──────────────────────────────────────────────


def test_grouped_emission_emits_all_terms():
    """Grouped emission still produces one Rz per non-identity term — grouping
    affects ordering / symmetrisation, not the gate count."""
    a, b = Symbol("a"), Symbol("b")
    # Z0Z1 and Z0 qubit-wise commute (no X/Y disagreement on any qubit).
    qop = QubitOperator("Z0 Z1", -1j) + QubitOperator("Z0", -1j)
    blk = TrotterAnsatzBlock(
        n_qubits=2,
        qubit_exponents=[qop, qop],
        symbols=[a, b],
        steps=1,
        time=1.0,
        order=1,
        grouping=FullyCommuting(),
        imaginary=True,
    ).build()
    rzs = [c for c in blk.commands() if c.gate.name == "Rz"]
    assert len(rzs) == 4  # 2 terms × 2 symbol pairs


# ── Symbolic time (deferred build + set_time rebuild) ───────────────────


def test_symbolic_time_defers_build_until_set_time():
    """time=Symbol leaves the C++ buffer empty (qx.Param is single-variable)
    but tracks the time symbol; set_time(value) returns a fresh built block."""
    t = Symbol("t")
    a = Symbol("a")
    blk = TrotterAnsatzBlock(
        n_qubits=1,
        qubit_exponents=[QubitOperator("Z0", -1j)],
        symbols=[a],
        steps=1,
        time=t,
        order=1,
        grouping=NoGrouping(),
        imaginary=True,
    ).build()
    assert blk.is_built
    assert len(blk.commands()) == 0
    assert blk._time_symbol == t

    concrete = blk.set_time(0.5)
    assert concrete.is_built
    rzs = [c for c in concrete.commands() if c.gate.name == "Rz"]
    assert len(rzs) == 1
    # With c=-1, steps=1, time=0.5: angle coefficient = -2 · -1 · 0.5 = +1
    assert abs(rzs[0].params[0].evaluate({"a": 1.0}) - 1.0) < 1e-10
    # Time symbol was substituted; only the variational symbol remains.
    assert [str(s) for s in concrete.symbols] == ["a"]


# ── Grouping-strategy injection ─────────────────────────────────────────


def test_grouping_strategy_reorders_but_preserves_commuting_unitary():
    """XX+YY under one symbol: fully commuting, non-QWC.  FullyCommuting packs
    them into one group (QWC keeps 2); either way the emitted product equals
    the exact exponential because the terms commute."""
    from qarp.operators import FullyCommuting, QubitWiseCommuting

    q = QubitOperator("X0 X1", 1j) + QubitOperator("Y0 Y1", 0.5j)
    a = Symbol("a")
    s_val = 0.37
    XX = _kron_le(X, X)
    YY = _kron_le(Y, Y)
    expected = expm(+1j * s_val * (XX + 0.5 * YY))
    for strategy in (FullyCommuting(), QubitWiseCommuting()):
        b = TrotterAnsatzBlock(
            n_qubits=2, qubit_exponents=[q], symbols=[a], imaginary=True, grouping=strategy
        )
        U = _unitary(b.set_symbols({a: s_val}))
        assert np.linalg.norm(U - expected) < 1e-10, repr(strategy)


def test_set_time_rebuild_preserves_grouping_strategy():
    """Symbolic-time set_time() constructs a fresh block — the injected
    strategy must survive the rebuild."""
    from qarp.operators import FullyCommuting

    q = QubitOperator("X0 X1", 1j)
    b = TrotterAnsatzBlock(
        n_qubits=2,
        qubit_exponents=[q],
        symbols=[Symbol("a")],
        time=Symbol("t"),
        imaginary=True,
        grouping=FullyCommuting(),
    )
    rebuilt = b.set_time(0.5)
    assert isinstance(rebuilt.grouping, FullyCommuting)


def test_group_commuting_kwarg_is_gone():
    """``group_commuting`` was a two-valued projection of ``grouping``; the
    hard cut means the old spelling fails loudly, not silently."""
    with pytest.raises(TypeError, match="group_commuting"):
        TrotterAnsatzBlock(
            n_qubits=2,
            qubit_exponents=[QubitOperator("X0 X1", 1j)],
            symbols=[Symbol("a")],
            group_commuting=False,
        )


def test_set_time_rebuild_preserves_no_grouping():
    """set_time reconstructs the block; the injected strategy must survive the
    rebuild rather than silently reverting to the FullyCommuting default."""
    b = TrotterAnsatzBlock(
        n_qubits=2,
        qubit_exponents=[QubitOperator("X0 X1", 1j)],
        symbols=[Symbol("a")],
        time=Symbol("t"),
        grouping=NoGrouping(),
        imaginary=True,
    )
    assert isinstance(b.set_time(0.5).grouping, NoGrouping)


def _hopping_generators():
    """Two JW hopping generators; each one's ``X_pY_q``/``Y_pX_q`` pair commutes
    generally but not qubit-wise."""
    return (
        QubitOperator("X0 Y1", 0.5j) + QubitOperator("Y0 X1", -0.5j),
        QubitOperator("X1 Y2", 0.5j) + QubitOperator("Y1 X2", -0.5j),
    )


def _particle_leak(blk, sv):
    blk.build()
    U = _unitary(blk.set_symbols(sv))
    psi = U[:, 1]  # |100⟩ LSB — one particle
    return sum(abs(psi[k]) ** 2 for k in range(8) if bin(k).count("1") != 1)


@pytest.mark.parametrize("grouping", [None, NoGrouping(), FullyCommuting(), QubitWiseCommuting()])
def test_grouped_ansatz_conserves_particle_number(grouping):
    """The ansatz conserves particle number under *every* strategy.

    Grouping is confined to a single generator, so a hopping pair can never be
    separated by a term that anticommutes with it — which is the only way the
    Hamming weight leaks.  Before that scoping, QWC interleaved the two
    generators' strings and leaked ~2e-3 here.
    """
    g01, g12 = _hopping_generators()
    a, b = Symbol("a"), Symbol("b")
    blk = TrotterAnsatzBlock(
        n_qubits=3,
        qubit_exponents=[g01, g12],
        symbols=[a, b],
        imaginary=True,
        grouping=grouping,
    )
    assert _particle_leak(blk, {a: 0.6, b: 0.9}) < 1e-12


def test_particle_number_leak_is_reachable_when_partners_can_be_separated():
    """Negative control, so the test above cannot pass vacuously.

    Merging both hoppings into one generator lets QWC put ``X0Y1`` with
    ``Y1X2`` and strand ``Y0X1`` behind an anticommuting term — the leak
    reappears.  The general-commuting default keeps each pair intact.
    """
    g01, g12 = _hopping_generators()
    a = Symbol("a")
    merged = g01 + g12

    qwc = TrotterAnsatzBlock(
        n_qubits=3,
        qubit_exponents=[merged],
        symbols=[a],
        imaginary=True,
        grouping=QubitWiseCommuting(),
    )
    assert _particle_leak(qwc, {a: 0.6}) > 1e-3

    default = TrotterAnsatzBlock(n_qubits=3, qubit_exponents=[merged], symbols=[a], imaginary=True)
    assert _particle_leak(default, {a: 0.6}) < 1e-12


def test_mismatched_symbols_and_exponents_raise():
    """S-L: pairing is strict — silent truncation advertised all symbols in
    .symbols while dropping exponents."""
    exps = [qx.QubitOperator("X0 Y1"), qx.QubitOperator("Z0")]
    syms = [Symbol("a")]
    with pytest.raises(ValueError):
        TrotterAnsatzBlock(2, qubit_exponents=exps, symbols=syms, grouping=None)


@pytest.mark.parametrize(
    "dead_exponent",
    [QubitOperator("Z0") * 0.0, QubitOperator("Z0") * 1j],
    ids=["zero-coefficient", "imaginary-under-real-ansatz"],
)
def test_a_generator_emitting_no_rotation_is_dropped_from_symbols(dead_exponent):
    """A symbol that gates nothing must not be published.

    Its generator contributes no rotation — every coefficient vanishes, or is
    imaginary while ``imaginary=False`` — so publishing it left ``.symbols``
    disagreeing with ``free_symbols()`` and let ``parameter_map`` accept a
    value the circuit ignores.  The pairing in ``symbol_qop_pairs`` is kept
    whole: ``set_time()`` rebuilds the ansatz from it.
    """
    a, z = Symbol("a"), Symbol("z")
    block = TrotterAnsatzBlock(
        1, qubit_exponents=[QubitOperator("Z0"), dead_exponent], symbols=[a, z]
    )
    with pytest.warns(RuntimeWarning, match="no rotation"):
        block.build()

    assert block.symbols == (a,)
    assert block.free_symbols() == ["a"]
    assert len(block.symbol_qop_pairs) == 2
    assert block.parameter_map([0.1]) == {a: 0.1}
    with pytest.raises(ValueError, match="2 values for 1 symbols"):
        block.parameter_map([0.1, 0.2])


def test_live_generators_keep_every_symbol():
    """The discriminating control: nothing is dropped and nothing is warned
    about when both generators actually emit rotations."""
    a, z = Symbol("a"), Symbol("z")
    block = TrotterAnsatzBlock(
        1, qubit_exponents=[QubitOperator("Z0"), QubitOperator("X0")], symbols=[a, z]
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        block.build()

    assert block.symbols == (a, z)
    assert sorted(block.free_symbols()) == ["a", "z"]
