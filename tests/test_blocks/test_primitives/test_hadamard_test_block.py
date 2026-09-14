"""Correctness tests for HadamardTestBlock.

Layout: q0 = ancilla, q1.. = state register.  After the test circuit
``H · state ·  C-U · H`` (plus optional Sdg before the final H for the
imaginary part), the ancilla expectation ``P(0) - P(1)`` equals
``Re⟨ψ|U|ψ⟩`` (or the imaginary part with the Sdg variant).
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import HadamardTestBlock, SimpleBlock


def _unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _ancilla_expectation(U, n_qubits: int) -> float:
    """⟨0|q0 - 1|q0⟩ on the all-zeros input.  qarpx is q0=LSB, so the
    ancilla bit is the lowest bit of the basis-state index."""
    psi0 = np.zeros(2**n_qubits, complex)
    psi0[0] = 1.0
    psi_out = U @ psi0
    p_anc0 = sum(abs(psi_out[i]) ** 2 for i in range(2**n_qubits) if (i & 1) == 0)
    p_anc1 = sum(abs(psi_out[i]) ** 2 for i in range(2**n_qubits) if (i & 1) == 1)
    return p_anc0 - p_anc1


# ── Construction validation ──────────────────────────────────────────────


def test_qubit_count_mismatch_raises():
    state = SimpleBlock(2, name="s")
    state.build()
    unit = SimpleBlock(1, name="u")
    unit.build()
    with pytest.raises(ValueError, match="must act on the same"):
        HadamardTestBlock(state=state, unitary=unit)


def test_dagger_qubit_mismatch_raises():
    state = SimpleBlock(1, name="s")
    state.build()
    unit = SimpleBlock(1, name="u")
    unit.build()
    udg = SimpleBlock(2, name="ud")
    udg.build()
    with pytest.raises(ValueError, match="unitary_dagger.*same"):
        HadamardTestBlock(state=state, unitary=unit, unitary_dagger=udg)


# ── Basic single-qubit cases ─────────────────────────────────────────────


def test_zero_state_X_ancilla_balanced():
    """⟨0|X|0⟩ = 0 → ancilla expectation = 0."""
    state = SimpleBlock(1, name="zero")
    state.build()
    unit = SimpleBlock(1, name="X")
    unit.x(0)
    unit.build()
    ht = HadamardTestBlock(state=state, unitary=unit)
    ht.build()
    U = _unitary(ht)
    assert abs(_ancilla_expectation(U, ht.n_qubits)) < 1e-12


def test_zero_state_Z_ancilla_one():
    """⟨0|Z|0⟩ = +1 → ancilla expectation = +1."""
    state = SimpleBlock(1, name="zero")
    state.build()
    unit = SimpleBlock(1, name="Z")
    unit.z(0)
    unit.build()
    ht = HadamardTestBlock(state=state, unitary=unit)
    ht.build()
    U = _unitary(ht)
    assert _ancilla_expectation(U, ht.n_qubits) == pytest.approx(1.0, abs=1e-12)


def test_plus_state_Z_ancilla_balanced():
    """⟨+|Z|+⟩ = 0 → ancilla expectation = 0."""
    state = SimpleBlock(1, name="plus")
    state.h(0)
    state.build()
    unit = SimpleBlock(1, name="Z")
    unit.z(0)
    unit.build()
    ht = HadamardTestBlock(state=state, unitary=unit)
    ht.build()
    U = _unitary(ht)
    assert abs(_ancilla_expectation(U, ht.n_qubits)) < 1e-12


# ── Imaginary-part estimation ────────────────────────────────────────────


def test_imaginary_part_for_S():
    """S = diag(1, i).  ⟨0|S|0⟩ = 1 (purely real) → Im = 0.

    For ⟨+|S|+⟩ = (1 + i)/2: Re = 0.5, Im = 0.5.  Verify both branches.
    """
    state = SimpleBlock(1, name="plus")
    state.h(0)
    state.build()
    unit = SimpleBlock(1, name="S")
    unit.s(0)
    unit.build()

    # Real part
    ht_re = HadamardTestBlock(state=state, unitary=unit, estimate_imaginary=False)
    ht_re.build()
    re_part = _ancilla_expectation(_unitary(ht_re), ht_re.n_qubits)
    assert re_part == pytest.approx(0.5, abs=1e-12)

    # Imaginary part
    ht_im = HadamardTestBlock(state=state, unitary=unit, estimate_imaginary=True)
    ht_im.build()
    im_part = _ancilla_expectation(_unitary(ht_im), ht_im.n_qubits)
    assert im_part == pytest.approx(0.5, abs=1e-12)


# ── Two-qubit unitary inner ──────────────────────────────────────────────


def test_two_qubit_inner_unitary():
    """⟨00|CX|00⟩ = 1 (CX leaves |00⟩ fixed) → ancilla expectation = +1."""
    state = SimpleBlock(2, name="zero2")
    state.build()
    unit = SimpleBlock(2, name="CX")
    unit.cx(0, 1)
    unit.build()
    ht = HadamardTestBlock(state=state, unitary=unit)
    ht.build()
    assert ht.n_qubits == 3
    U = _unitary(ht)
    assert _ancilla_expectation(U, ht.n_qubits) == pytest.approx(1.0, abs=1e-12)


# ── Optional U† branch ───────────────────────────────────────────────────


def test_with_unitary_dagger_branch():
    """When U† is provided, the circuit also applies C-U†|ancilla=0⟩ → for
    Hermitian U the two branches reinforce; we just verify the structure
    exists and the unitary is unitary."""
    state = SimpleBlock(1, name="plus")
    state.h(0)
    state.build()
    unit = SimpleBlock(1, name="X")
    unit.x(0)
    unit.build()
    udg = SimpleBlock(1, name="X_dag")
    udg.x(0)  # X is Hermitian
    udg.build()
    ht = HadamardTestBlock(state=state, unitary=unit, unitary_dagger=udg)
    ht.build()
    U = _unitary(ht)
    I = np.eye(2**ht.n_qubits)
    assert np.linalg.norm(U @ U.conj().T - I) < 1e-10


# ── Measurement option ───────────────────────────────────────────────────


def test_measure_option_emits_measure_command():
    state = SimpleBlock(1, name="zero")
    state.build()
    unit = SimpleBlock(1, name="X")
    unit.x(0)
    unit.build()
    ht = HadamardTestBlock(state=state, unitary=unit, measure=True)
    ht.build()
    cmds = ht.flatten()
    assert any(c.gate.name == "Measure" for c in cmds)
    assert ht.n_cbits >= 1


# ── set_symbols-resolved branches (lazy binding survives control) ─────────


def _resolved_ansatz(values):
    """Same symbolic 2-qubit ansatz, ``set_symbols``-bound to ``values``.

    ``set_symbols`` binds lazily on the Python wrapper (canonical C++ buffer
    stays symbolic).  Returned block reports ``.symbols == ()`` but its raw
    buffer still names the symbols — the input shape that stresses
    controlling.
    """
    b = SimpleBlock(2, name="ansatz")
    b.h(0)
    b.rz(0, qx.Param.symbol("a"))
    b.cx(0, 1)
    b.ry(1, qx.Param.symbol("b"))
    b.build()
    return b.set_symbols(dict(zip(b.symbols, values, strict=True)))


def test_controlled_resolved_unitary_uses_bound_value():
    """Expectation-value path controls the whole ``unitary``.  A
    ``set_symbols``-resolved unitary must keep its bound angles once
    controlled — regression: controlling resurrected the ansatz symbols, so
    the branch evolved with unbound (later wrong-bound) angles."""
    U = _resolved_ansatz((0.6, -0.9))
    assert U.symbols == ()  # bound on the wrapper
    inner = _unitary(U)  # oracle: the resolved unitary
    expected = inner[0, 0]  # ⟨0|U|0⟩

    state = SimpleBlock(2, name="zero")
    state.build()
    ht_re = HadamardTestBlock(state=state, unitary=U, estimate_imaginary=False)
    ht_re.build()
    ht_im = HadamardTestBlock(state=state, unitary=U, estimate_imaginary=True)
    ht_im.build()
    assert _ancilla_expectation(_unitary(ht_re), ht_re.n_qubits) == pytest.approx(
        expected.real, abs=1e-9
    )
    assert _ancilla_expectation(_unitary(ht_im), ht_im.n_qubits) == pytest.approx(
        expected.imag, abs=1e-9
    )


def test_transition_amplitude_with_resolved_ansatz_matches_overlap():
    """Multiplexor path: bra and ket are the same ansatz bound to different
    values via ``set_symbols``.  Ancilla Re/Im must equal ⟨bra|ket⟩ computed
    from the two resolved statevectors."""
    bra = _resolved_ansatz((0.3, 0.7))
    ket = _resolved_ansatz((1.1, -0.4))
    e0 = np.zeros(4, complex)
    e0[0] = 1.0
    overlap = np.vdot(_unitary(bra) @ e0, _unitary(ket) @ e0)  # ⟨bra|ket⟩

    state = SimpleBlock(2, name="zero")
    state.build()
    ht_re = HadamardTestBlock(state=state, unitary=ket, unitary_dagger=bra)
    ht_re.build()
    ht_im = HadamardTestBlock(state=state, unitary=ket, unitary_dagger=bra, estimate_imaginary=True)
    ht_im.build()
    assert _ancilla_expectation(_unitary(ht_re), ht_re.n_qubits) == pytest.approx(
        overlap.real, abs=1e-9
    )
    assert _ancilla_expectation(_unitary(ht_im), ht_im.n_qubits) == pytest.approx(
        overlap.imag, abs=1e-9
    )


def test_transition_multiplexor_reuses_common_ansatz_gates():
    """Shared branch gates are not needlessly ancilla-controlled."""
    bra = SimpleBlock(2, name="bra")
    bra.h(0)
    bra.cx(0, 1)
    bra.rz(0, 0.2)
    bra.build()

    ket = SimpleBlock(2, name="ket")
    ket.h(0)
    ket.cx(0, 1)
    ket.rz(0, -0.3)
    ket.build()

    state = SimpleBlock(2, name="zero")
    state.build()
    ht = HadamardTestBlock(
        state=state,
        unitary=ket,
        unitary_dagger=bra,
    )
    ht.build()

    commands = list(ht.flatten())
    # The common CX is emitted once.  A generic controlled-CX would be CCX.
    assert any(command.gate.name == "CX" for command in commands)
    assert not any(command.gate.name == "CCX" for command in commands)
