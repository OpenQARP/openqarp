"""Correctness tests for SelectBlock.

SelectBlock layout: control register at qubits ``[0..num_controls)``,
target register at ``[num_controls..]``.  For each control basis state |i⟩,
applying SelectBlock to |i⟩|ψ⟩ should produce |i⟩(U_i|ψ⟩) where U_i is the
i-th unitary in ``unitaries`` and ``i`` ranges over
``itertools.product([False, True], repeat=num_controls)`` in lexicographic
order.
"""

import itertools

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import MeasureBlock, SelectBlock, SimpleBlock

# ── Pauli matrix references for ground-truth unitaries ──────────────────

I = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]], complex)
Z = np.array([[1, 0], [0, -1]], complex)
PAULI = {"I": I, "X": X, "Y": Y, "Z": Z}


def _unitary(block):
    block.build()  # idempotent — safe for already-built blocks
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def _kron_le(*ops):
    """Little-endian Kronecker: ``_kron_le(A_q0, A_q1, ...)``  — q0 is LSB."""
    result = np.array([[1.0]], complex)
    for op in reversed(ops):
        result = np.kron(result, op)
    return result


def _expected_unitary(unitaries, num_controls: int) -> np.ndarray:
    """Build the expected diag-block unitary for SelectBlock.

    For ctrl basis state ``c`` (taken in the same lex order as
    ``itertools.product([False, True], repeat=num_controls)``):
        U_total acts as I_ctrl ⊗ (e^{iφ_c} · U_c) on the target register,
    where ``φ_c`` is the phase entry for unitary ``c``.
    """
    target_size = len(unitaries[0][1])
    n = num_controls + target_size
    dim = 2**n
    target_dim = 2**target_size

    ctrl_states = list(itertools.product([False, True], repeat=num_controls))

    U = np.zeros((dim, dim), complex)
    for c_idx, c_bits in enumerate(ctrl_states):
        # qarpx is q0=LSB, so the control register occupies the lowest
        # `num_controls` bits of the basis-state index.
        c_int = sum(int(b) << i for i, b in enumerate(c_bits))

        if c_idx < len(unitaries):
            phase, pauli_str = unitaries[c_idx]
            ops = [PAULI[ch] for ch in pauli_str]
            U_c = (np.exp(1j * (phase or 0))) * _kron_le(*ops)
        else:
            U_c = np.eye(target_dim, dtype=complex)

        for t_in in range(target_dim):
            for t_out in range(target_dim):
                full_in = (t_in << num_controls) | c_int
                full_out = (t_out << num_controls) | c_int
                U[full_out, full_in] = U_c[t_out, t_in]
    return U


# ── Construction validation ──────────────────────────────────────────────


def test_empty_unitaries_raises():
    with pytest.raises(ValueError, match="cannot be empty"):
        SelectBlock(unitaries=[], num_controls=1)


def test_too_many_unitaries_raises():
    """3 unitaries with num_controls=1 (only 2 ctrl-states available)."""
    with pytest.raises(ValueError, match="ctrl-states available"):
        SelectBlock(unitaries=[(0, "X"), (0, "Y"), (0, "Z")], num_controls=1).build()


def test_one_control_with_nonzero_phase():
    """SelectBlock entries with non-zero phase must lower under control: the
    phase flows through PauliBlock → ControlledBlock → multi-controlled
    GPhase (Barenco) → P / CP, rather than hitting ``NotImplementedError``."""
    unitaries = [(0.0, "X"), (np.pi / 3, "Z")]
    sel = SelectBlock(unitaries=unitaries, num_controls=1)
    sel.build()
    U = _unitary(sel)
    expected = _expected_unitary(unitaries, 1)
    assert np.linalg.norm(U - expected) < 1e-12


def test_two_controls_with_nonzero_phases():
    """C^2 lowering of a SelectBlock with non-zero phases on every entry —
    exercises the recursive Barenco path on ``GPhase`` (which reduces to
    ``C^{n-1}(P(θ))``) inside the per-entry ControlledBlock."""
    unitaries = [
        (0.0, "X"),
        (np.pi / 3, "Z"),
        (-np.pi / 5, "Y"),
        (np.pi / 7, "I"),
    ]
    sel = SelectBlock(unitaries=unitaries, num_controls=2)
    sel.build()
    U = _unitary(sel)
    expected = _expected_unitary(unitaries, 2)
    assert np.linalg.norm(U - expected) < 1e-12


# ── Single-control selection ─────────────────────────────────────────────


def test_one_control_selects_pauli_pair():
    """num_controls=1 with U_0=X, U_1=Z on a single target qubit."""
    sel = SelectBlock(unitaries=[(0, "X"), (0, "Z")], num_controls=1)
    sel.build()
    U = _unitary(sel)
    expected = _expected_unitary([(0, "X"), (0, "Z")], 1)
    assert np.linalg.norm(U - expected) < 1e-12


def test_one_control_two_qubit_pauli_strings():
    """num_controls=1, target_size=2."""
    unitaries = [(0, "XI"), (0, "ZZ")]
    sel = SelectBlock(unitaries=unitaries, num_controls=1)
    sel.build()
    U = _unitary(sel)
    expected = _expected_unitary(unitaries, 1)
    assert np.linalg.norm(U - expected) < 1e-12


# ── Multi-control selection (the new C++ path) ───────────────────────────


def test_two_controls_four_paulis_on_two_targets():
    """4 unitaries on 2 target qubits with 2 controls — exercises the
    multi-control lowering for X, Y, Z."""
    unitaries = [(0, "II"), (0, "XX"), (0, "YY"), (0, "ZZ")]
    sel = SelectBlock(unitaries=unitaries, num_controls=2)
    sel.build()
    U = _unitary(sel)
    expected = _expected_unitary(unitaries, 2)
    assert np.linalg.norm(U - expected) < 1e-12


def test_two_controls_partial_coverage():
    """Fewer unitaries than ctrl-states: unused ctrl-states act as I."""
    # Only 2 of 4 ctrl-states used.
    unitaries = [(0, "XX"), (0, "ZZ")]
    sel = SelectBlock(unitaries=unitaries, num_controls=2)
    sel.build()
    U = _unitary(sel)
    expected = _expected_unitary(unitaries, 2)
    assert np.linalg.norm(U - expected) < 1e-12


def test_three_controls_pauli_z_strings():
    """Exercises 3-control MCZ lowering across multiple ctrl-states."""
    unitaries = [(0, "Z")] * 8  # all 8 ctrl-states pick Z on the single target qubit
    sel = SelectBlock(unitaries=unitaries, num_controls=3)
    sel.build()
    U = _unitary(sel)
    # When all ctrl-states pick the same U, total = I_ctrl ⊗ U.
    expected_target_op = Z
    expected = _kron_le(*([np.eye(2, dtype=complex)] * 3 + [expected_target_op]))
    # Wait — we need q0 ⊗ ... ⊗ q3.  The target is q3, controls are q0..q2.
    # _kron_le(A_q0, A_q1, A_q2, A_q3) = A_q3 ⊗ A_q2 ⊗ A_q1 ⊗ A_q0.
    # So we want target_op on q3: pass it last.
    expected = _kron_le(I, I, I, expected_target_op)
    assert np.linalg.norm(U - expected) < 1e-12


# ── Unitarity sanity ─────────────────────────────────────────────────────


def test_unitary_property():
    """Random-ish 2-control configuration — output must be unitary."""
    unitaries = [(0, "X"), (0, "Y"), (0, "Z"), (0, "I")]
    sel = SelectBlock(unitaries=unitaries, num_controls=2)
    sel.build()
    U = _unitary(sel)
    Inn = np.eye(2**sel.n_qubits)
    assert np.linalg.norm(U @ U.conj().T - Inn) < 1e-10
    assert np.linalg.norm(U.conj().T @ U - Inn) < 1e-10


# ── General Unitary Blocks ───────────────────────────────────────────────


class HBlock(SimpleBlock):
    def __init__(self):
        super().__init__(1, name="HBlock")

    def build_vanilla(self) -> None:
        self.h(0)


class SBlock(SimpleBlock):
    def __init__(self):
        super().__init__(1, name="SBlock")

    def build_vanilla(self) -> None:
        self.s(0)


class XThenHBlock(SimpleBlock):
    def __init__(self):
        super().__init__(1, name="XThenHBlock")

    def build_vanilla(self) -> None:
        self.x(0)
        self.h(0)


class TwoQubitEntanglerBlock(SimpleBlock):
    def __init__(self):
        super().__init__(2, name="TwoQubitEntanglerBlock")

    def build_vanilla(self) -> None:
        self.h(0)
        self.cx(0, 1)


class TwoQubitLocalBlock(SimpleBlock):
    def __init__(self):
        super().__init__(2, name="TwoQubitLocalBlock")

    def build_vanilla(self) -> None:
        self.x(0)
        self.s(1)


def _pauli_dict_unitary(pauli_dict, target_size=None):
    if target_size is None:
        target_size = max(pauli_dict.keys()) + 1 if pauli_dict else 1
    paulis = ["I"] * target_size
    for qubit, pauli in pauli_dict.items():
        paulis[qubit] = pauli
    return _kron_le(*(PAULI[p] for p in paulis))


def _entry_target_size(entry):
    _, unitary = _split_entry(entry)
    if isinstance(unitary, str):
        return len(unitary)
    if isinstance(unitary, dict):
        return max(unitary.keys()) + 1 if unitary else 1
    return unitary.n_qubits


def _split_entry(entry):
    if isinstance(entry, tuple):
        return entry
    return 0.0, entry


def _entry_unitary(entry, target_size):
    phase, unitary = _split_entry(entry)
    phase_factor = np.exp(1j * float(phase or 0.0))

    if isinstance(unitary, str):
        base = _kron_le(*(PAULI[p] for p in unitary))
    elif isinstance(unitary, dict):
        base = _pauli_dict_unitary(unitary, target_size)
    else:
        base = _unitary(unitary)

    assert base.shape == (2**target_size, 2**target_size)
    return phase_factor * base


def _expected_select_unitary(unitaries, num_controls: int, target_size=None) -> np.ndarray:
    if target_size is None:
        target_size = _entry_target_size(unitaries[0])
    n_qubits = num_controls + target_size
    dim = 2**n_qubits
    target_dim = 2**target_size

    ctrl_states = list(itertools.product([False, True], repeat=num_controls))
    U = np.zeros((dim, dim), complex)

    for c_idx, c_bits in enumerate(ctrl_states):
        c_int = sum(int(bit) << i for i, bit in enumerate(c_bits))
        if c_idx < len(unitaries):
            U_c = _entry_unitary(unitaries[c_idx], target_size)
        else:
            U_c = np.eye(target_dim, dtype=complex)

        for t_in in range(target_dim):
            for t_out in range(target_dim):
                full_in = (t_in << num_controls) | c_int
                full_out = (t_out << num_controls) | c_int
                U[full_out, full_in] = U_c[t_out, t_in]

    return U


def test_one_control_selects_arbitrary_single_qubit_blocks():
    unitaries = [HBlock(), SBlock()]

    select = SelectBlock(unitaries=unitaries, num_controls=1)
    select.build()

    assert np.linalg.norm(_unitary(select) - _expected_select_unitary(unitaries, 1)) < 1e-12


def test_one_control_selects_phased_arbitrary_blocks():
    unitaries = [(np.pi / 5, HBlock()), (-np.pi / 7, SBlock())]

    select = SelectBlock(unitaries=unitaries, num_controls=1)
    select.build()

    assert np.linalg.norm(_unitary(select) - _expected_select_unitary(unitaries, 1)) < 1e-12


def test_select_accepts_mixed_block_and_pauli_entries():
    unitaries = [HBlock(), (np.pi / 3, "Z")]

    select = SelectBlock(unitaries=unitaries, num_controls=1)
    select.build()

    assert np.linalg.norm(_unitary(select) - _expected_select_unitary(unitaries, 1)) < 1e-12


def test_two_controls_select_arbitrary_single_qubit_blocks():
    unitaries = [HBlock(), SBlock(), XThenHBlock(), (np.pi / 11, "Y")]

    select = SelectBlock(unitaries=unitaries, num_controls=2)
    select.build()

    assert np.linalg.norm(_unitary(select) - _expected_select_unitary(unitaries, 2)) < 1e-12


def test_arbitrary_two_qubit_block_with_partial_selector_coverage():
    unitaries = [TwoQubitEntanglerBlock(), TwoQubitLocalBlock()]

    select = SelectBlock(unitaries=unitaries, num_controls=2)
    select.build()

    assert np.linalg.norm(_unitary(select) - _expected_select_unitary(unitaries, 2)) < 1e-12


def test_arbitrary_block_widths_must_match():
    with pytest.raises(ValueError, match="same number of target qubits"):
        SelectBlock(unitaries=[HBlock(), TwoQubitEntanglerBlock()], num_controls=1)


def test_non_unitary_block_entries_are_rejected():
    with pytest.raises(ValueError, match="unitary Blocks"):
        SelectBlock(unitaries=[MeasureBlock(0, 0)], num_controls=1)


def test_dict_entries_pad_with_identity_to_dense_width():
    unitaries = [(0.0, "XX"), (0.0, {0: "Z"})]

    select = SelectBlock(unitaries=unitaries, num_controls=1)
    select.build()

    assert select.n_qubits == 3
    assert np.linalg.norm(_unitary(select) - _expected_select_unitary(unitaries, 1)) < 1e-12


def test_dict_only_entries_size_register_by_highest_qubit():
    unitaries = [(0.0, {0: "Z"}), (0.0, {2: "X"})]

    select = SelectBlock(unitaries=unitaries, num_controls=1)
    select.build()

    assert select.n_qubits == 4
    expected = _expected_select_unitary(unitaries, 1, target_size=3)
    assert np.linalg.norm(_unitary(select) - expected) < 1e-12


def test_dict_entry_beyond_dense_width_raises():
    with pytest.raises(ValueError, match="do not fit"):
        SelectBlock(unitaries=[(0.0, "XX"), (0.0, {5: "Z"})], num_controls=1)
