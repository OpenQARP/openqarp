import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import ControlledBlock, PauliBlock
from qarp.operators import QubitOperator
from qarp.operators.functions import hermitian_conjugated

_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _gate_names(block) -> list:
    return [qx.gate_name(c.gate) for c in block.commands()]


def _unitary(block, n_qubits: int) -> np.ndarray:
    """Dense unitary of a built block, LSB (qubit 0 = least significant)."""
    block.build()
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), n_qubits))


def _qubit_for_gate(block, gate_name: str) -> list:
    return [list(c.qubits)[0] for c in block.commands() if qx.gate_name(c.gate) == gate_name]


class TestPauliBlock:
    def test_init_with_string_format(self):
        block = PauliBlock("XYZ")
        assert block.pauli_string == "XYZ"
        assert block.string_format is True
        assert block.n_qubits == 3
        assert block.coefficient == 1
        assert block.phase == 0.0

    def test_init_with_dict_format(self):
        pauli_dict = {0: "X", 1: "Y", 2: "Z"}
        block = PauliBlock(pauli_dict)
        assert block.pauli_string == pauli_dict
        assert block.string_format is False
        assert block.n_qubits == 3
        assert block.coefficient == 1
        assert block.phase == 0.0

    def test_init_with_dict_format_explicit_n_qubits(self):
        pauli_dict = {1: "X", 3: "Y"}
        block = PauliBlock(pauli_dict, n_qubits=5)
        assert block.n_qubits == 5

    def test_init_with_dict_format_auto_n_qubits(self):
        pauli_dict = {1: "X", 3: "Y"}
        block = PauliBlock(pauli_dict)
        assert block.n_qubits == 4  # max(keys) + 1

    def test_init_with_empty_dict(self):
        block = PauliBlock({}, n_qubits=4)
        assert block.n_qubits == 4
        assert block.string_format is False

    def test_init_with_coefficient(self):
        coefficient = 2.5 + 1.5j
        block = PauliBlock("XY", coefficient=coefficient)
        assert block.coefficient == coefficient
        assert abs(block.phase - np.angle(coefficient)) < 1e-10

    def test_init_with_real_coefficient(self):
        block = PauliBlock("XY", coefficient=2.5)
        assert block.coefficient == 2.5
        assert block.phase == 0.0

    def test_init_with_phase(self):
        phase = np.pi / 4
        block = PauliBlock("XY", phase=phase)
        assert block.coefficient is None
        assert block.phase == phase

    def test_init_with_both_coefficient_and_phase_raises_error(self):
        with pytest.raises(ValueError, match="Cannot specify both coefficient and phase"):
            PauliBlock("XY", coefficient=1.0, phase=0.5)

    def test_init_with_invalid_pauli_string_type_raises_error(self):
        with pytest.raises(
            ValueError, match="pauli_string must be either a dictionary or a string"
        ):
            PauliBlock(123)

    def test_init_with_mismatched_n_qubits_raises_error(self):
        with pytest.raises(ValueError, match="n_qubits \\(5\\) must match string length \\(3\\)"):
            PauliBlock("XYZ", n_qubits=5)

    def test_build_string_x_gates(self):
        block = PauliBlock("XXX")
        block.build()
        assert block.n_qubits == 3
        names = _gate_names(block)
        assert names.count("X") == 3

    def test_build_string_mixed_gates(self):
        block = PauliBlock("XYZ")
        block.build()
        names = _gate_names(block)
        assert "X" in names
        assert "Y" in names
        assert "Z" in names

    def test_build_dict_format(self):
        block = PauliBlock({0: "X", 2: "Y", 4: "Z"}, n_qubits=5)
        block.build()
        assert block.n_qubits == 5
        cmds = list(block.commands())
        qubit_gate_map = {list(c.qubits)[0]: qx.gate_name(c.gate) for c in cmds}
        assert qubit_gate_map.get(0) == "X"
        assert qubit_gate_map.get(2) == "Y"
        assert qubit_gate_map.get(4) == "Z"

    def test_phase_encoded_as_gphase(self):
        """Non-zero phase should produce a GPhase gate."""
        phase = np.pi / 3
        block = PauliBlock("X", phase=phase)
        block.build()
        assert abs(block.phase - phase) < 1e-10
        names = _gate_names(block)
        assert "GPhase" in names
        gphase_cmd = next(c for c in block.commands() if qx.gate_name(c.gate) == "GPhase")
        assert abs(list(gphase_cmd.params)[0].value() - phase) < 1e-10

    def test_zero_phase_no_gphase(self):
        """Zero phase (default) should not emit a GPhase gate."""
        block = PauliBlock("X")
        block.build()
        assert "GPhase" not in _gate_names(block)

    def test_coefficient_phase_encoded(self):
        coefficient = 1.0 + 1.0j
        block = PauliBlock("X", coefficient=coefficient)
        block.build()
        assert abs(block.phase - np.angle(coefficient)) < 1e-10
        assert "GPhase" in _gate_names(block)

    # ── coefficient → phase: the angle arg(c), guarded and folded ─────────────
    # Oracle: analytic np.angle.  These close the gaps that let the negative-real
    # sign-drop (bug 1) through — there was no negative-real or -0.0-imag case.

    @pytest.mark.parametrize(
        "coefficient, expected_phase",
        [
            (1, 0.0),
            (-1, np.pi),  # negative real: angle pi, NOT dropped (bug 1)
            (1j, np.pi / 2),
            (-1j, -np.pi / 2),  # lower half-plane: kept as -pi/2, not folded
            (0.5 + 0.5j, np.pi / 4),
            (-0.3, np.pi),  # negative real again, sub-unit magnitude
        ],
    )
    def test_coefficient_phase_equals_angle(self, coefficient, expected_phase):
        block = PauliBlock("Z", coefficient=complex(coefficient))
        assert block.phase == pytest.approx(expected_phase, abs=1e-12)
        assert block.coefficient == complex(coefficient)  # full complex value kept

    @pytest.mark.parametrize("coefficient", [complex(-1, -0.0), complex(-1, -1e-15)])
    def test_negative_real_minus_zero_imag_folds_to_plus_pi(self, coefficient):
        """np.angle returns -pi for a negative real with -0.0 imaginary part (as
        hermitian_conjugated delivers), and -pi + dust for a hair of negative
        imaginary.  Both fold to +pi so two blocks for the same operator compare
        equal.  Oracle: analytic; +pi and -pi are the same rotation."""
        block = PauliBlock("Z", coefficient=coefficient)
        assert block.phase == pytest.approx(np.pi, abs=1e-10)
        assert block.phase > 0.0  # +pi, never -pi

    def test_hermitian_conjugated_negative_real_folds_to_plus_pi(self):
        """The fold's real-world trigger, not a hand-written literal: conjugating
        a negative-real term is what produces the -0.0 imaginary part (adapt_vqd
        Hermitianizes its pool generators this way).  Pins the upstream behaviour
        too — if the operators layer ever normalized the sign of zero, the case
        the fold exists for would silently stop being exercised.

        Oracle: analytic.  A block built from the conjugate must store the same
        phase as one built from a plain -1.0; +pi and -pi are the same rotation,
        but they compare unequal at the 1e-9 parameter tolerance."""
        conjugated = hermitian_conjugated(QubitOperator("Z0", -1.0))
        coefficient = complex(next(iter(dict(conjugated.terms).values())))
        assert np.signbit(coefficient.imag)  # -0.0, the trigger
        assert np.angle(coefficient) == pytest.approx(-np.pi)  # numpy's branch cut

        block = PauliBlock("Z", coefficient=coefficient)
        assert block.phase == pytest.approx(np.pi, abs=1e-10)
        assert block.phase == pytest.approx(PauliBlock("Z", coefficient=-1.0).phase, abs=1e-9)

    def test_angle_noise_below_tolerance_emits_no_gphase(self):
        """A coefficient that is real up to float dust must not emit a GPhase."""
        block = PauliBlock("X", coefficient=1.0 + 1e-17j)
        block.build()
        assert block.phase == 0.0
        assert "GPhase" not in _gate_names(block)

    def test_zero_magnitude_coefficient_emits_no_gphase(self):
        """A ±0.0 coefficient has angle pi but is the zero operator, not a phase;
        the magnitude guard suppresses the spurious GPhase(pi)."""
        block = PauliBlock("X", coefficient=complex(-0.0, 0.0))
        block.build()
        assert block.phase == 0.0
        assert "GPhase" not in _gate_names(block)

    @pytest.mark.parametrize("coefficient", [-1e-11, 1e-11j, -1e-30])
    def test_tiny_nonzero_coefficient_keeps_its_angle(self, coefficient):
        """The guard above is exact-zero, not a tolerance band, and the two must
        not be confused: ``arg`` is scale-free, so a coefficient far below any
        plausible magnitude tolerance still has a well-defined direction.  A band
        would silently return ``+P`` here — free standalone, an O(1) error under
        ``ControlledBlock``, which keeps the direction and discards ``|c|``.

        Oracle: analytic ``np.angle`` on the input."""
        block = PauliBlock("Z", coefficient=coefficient)
        block.build()
        expected = np.angle(coefficient)
        if expected <= -np.pi + 1e-10:  # the -pi -> +pi fold applies here too
            expected += 2 * np.pi
        assert block.phase == pytest.approx(expected, abs=1e-10)
        assert "GPhase" in _gate_names(block)

    def test_change_basis_suppresses_coefficient_phase(self):
        """A change_basis circuit is a measurement-basis rotation, not
        e^{i arg c}·P; no coefficient phase belongs on it."""
        block = PauliBlock("X", coefficient=0.5 + 0.5j, change_basis=True)
        block.build()
        assert block.phase == 0.0
        assert "GPhase" not in _gate_names(block)
        assert block.coefficient == 0.5 + 0.5j  # full value still kept

    def test_negative_coefficient_unitary_is_minus_pauli(self):
        """PauliBlock('Z', coefficient=-1) must equal -Z exactly (phase included,
        not modulo global phase).  Oracle: the hand-written -Z matrix."""
        assert np.allclose(_unitary(PauliBlock("Z", coefficient=-1), 1), -_Z, atol=1e-12)

    @pytest.mark.parametrize("coefficient", [-1, 1j, 0.5 + 0.5j, -0.3])
    def test_controlled_coefficient_block_full_matrix(self, coefficient):
        """Under control the coefficient's phase is physically observable (§13).
        Oracle: the full controlled matrix — control-on subspace = (c/|c|)·Z,
        control-off = I, off-diagonal corners = 0 (§6.1).  Control is qubit 0
        (LSB), so the control-on subspace is the odd basis indices."""
        c = complex(coefficient)
        m = _unitary(ControlledBlock(PauliBlock("Z", coefficient=c)), 2)
        on = m[np.ix_([1, 3], [1, 3])]
        off = m[np.ix_([0, 2], [0, 2])]
        assert np.allclose(on, (c / abs(c)) * _Z, atol=1e-12)
        assert np.allclose(off, np.eye(2), atol=1e-12)
        assert np.allclose(m[np.ix_([0, 2], [1, 3])], 0.0, atol=1e-12)
        assert np.allclose(m[np.ix_([1, 3], [0, 2])], 0.0, atol=1e-12)

    def test_build_identity_dict(self):
        block = PauliBlock({}, n_qubits=1)
        block.build()
        assert block.n_qubits == 1
        names = _gate_names(block)
        assert "X" not in names and "Y" not in names and "Z" not in names

    def test_name_parameter(self):
        block = PauliBlock("X", name="TestPauli")
        assert block.name == "TestPauli"
        block.build()
        assert block.name == "TestPauli"

    def test_default_name(self):
        block = PauliBlock("X")
        assert block.name == "Pauli"

    def test_all_pauli_operators(self):
        for pauli_op in ["X", "Y", "Z"]:
            block = PauliBlock(pauli_op)
            block.build()
            assert pauli_op in _gate_names(block)

    def test_identity_operator_no_gate(self):
        block = PauliBlock("I")
        block.build()
        assert _gate_names(block) == []

    def test_multiple_same_operators(self):
        block = PauliBlock("XXXX")
        block.build()
        assert _gate_names(block).count("X") == 4

    def test_large_qubit_indices_dict(self):
        block = PauliBlock({10: "X", 20: "Y", 30: "Z"}, n_qubits=31)
        block.build()
        assert block.n_qubits == 31
        cmds = list(block.commands())
        qubit_indices = [list(c.qubits)[0] for c in cmds]
        assert 10 in qubit_indices
        assert 20 in qubit_indices
        assert 30 in qubit_indices

    def test_change_basis_x(self):
        """X with change_basis=True → single H gate."""
        block = PauliBlock("X", change_basis=True)
        block.build()
        assert _gate_names(block) == ["H"]

    def test_change_basis_y(self):
        """Y with change_basis=True → Sdg then H."""
        block = PauliBlock("Y", change_basis=True)
        block.build()
        assert _gate_names(block) == ["Sdg", "H"]

    def test_change_basis_z(self):
        """Z with change_basis=True → no gates."""
        block = PauliBlock("Z", change_basis=True)
        block.build()
        assert _gate_names(block) == []

    def test_measure_adds_measure_gates(self):
        block = PauliBlock("XX", measure=True)
        block.build()
        names = _gate_names(block)
        assert names.count("Measure") == 2

    def test_measure_with_pauli_gates(self):
        """Pauli gates come before Measure gates."""
        block = PauliBlock("XY", measure=True)
        block.build()
        names = _gate_names(block)
        assert "X" in names
        assert "Y" in names
        assert names.index("Measure") > names.index("X")
