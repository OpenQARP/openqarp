"""Independent-oracle tests for the piecewise-linear payoff operator.

The oracle is a from-scratch, non-circuit computation of the Woerner–Egger
column 2^{-n/2} Σₓ |x⟩(cos(θ(x)/2)|0⟩ + sin(θ(x)/2)|1⟩), θ piecewise-linear
by hand from the slopes/intercepts — no shared code with the block's
comparator / weighted-sum-rotation circuit or with its `_piecewise_theta`
(§18: a target derived the same way as a wrong circuit would pass a
self-consistency check but not this one).
"""

import numpy as np
import pytest

from qarp import PostSelection
from qarp.blocks import ControlledBlock, PiecewiseLinearStateBlock, SimpleBlock
from qarp.blocks._state_preparation.piecewise_linear_state_block import _apply_comparator


def _brute_force_comparator_unitary(n, constant):
    """The XOR-toggle truth table 'flag ^= (x >= constant)' built directly
    from the definition, no shared code with _apply_comparator."""
    dim = 2 ** (n + 1)
    matrix = np.zeros((dim, dim))
    for x in range(2**n):
        result = 1 if x >= constant else 0
        for flag_in in (0, 1):
            idx_in = x | (flag_in << n)
            idx_out = x | ((flag_in ^ result) << n)
            matrix[idx_out, idx_in] = 1.0
    return matrix


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_comparator_matches_brute_force_truth_table(n):
    for constant in range(2**n):
        block = SimpleBlock(n + 1, name="cmp")
        _apply_comparator(block, list(range(n)), constant, n)
        block.build()
        got = block.unitary_matrix()
        expected = _brute_force_comparator_unitary(n, constant)
        np.testing.assert_allclose(got, expected, atol=1e-9)


def _brute_force_theta(n_domain_qubits, breakpoints, slopes, intercepts):
    domain_size = 2**n_domain_qubits
    bounds = [0, *breakpoints, domain_size]
    theta = np.zeros(domain_size)
    for i in range(len(slopes)):
        for x in range(bounds[i], bounds[i + 1]):
            theta[x] = slopes[i] * x + intercepts[i]
    return theta


def _brute_force_column(n_domain_qubits, breakpoints, slopes, intercepts):
    """Index x + 2^n f: f=0 carries cos(θ/2), f=1 carries sin(θ/2)."""
    theta = _brute_force_theta(n_domain_qubits, breakpoints, slopes, intercepts)
    domain_size = 2**n_domain_qubits
    column = np.zeros(2 * domain_size, dtype=complex)
    for x in range(domain_size):
        column[x] = np.cos(theta[x] / 2)
        column[x + domain_size] = np.sin(theta[x] / 2)
    return column / np.sqrt(domain_size)


_CASES = [
    (3, [], [0.15], [0.2]),
    (3, [3], [0.1, 0.3], [0.0, -0.2]),
    (4, [4, 9], [0.05, -0.1, 0.2], [0.1, 0.5, -0.3]),
    (4, [2, 5, 11], [0.1, -0.05, 0.15, 0.02], [0.0, 0.3, -0.2, 0.1]),
    (2, [1, 2, 3], [0.5, -0.3, 0.4, 0.1], [0.0, 0.2, -0.1, 0.3]),
]


@pytest.mark.parametrize("n_domain_qubits,breakpoints,slopes,intercepts", _CASES)
def test_deterministic_column_matches_brute_force(n_domain_qubits, breakpoints, slopes, intercepts):
    block = PiecewiseLinearStateBlock(n_domain_qubits, breakpoints, slopes, intercepts)
    block.build()
    prepared, probability = block.prepared_statevector()

    expected = _brute_force_column(n_domain_qubits, breakpoints, slopes, intercepts)
    assert probability == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(prepared, expected, atol=1e-10)
    np.testing.assert_allclose(block.target_statevector(), expected, atol=1e-12)


def test_declaration_is_domain_plus_flag_with_comparator_ancillas():
    block = PiecewiseLinearStateBlock(3, [3, 5], [0.1, 0.3, 0.2], [0.0, -0.2, 0.1])
    assert block.n_qubits == 6
    assert block.flag_qubit == 5
    assert block.state_qubits == (0, 1, 2, 5)
    assert block.ancilla_qubits == (3, 4)  # the two comparators
    assert block.ancilla_postselection is None
    assert block.target_statevector().shape == (2**4,)


def test_postselecting_the_flag_recovers_the_sin_profile():
    """The old postselected view: flag=1 (and comparators 0) leaves the
    normalised sin(θ(x)/2) profile on the domain register."""
    n_domain_qubits, breakpoints, slopes, intercepts = 4, [6], [0.2, -0.15], [0.1, 0.4]
    block = PiecewiseLinearStateBlock(n_domain_qubits, breakpoints, slopes, intercepts)
    block.build()

    conditions = {q: 0 for q in block.ancilla_qubits}
    conditions[block.flag_qubit] = 1
    domain_state, probability = PostSelection(conditions).apply_statevector(
        block.statevector(), block.n_qubits
    )

    theta = _brute_force_theta(n_domain_qubits, breakpoints, slopes, intercepts)
    amplitudes = np.sin(theta / 2)
    assert probability == pytest.approx(np.mean(amplitudes**2), abs=1e-10)
    np.testing.assert_allclose(domain_state, amplitudes / np.linalg.norm(amplitudes), atol=1e-10)


def test_zero_profile_puts_all_weight_on_flag_zero():
    block = PiecewiseLinearStateBlock(2, [], [0.0], [0.0])
    block.build()
    prepared, probability = block.prepared_statevector()
    expected = np.zeros(8, dtype=complex)
    expected[:4] = 0.5
    assert probability == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(prepared, expected, atol=1e-10)
    np.testing.assert_allclose(block.target_statevector(), expected, atol=1e-12)


def test_controlled_block_applies_declared_column_iff_control_set():
    """Phase-exactness of the declared column: under one control the target
    register is untouched for |0⟩ and equals target_statevector() for |1⟩."""
    inner = PiecewiseLinearStateBlock(3, [3], [0.1, 0.3], [0.0, -0.2])
    inner.build()
    controlled = ControlledBlock(inner, 1)
    controlled.build()
    n = controlled.n_qubits
    assert n == inner.n_qubits + 1
    control = 0  # inner qubit q sits at q + 1

    off = np.zeros(2**n, dtype=complex)
    off[0] = 1.0
    np.testing.assert_allclose(controlled.statevector(off), off, atol=1e-12)

    on = np.zeros(2**n, dtype=complex)
    on[1 << control] = 1.0
    psi = controlled.statevector(on)
    # Control stays |1⟩, comparators |0⟩; read the (domain, flag) column off the rest.
    conditions = {control: 1, **{q + 1: 0 for q in inner.ancilla_qubits}}
    prepared, probability = PostSelection(conditions).apply_statevector(psi, n)
    assert probability == pytest.approx(1.0, abs=1e-10)
    np.testing.assert_allclose(prepared, inner.target_statevector(), atol=1e-10)


def test_rejects_non_ascending_breakpoints():
    with pytest.raises(ValueError):
        PiecewiseLinearStateBlock(3, [5, 3], [0.1, 0.2, 0.3], [0.0, 0.0, 0.0])


def test_rejects_breakpoint_out_of_domain():
    with pytest.raises(ValueError):
        PiecewiseLinearStateBlock(3, [8], [0.1, 0.2], [0.0, 0.0])


def test_rejects_mismatched_piece_count():
    with pytest.raises(ValueError):
        PiecewiseLinearStateBlock(3, [3], [0.1], [0.0])


@pytest.mark.parametrize("bad", [3.0, 3.5, True])
def test_rejects_non_integer_breakpoint(bad):
    with pytest.raises(ValueError, match="integers"):
        PiecewiseLinearStateBlock(3, [bad], [0.1, 0.2], [0.0, 0.0])


def test_accepts_numpy_integer_breakpoint():
    block = PiecewiseLinearStateBlock(3, [np.int64(3)], [0.1, 0.2], [0.0, 0.0])
    assert block.breakpoints == [3]
