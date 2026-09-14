"""Random-Pauli shadow correctness.

The load-bearing oracle is **exact enumeration** of the inverse channel: averaged
over all settings with the analytic Born weights, the single-snapshot estimator
equals ``⟨O⟩`` — deterministic, and it fails immediately for any normalization or
sign error.  Sampled convergence checks are additional (``slow``), never the
oracle.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qarp.algorithms import PauliKernel, PauliShadow, ShadowDataset
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator

# --- independent numpy oracle (LSB convention, matching the protocol) ----------

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
# Basis-change applied before a Z measurement: X→H, Y→H·S†, Z→I.
_U_AXIS = {0: _H, 1: _H @ _SDG, 2: np.eye(2, dtype=complex)}
_PAULI = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}
_CODE = {"X": 0, "Y": 1, "Z": 2}


def _kron_lsb(mats):
    """Tensor per-qubit matrices with qubit 0 as the least-significant bit."""
    order = list(reversed(mats))  # qubit n-1 highest-order, qubit 0 lowest
    full = order[0]
    for m in order[1:]:
        full = np.kron(full, m)
    return full


def _pauli_matrix(term, n):
    mats = [_PAULI["I"]] * n
    for q, axis in term:
        mats[q] = _PAULI[axis]
    return _kron_lsb(mats)


def _random_state(n, seed):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=2**n) + 1j * rng.normal(size=2**n)
    return v / np.linalg.norm(v)


@pytest.mark.parametrize("n", [1, 2])
def test_pauli_inverse_channel_unbiased(n):
    """E[snapshot] over all settings (analytic Born weights) == Tr(Oρ) to 1e-10."""
    psi = _random_state(n, seed=42 + n)
    kernel = PauliKernel(n)
    # A spanning set of Paulis up to weight 2.
    terms = [((q, ax),) for q in range(n) for ax in "XYZ"]
    if n == 2:
        terms += [((0, a), (1, b)) for a in "XYZ" for b in "XYZ"]

    for term in terms:
        analytic = float(np.real(psi.conj() @ _pauli_matrix(term, n) @ psi))
        expected = 0.0
        n_settings = 0
        for axes in itertools.product((0, 1, 2), repeat=n):
            setting = np.array(axes, dtype=np.int8)
            u = _kron_lsb([_U_AXIS[a] for a in axes])
            probs = np.abs(u @ psi) ** 2
            expected += sum(
                probs[b] * kernel.snapshot_estimate(setting, b, term) for b in range(2**n)
            )
            n_settings += 1
        expected /= n_settings
        assert abs(expected - analytic) < 1e-10, (term, expected, analytic)


def test_enumerated_mixed_sign_multi_term_operator_is_exact():
    """Exact-enumeration ``⟨H⟩`` for a mixed-sign multi-term ``H`` == the analytic value.

    The unbiasedness test above enumerates single Pauli *terms*; this pins the
    estimator's coefficient path (``_decompose`` + sum-inside) on an operator
    carrying a negative coefficient.  The negative coefficient must be
    load-bearing: it sits on ``X0 X1``, whose expectation is 1, not on the
    ``Z0`` term the Bell state annihilates.  The oracle is analytic, not another
    qarp path: ``⟨Z0 Z1⟩ = ⟨X0 X1⟩ = 1`` and ``⟨Z0⟩ = 0``, so
    ``⟨H⟩ = 1 - 0.5 - 0.3*0 + 2 = 2.5`` exactly.
    """
    ket = SimpleBlock(2)
    ket.h(0)
    ket.cx(0, 1)
    ket.build()
    psi = np.asarray(ket.statevector())
    h = (
        QubitOperator("Z0 Z1")
        - 0.5 * QubitOperator("X0 X1")
        - 0.3 * QubitOperator("Z0")
        + 2.0 * QubitOperator("")
    )

    records = []
    for axes in itertools.product((0, 1, 2), repeat=2):
        probs = np.abs(_kron_lsb([_U_AXIS[a] for a in axes]) @ psi) ** 2
        records.append(
            (
                np.array(axes, dtype=np.int8),
                {b: float(probs[b]) for b in range(4) if probs[b] > 1e-15},
            )
        )
    dataset = ShadowDataset(PauliKernel(2), 2, records, shot_exact=True)

    value = dataset.estimator().expval(h, n_batches=1).value
    assert value == pytest.approx(2.5, abs=1e-12)


@pytest.mark.parametrize("axis_char,code", [("X", 0), ("Y", 1), ("Z", 2)])
def test_basis_change_matches_kernel_axis_convention(axis_char, code):
    """The real ``_setting_block`` rotation R obeys R† Z R = P_axis (on the gate).

    This is the one seam where the circuit meets the kernel's axis convention.
    It exercises ``PauliShadow._basis_change_block`` itself — not a numpy copy —
    so the ``Y -> S†H`` sign (``sdg`` vs ``s``) is now pinned: the ``s`` mutation
    makes R† Z R = -Y and fails here, where every sampled ⟨Z⟩/⟨X⟩ test stayed green.
    """
    import qarpx as qx

    shadow = PauliShadow(QubitOperator("Z0"), SimpleBlock(1))
    shadow.n_qubits = 1  # helper reads only n_qubits; skip a full build()
    basis = shadow._basis_change_block(np.array([code], dtype=np.int8))
    r = np.array(qx.QarpSimulator().unitary_matrix(basis.flatten(), 1))
    z = _PAULI["Z"]
    np.testing.assert_allclose(r.conj().T @ z @ r, _PAULI[axis_char], atol=1e-12)


@pytest.mark.parametrize("term,weight", [(((0, "Z"),), 1), (((0, "X"), (1, "Y")), 2)])
def test_single_setting_second_moment_scales_as_three_to_weight(term, weight):
    """E[ô²] over all settings == 3^weight, EXACTLY and state-independently.

    This is the variance blow-up that sets the shadow sample cost (Var ≈ 3^k for a
    weight-k Pauli).  Done as a deterministic enumeration oracle, not a sampled
    "grows ~3^weight" check: ô² = 9^k on the 3^{-k} fraction of settings whose axes
    match, so the uniform-setting average is exactly 3^k for any state.
    """
    n = 2
    psi = _random_state(n, seed=7)
    kernel = PauliKernel(n)
    second = 0.0
    for axes in itertools.product((0, 1, 2), repeat=n):
        setting = np.array(axes, dtype=np.int8)
        probs = np.abs(_kron_lsb([_U_AXIS[a] for a in axes]) @ psi) ** 2
        second += sum(
            probs[b] * kernel.snapshot_estimate(setting, b, term) ** 2 for b in range(2**n)
        )
    second /= 3**n  # settings are uniform
    assert second == pytest.approx(3.0**weight, abs=1e-10)


def test_pauli_kernel_hand_built_values():
    """Single-snapshot values are exactly 3·(±1)^weight on a match, 0 on a mismatch."""
    k = PauliKernel(2)
    z = np.array([2, 2], dtype=np.int8)  # both measured in Z
    assert k.snapshot_estimate(z, 0b00, ((0, "Z"),)) == 3.0  # b0=0 → +3
    assert k.snapshot_estimate(z, 0b01, ((0, "Z"),)) == -3.0  # b0=1 → -3
    assert k.snapshot_estimate(z, 0b00, ((0, "Z"), (1, "Z"))) == 9.0  # 3·3
    assert k.snapshot_estimate(z, 0b01, ((0, "Z"), (1, "Z"))) == -9.0  # 3·(-3)
    # axis mismatch (asked X, measured Z) → 0
    assert k.snapshot_estimate(z, 0b00, ((0, "X"),)) == 0.0


def test_endianness_qubit_indexing():
    """Outcome bit q is (outcome >> q) & 1 — qubit 0 is the LSB (§1)."""
    k = PauliKernel(3)
    zzz = np.array([2, 2, 2], dtype=np.int8)
    outcome = 0b001  # only qubit 0 is 1
    assert k.snapshot_estimate(zzz, outcome, ((0, "Z"),)) == -3.0  # qubit0 flipped
    assert k.snapshot_estimate(zzz, outcome, ((2, "Z"),)) == 3.0  # qubit2 is 0


# --- sampled convergence (additional, never the oracle) ------------------------


@pytest.mark.slow
def test_pauli_convergence_ghz():
    """Sampled estimates approach the analytic values on a Bell state."""
    ket = SimpleBlock(2)
    ket.h(0)
    ket.cx(0, 1)
    shadow = PauliShadow(QubitOperator("Z0 Z1"), ket, n_settings=8000, seed=0)
    eng = QarpEngine(seed=1)
    eng.build([shadow])
    eng.run()
    est = shadow.dataset.estimator()
    assert abs(est.expval("Z0 Z1").value - 1.0) < 0.1
    assert abs(est.expval("X0 X1").value - 1.0) < 0.1
    assert abs(est.expval("Z0").value - 0.0) < 0.1


@pytest.mark.slow
def test_cross_check_against_pauli_averaging():
    """Sampled ⟨H⟩ approaches the exact PauliAveraging value."""
    from qarp.algorithms import PauliAveraging

    ket = SimpleBlock(2)
    ket.h(0)
    ket.cx(0, 1)
    H = QubitOperator("Z0 Z1") + 0.5 * QubitOperator("X0 X1") - 0.3 * QubitOperator("Z0")
    shadow = PauliShadow(H, ket, n_settings=8000, seed=2)
    eng = QarpEngine(seed=3)
    eng.build([shadow])
    sampled = eng.run()[0]

    exact = PauliAveraging(ket=ket, operator=H, n_shots=__import__("qarp").EXACT)
    eng2 = QarpEngine(seed=0)
    eng2.build([exact])
    reference = eng2.run()[0]
    assert abs(sampled - reference) < 0.15
