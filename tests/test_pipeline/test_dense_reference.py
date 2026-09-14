"""Phase 5b of the pipeline property suite: the dense numpy cross-check.

One drawn gate-spec list is evaluated twice: through the full pipeline
(SimpleBlock → StateVector / Sampler → QarpEngine EXACT) and by in-test
dense linear algebra built from nothing but numpy and the documented
conventions — LSB indexing (qubit i = bit i of the amplitude index) and
``exp(-iθP/2)`` rotations in radians (§1–§12).  The dense side shares no
code with qarp, so a convention drift anywhere in the pipeline (endianness,
a factor π, a transposed control) breaks the agreement even if every
qarp-internal test stays self-consistent.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import qarp
from qarp.algorithms import Sampler, StateVector
from qarp.blocks import SimpleBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator
from tests.strategies import angles, pauli_term_maps, render_pauli_term, results_allclose

pytestmark = pytest.mark.property

# ── The dense oracle: numpy only ─────────────────────────────────────────

_I2 = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.diag([1, -1]).astype(complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.diag([1, 1j]).astype(complex)
_T = np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex)
_PAULI = {"X": _X, "Y": _Y, "Z": _Z}


def _rot(pauli: np.ndarray, theta: float) -> np.ndarray:
    return np.cos(theta / 2) * _I2 - 1j * np.sin(theta / 2) * pauli


# Two-qubit matrices in the sub-index convention bit 0 = qubits[0],
# bit 1 = qubits[1] (LSB within the gate, matching the global convention).
_CX = np.eye(4, dtype=complex)[:, [0, 3, 2, 1]]  # control=bit0 flips target=bit1
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.eye(4, dtype=complex)[:, [0, 2, 1, 3]]


def _gate_matrix(gate: str, angle) -> np.ndarray:
    return {
        "h": _H,
        "x": _X,
        "y": _Y,
        "z": _Z,
        "s": _S,
        "t": _T,
        "rx": _rot(_X, angle) if angle is not None else None,
        "ry": _rot(_Y, angle) if angle is not None else None,
        "rz": _rot(_Z, angle) if angle is not None else None,
        "p": np.diag([1, np.exp(1j * angle)]).astype(complex) if angle is not None else None,
        "cx": _CX,
        "cz": _CZ,
        "swap": _SWAP,
    }[gate]


def _embed(u: np.ndarray, qubits: tuple, n: int) -> np.ndarray:
    """Lift a k-qubit matrix onto qubits ``qubits`` of an n-qubit register,
    LSB indexing, by direct basis enumeration (n ≤ 3 here)."""
    dim = 2**n
    mask = sum(1 << q for q in qubits)
    full = np.zeros((dim, dim), dtype=complex)
    for col in range(dim):
        in_sub = sum(((col >> q) & 1) << j for j, q in enumerate(qubits))
        base = col & ~mask
        for out_sub in range(2 ** len(qubits)):
            amp = u[out_sub, in_sub]
            if amp == 0:
                continue
            row = base | sum(((out_sub >> j) & 1) << q for j, q in enumerate(qubits))
            full[row, col] += amp
    return full


def _dense_unitary(n: int, ops: tuple) -> np.ndarray:
    u = np.eye(2**n, dtype=complex)
    for gate, qubits, angle in ops:
        u = _embed(_gate_matrix(gate, angle), qubits, n) @ u
    return u


def _dense_pauli(n: int, term_items: tuple) -> np.ndarray:
    h = np.eye(2**n, dtype=complex)
    for q, p in term_items:
        h = _embed(_PAULI[p], (q,), n) @ h
    return h


def test_dense_oracle_sanity():
    """§18 for the oracle itself: hand-checked actions, no qarp involved."""
    cx01 = _embed(_CX, (0, 1), 2)
    assert np.allclose(cx01 @ np.eye(4)[:, 1], np.eye(4)[:, 3])  # |q0=1⟩ → |11⟩
    assert np.allclose(cx01 @ np.eye(4)[:, 2], np.eye(4)[:, 2])  # control unset
    rx_pi = _embed(_rot(_X, np.pi), (0,), 1)
    assert np.allclose(rx_pi @ np.array([1, 0]), np.array([0, -1j]))  # exp(-iπX/2)


# ── The drawn spec, evaluated on both sides ──────────────────────────────

_PARAMETRIC = ("rx", "ry", "rz", "p")
_ONE_QUBIT = ("h", "x", "y", "z", "s", "t") + _PARAMETRIC
_TWO_QUBIT = ("cx", "cz", "swap")


@st.composite
def circuit_specs(draw):
    n = draw(st.integers(min_value=1, max_value=3))
    ops = []
    for _ in range(draw(st.integers(min_value=1, max_value=8))):
        if n > 1 and draw(st.booleans()):
            gate = draw(st.sampled_from(_TWO_QUBIT))
            q0 = draw(st.integers(min_value=0, max_value=n - 1))
            q1 = draw(st.sampled_from([q for q in range(n) if q != q0]))
            ops.append((gate, (q0, q1), None))
        else:
            gate = draw(st.sampled_from(_ONE_QUBIT))
            q = draw(st.integers(min_value=0, max_value=n - 1))
            angle = draw(angles) if gate in _PARAMETRIC else None
            ops.append((gate, (q,), angle))
    term = tuple(sorted(draw(pauli_term_maps(n)).items()))
    return n, tuple(ops), term


def _qarp_block(n: int, ops: tuple) -> SimpleBlock:
    b = SimpleBlock(n, name="dense_ref")
    for gate, qubits, angle in ops:
        args = list(qubits) + ([angle] if angle is not None else [])
        getattr(b, gate)(*args)
    b.build()
    return b


@given(spec=circuit_specs())
def test_pipeline_matches_dense_numpy_reference(spec):
    n, ops, term_items = spec
    psi = _dense_unitary(n, ops)[:, 0]
    expected_ev = complex(np.vdot(psi, _dense_pauli(n, term_items) @ psi))
    expected_born = {
        tuple((b >> i) & 1 for i in range(n)): float(p)
        for b, p in enumerate(np.abs(psi) ** 2)
        if p > 1e-15
    }

    block = _qarp_block(n, ops)
    op = QubitOperator(render_pauli_term(dict(term_items)), 1.0)
    eng = QarpEngine(n_shots=qarp.EXACT)
    eng.build([StateVector(ket=block, operator=op), Sampler(ket=block, n_shots=qarp.EXACT)])
    r_ev, r_dist = eng.run()

    assert abs(complex(r_ev) - expected_ev) < 1e-10, (
        f"EV {r_ev} vs dense {expected_ev} for {ops} / {term_items}"
    )
    assert results_allclose(dict(r_dist), expected_born, atol=1e-9), (
        f"Born distribution mismatch for {ops}"
    )
