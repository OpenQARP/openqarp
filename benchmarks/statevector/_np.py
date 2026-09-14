"""Plain-numpy statevector reference — the track's oracle stack (§18).

An independent implementation living in this repo: gates are applied by
reshape + einsum, Pauli expectations by direct index manipulation, with no
SDK involved.  Every other stack's fingerprint is checked against this one,
and its timings double as the "unoptimized baseline" column.

LSB convention throughout (§1): qubit q is bit q of the amplitude index.
"""

import cmath
import math

import numpy as np

MSB = False
LABEL = "numpy"


def _gate_matrix(name: str, theta: float) -> np.ndarray:
    half = theta / 2
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    if name == "rx":
        return np.array(
            [[math.cos(half), -1j * math.sin(half)], [-1j * math.sin(half), math.cos(half)]],
            dtype=complex,
        )
    if name == "ry":
        return np.array(
            [[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]], dtype=complex
        )
    if name == "rz":
        return np.array([[cmath.exp(-1j * half), 0], [0, cmath.exp(1j * half)]], dtype=complex)
    raise ValueError(f"unknown gate {name}")


def _apply_1q(psi: np.ndarray, n: int, q: int, mat: np.ndarray) -> np.ndarray:
    view = psi.reshape(1 << (n - q - 1), 2, 1 << q)
    return np.einsum("ij,ajb->aib", mat, view, optimize=True).reshape(-1)


def _two_axis_view(psi: np.ndarray, n: int, a: int, b: int):
    hi, lo = max(a, b), min(a, b)
    return psi.reshape(1 << (n - hi - 1), 2, 1 << (hi - lo - 1), 2, 1 << lo), hi


def _apply_cx(psi: np.ndarray, n: int, c: int, t: int) -> np.ndarray:
    view, hi = _two_axis_view(psi, n, c, t)
    if c == hi:
        keep = view[:, 1, :, 0, :].copy()
        view[:, 1, :, 0, :] = view[:, 1, :, 1, :]
        view[:, 1, :, 1, :] = keep
    else:
        keep = view[:, 0, :, 1, :].copy()
        view[:, 0, :, 1, :] = view[:, 1, :, 1, :]
        view[:, 1, :, 1, :] = keep
    return view.reshape(-1)


def build_circuit(n: int, ops: list):
    """No native object to build — the op list is already the program."""
    return ops


def evolve(n: int, ops, psi: np.ndarray | None = None) -> np.ndarray:
    """Apply an op list to `psi` (default |0...0>).

    `swap` is accepted so the compilation track can simulate routed circuits
    with the same independent reference the statevector track is checked
    against; nothing the statevector track emits uses it.
    """
    if psi is None:
        psi = np.zeros(1 << n, dtype=complex)
        psi[0] = 1.0
    else:
        psi = np.array(psi, dtype=complex).reshape(-1)
    for op in ops:
        if op[0] == "cx":
            psi = _apply_cx(psi, n, op[1], op[2])
        elif op[0] == "swap":
            a, b = op[1], op[2]
            psi = _apply_cx(psi, n, a, b)
            psi = _apply_cx(psi, n, b, a)
            psi = _apply_cx(psi, n, a, b)
        elif op[0] == "h":
            psi = _apply_1q(psi, n, op[1], _gate_matrix("h", 0.0))
        else:
            psi = _apply_1q(psi, n, op[1], _gate_matrix(op[0], op[2]))
    return psi


def run_state(n: int, circuit) -> np.ndarray:
    return evolve(n, circuit)


def _apply_pauli(psi: np.ndarray, n: int, factors) -> np.ndarray:
    out = psi
    for qubit, letter in factors:
        if letter == "Z":
            view = out.reshape(1 << (n - qubit - 1), 2, 1 << qubit).copy()
            view[:, 1, :] *= -1
            out = view.reshape(-1)
        elif letter == "X":
            out = _apply_1q(out, n, qubit, np.array([[0, 1], [1, 0]], dtype=complex))
        elif letter == "Y":
            out = _apply_1q(out, n, qubit, np.array([[0, -1j], [1j, 0]], dtype=complex))
        else:
            raise ValueError(f"unknown Pauli {letter}")
    return out


def build_energy(n: int, ops: list, terms: list):
    return ops, terms


def run_energy(n: int, prepared) -> complex:
    ops, terms = prepared
    psi = run_state(n, ops)
    total = 0j
    for factors, coeff in terms:
        total += coeff * complex(np.vdot(psi, _apply_pauli(psi, n, factors)))
    return total
