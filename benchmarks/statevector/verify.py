"""Independent oracles for the statevector track's own reference (§18).

Every stack is checked against `_np`, so `_np` itself needs an oracle that
shares none of its code.  These build the circuit's full unitary by kron
products of 2x2 gate literals (LSB embedding per §1 — qubit q is the q-th
factor from the right) and, for the physics families, by matrix exponentials
of the Hamiltonian.  A convention error in `_np` — wrong rotation sign, wrong
CX orientation, a bad decomposition in `inputs` — fails here.

Sizes are tiny by construction: these are correctness oracles, not timings.

Usage:  python -m benchmarks.statevector.verify
"""

import cmath
import math

import numpy as np

from benchmarks.statevector import _np as reference
from benchmarks.statevector import inputs

TOL = 1e-10
_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_PAULI = {"X": _X, "Y": _Y, "Z": _Z}


def _embed(n: int, q: int, gate: np.ndarray) -> np.ndarray:
    """kron-embed a 1-qubit gate; qubit 0 is the rightmost factor (LSB)."""
    full = np.eye(1, dtype=complex)
    for k in reversed(range(n)):
        full = np.kron(full, gate if k == q else _I)
    return full


def _embed_cx(n: int, control: int, target: int) -> np.ndarray:
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=complex)
    for index in range(dim):
        flipped = index ^ (1 << target) if (index >> control) & 1 else index
        out[flipped, index] = 1.0
    return out


def _embed_swap(n: int, a: int, b: int) -> np.ndarray:
    """Permutation matrix exchanging two bits; used by the compilation track."""
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=complex)
    for index in range(dim):
        bit_a, bit_b = (index >> a) & 1, (index >> b) & 1
        swapped = index
        if bit_a != bit_b:
            swapped ^= (1 << a) | (1 << b)
        out[swapped, index] = 1.0
    return out


def _gate(name: str, theta: float) -> np.ndarray:
    half = theta / 2
    if name == "h":
        return np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2)
    if name == "rx":
        return math.cos(half) * _I - 1j * math.sin(half) * _X
    if name == "ry":
        return math.cos(half) * _I - 1j * math.sin(half) * _Y
    if name == "rz":
        return np.array([[cmath.exp(-1j * half), 0], [0, cmath.exp(1j * half)]], dtype=complex)
    raise ValueError(name)


def circuit_unitary(n: int, ops: list) -> np.ndarray:
    """Dense unitary of an op list, built independently of `_np`."""
    total = np.eye(1 << n, dtype=complex)
    for op in ops:
        if op[0] == "cx":
            total = _embed_cx(n, op[1], op[2]) @ total
        elif op[0] == "swap":
            total = _embed_swap(n, op[1], op[2]) @ total
        elif op[0] == "h":
            total = _embed(n, op[1], _gate("h", 0.0)) @ total
        else:
            total = _embed(n, op[1], _gate(op[0], op[2])) @ total
    return total


def pauli_matrix(n: int, factors) -> np.ndarray:
    out = np.eye(1 << n, dtype=complex)
    for qubit, letter in factors:
        out = _embed(n, qubit, _PAULI[letter]) @ out
    return out


def _phase_aligned_diff(a: np.ndarray, b: np.ndarray) -> float:
    """Max |a - b| after removing the global phase between them.

    The decompositions in `inputs` are exact only up to global phase, and the
    check layer is global-phase insensitive for the same reason.
    """
    pivot = int(np.argmax(np.abs(b)))
    if abs(b[pivot]) < 1e-12:
        return float(np.max(np.abs(a - b)))
    phase = (a[pivot] / b[pivot]) / abs(a[pivot] / b[pivot])
    return float(np.max(np.abs(a - phase * b)))


def check_circuit_families() -> list[str]:
    """Every family's reference state equals the kron-built unitary on |0...0>."""
    failures = []
    for family, n in (("brickwork", 6), ("qft", 6), ("trotter_step", 6), ("qpe_phase", 6)):
        n_qubits, ops = inputs.CIRCUITS[family](n)
        psi = reference.run_state(n_qubits, ops)
        expected = circuit_unitary(n_qubits, ops)[:, 0]
        delta = _phase_aligned_diff(psi, expected)
        if delta > TOL:
            failures.append(f"{family}: reference state differs from kron unitary by {delta:.2e}")
    return failures


def check_qft_analytic(n: int = 5) -> list[str]:
    """QFT|0...0> is the uniform superposition, exactly."""
    n_qubits, ops = inputs.qft(n)
    psi = reference.run_state(n_qubits, ops)
    amplitude = 1.0 / math.sqrt(1 << n_qubits)
    delta = float(np.max(np.abs(np.abs(psi) - amplitude)))
    return [] if delta <= TOL else [f"qft: |amplitude| deviates from 2^-n/2 by {delta:.2e}"]


def check_trotter_vs_expm(n: int = 6) -> list[str]:
    """The Trotter circuit equals the ordered product of exact exponentials.

    First-order Trotter is not the exact evolution, but the circuit implements
    each factor exactly — so `expm` of each term, multiplied in circuit order,
    is an exact oracle for the rzz/rx decompositions.
    """
    from scipy.linalg import expm

    dt = 0.1
    n_qubits, ops = inputs.trotter_step(n)
    psi = reference.run_state(n_qubits, ops)

    total = np.eye(1 << n_qubits, dtype=complex)
    for _ in range(inputs.TROTTER_STEPS):
        for a in range(n_qubits - 1):
            zz = pauli_matrix(n_qubits, ((a, "Z"), (a + 1, "Z")))
            total = expm(-1j * inputs.TFIM_J * dt * zz) @ total
        for q in range(n_qubits):
            xx = pauli_matrix(n_qubits, ((q, "X"),))
            total = expm(-1j * inputs.TFIM_H * dt * xx) @ total
    delta = _phase_aligned_diff(psi, total[:, 0])
    return [] if delta <= TOL else [f"trotter_step: differs from expm product by {delta:.2e}"]


def check_vqe_energy(n: int = 4) -> list[str]:
    """Reference energy equals <psi|H|psi> from an independently built matrix."""
    n_qubits, ops = inputs.vqe_ansatz(n)
    terms = inputs.hubbard_terms(n)
    energy = reference.run_energy(n_qubits, reference.build_energy(n_qubits, ops, terms))

    psi = circuit_unitary(n_qubits, ops)[:, 0]
    hamiltonian = np.zeros((1 << n_qubits, 1 << n_qubits), dtype=complex)
    for factors, coeff in terms:
        hamiltonian += complex(coeff) * pauli_matrix(n_qubits, factors)
    expected = complex(np.vdot(psi, hamiltonian @ psi))

    failures = []
    if abs(energy - expected) > 1e-9:
        failures.append(f"vqe_energy: {energy!r} vs kron-built {expected!r}")
    if abs(hamiltonian - hamiltonian.conj().T).max() > TOL:
        failures.append("vqe_energy: Hamiltonian is not Hermitian")
    ground = float(np.linalg.eigvalsh(hamiltonian)[0])
    if expected.real < ground - 1e-9:
        failures.append(f"vqe_energy: {expected.real:.9f} below exact ground state {ground:.9f}")
    return failures


def check_molecular_fixture(n: int = 4) -> list[str]:
    """The committed H2 fixture reproduces the published FCI energy.

    Anchors the whole vqe_molecular family against a value that exists outside
    this repository: H2/STO-3G at 0.735 A has FCI energy -1.1373 Ha in the
    literature.  A generator bug (wrong integral convention, wrong mapping)
    lands parts of a Hartree away, not within a millihartree.  All three
    fixtures are additionally checked for real coefficients — a Pauli string
    with a real coefficient is Hermitian term by term.
    """
    failures = []
    for size in (4, 12, 14):
        terms, payload = inputs.molecular_terms(size)
        worst = max(abs(coeff.imag) for _, coeff in terms)
        if worst > 1e-9:
            failures.append(f"molecular n={size}: complex coefficient (imag {worst:.2e})")
        if payload["n_terms"] != len(terms):
            failures.append(f"molecular n={size}: JSON n_terms disagrees with term list")

    terms, payload = inputs.molecular_terms(n)
    hamiltonian = np.zeros((1 << n, 1 << n), dtype=complex)
    for factors, coeff in terms:
        hamiltonian += complex(coeff) * pauli_matrix(n, factors)
    ground = float(np.linalg.eigvalsh(hamiltonian)[0])
    published = -1.1373  # H2/STO-3G, 0.735 A — literature FCI value
    if abs(ground - published) > 1e-3:
        failures.append(f"H2 fixture ground state {ground:.6f} Ha vs published {published} Ha")
    if abs(ground - payload["fci_energy"]) > 1e-6:
        failures.append(
            f"H2 fixture ground state {ground:.6f} Ha vs generator's {payload['fci_energy']:.6f}"
        )
    return failures


CHECKS = (
    check_molecular_fixture,
    check_circuit_families,
    check_qft_analytic,
    check_trotter_vs_expm,
    check_vqe_energy,
)


def run_all() -> list[str]:
    failures = []
    for check in CHECKS:
        found = check()
        print(f"  {'FAIL' if found else 'ok  '} statevector/verify/{check.__name__}")
        failures += found
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(1 if run_all() else 0)
