"""Seeded, stack-agnostic circuit and Hamiltonian specs.

A circuit is `(n_qubits, ops)` where each op is a plain tuple drawn from a
deliberately minimal gate set — every stack under test supports all five
natively, so no stack pays a decomposition tax the others avoid:

    ("h",  q)                ("cx", control, target)
    ("rx"/"ry"/"rz", q, theta)

Rotation convention is qarp's (§2): `r<P>(theta) = exp(-i theta P / 2)`,
radians.  Stacks that disagree flip the sign in their own adapter.

Two-qubit gates the algorithms want (controlled-phase, ZZ rotation,
controlled-rotations) are decomposed here, once, so every stack executes an
identical gate sequence.  Decompositions are exact up to global phase; the
check layer is global-phase insensitive for that reason.
"""

import math

SEED = 20260821
BRICKWORK_LAYERS = 4
TROTTER_STEPS = 5
VQE_LAYERS = 3
QPE_ANCILLAS = 4
TFIM_J = 1.0
TFIM_H = 0.6
HUBBARD_T = 1.0
HUBBARD_U = 4.0

# 1x2, 2x2, 2x3, 2x4 sites -> 4, 8, 12, 16 spin-orbitals (qubits).
HUBBARD_GRID = {4: (1, 2), 8: (2, 2), 12: (2, 3), 16: (2, 4)}


class _Rng:
    """Deterministic LCG — identical angles on every host and numpy version.

    numpy's Generator stream is stable, but pinning it here keeps input
    generation free of any numpy import in the child's build phase.
    """

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def uniform(self, lo: float = 0.0, hi: float = 2 * math.pi) -> float:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return lo + (hi - lo) * ((self.state >> 11) / float(1 << 53))


# --- gate-level helpers -----------------------------------------------------


def _u3(ops: list, q: int, rng: _Rng) -> None:
    ops.append(("rz", q, rng.uniform()))
    ops.append(("ry", q, rng.uniform()))
    ops.append(("rz", q, rng.uniform()))


def _rzz(ops: list, a: int, b: int, theta: float) -> None:
    ops.append(("cx", a, b))
    ops.append(("rz", b, theta))
    ops.append(("cx", a, b))


def _cphase(ops: list, c: int, t: int, theta: float) -> None:
    """Controlled-phase, exact up to the global phase exp(i theta / 4)."""
    ops.append(("rz", c, theta / 2))
    ops.append(("rz", t, theta / 2))
    ops.append(("cx", c, t))
    ops.append(("rz", t, -theta / 2))
    ops.append(("cx", c, t))


def _crz(ops: list, c: int, t: int, theta: float) -> None:
    ops.append(("rz", t, theta / 2))
    ops.append(("cx", c, t))
    ops.append(("rz", t, -theta / 2))
    ops.append(("cx", c, t))


def _crx(ops: list, c: int, t: int, theta: float) -> None:
    ops.append(("h", t))
    _crz(ops, c, t, theta)
    ops.append(("h", t))


def _crzz(ops: list, c: int, a: int, b: int, theta: float) -> None:
    ops.append(("cx", a, b))
    _crz(ops, c, b, theta)
    ops.append(("cx", a, b))


def _swap(ops: list, a: int, b: int) -> None:
    ops.append(("cx", a, b))
    ops.append(("cx", b, a))
    ops.append(("cx", a, b))


# --- circuit families -------------------------------------------------------


def brickwork(n: int) -> tuple[int, list]:
    """Random two-qubit blocks (2 CX + 6 rotations) in an even/odd brick pattern.

    Fixed layer count, so gate count grows as O(n) and total cost as O(n 2^n).
    """
    rng = _Rng(SEED + n)
    ops: list = []
    for layer in range(BRICKWORK_LAYERS):
        for a in range(layer % 2, n - 1, 2):
            b = a + 1
            _u3(ops, a, rng)
            _u3(ops, b, rng)
            ops.append(("cx", a, b))
            ops.append(("ry", a, rng.uniform()))
            ops.append(("ry", b, rng.uniform()))
            ops.append(("cx", a, b))
    return n, ops


def qft(n: int) -> tuple[int, list]:
    """Textbook QFT including the final bit-reversal swaps.

    Gate count is O(n^2), unlike the O(n) families — the point of keeping it.
    """
    ops: list = []
    for j in range(n):
        ops.append(("h", j))
        for k in range(j + 1, n):
            _cphase(ops, k, j, math.pi / (1 << (k - j)))
    for j in range(n // 2):
        _swap(ops, j, n - 1 - j)
    return n, ops


def trotter_step(n: int) -> tuple[int, list]:
    """First-order Trotter steps of the 1D transverse-field Ising chain."""
    dt = 0.1
    ops: list = []
    for _ in range(TROTTER_STEPS):
        for a in range(n - 1):
            _rzz(ops, a, a + 1, 2.0 * TFIM_J * dt)
        for q in range(n):
            ops.append(("rx", q, 2.0 * TFIM_H * dt))
    return n, ops


def vqe_ansatz(n: int) -> tuple[int, list]:
    """Hardware-efficient ansatz: Ry layers separated by a CX ring.

    Identical on every stack by construction — this row measures simulation
    throughput per energy evaluation, not whose UCCSD builder is cleverer.
    """
    rng = _Rng(SEED + 7 * n)
    ops: list = []
    for _ in range(VQE_LAYERS):
        for q in range(n):
            ops.append(("ry", q, rng.uniform()))
        for q in range(n):
            ops.append(("cx", q, (q + 1) % n))
    for q in range(n):
        ops.append(("ry", q, rng.uniform()))
    return n, ops


def qpe_circuit(n: int) -> tuple[int, list]:
    """QPE-shaped circuit: ancillas 0..A-1, TFIM system register above them.

    Controlled-U^(2^k) uses one Trotter step with the time scaled by 2^k rather
    than 2^k repetitions, keeping depth linear in the ancilla count.  Trotter
    error is therefore uncontrolled: this row is a timing fixture whose check
    is exact-vs-SDK agreement, not a phase-accuracy claim.
    """
    n_anc = QPE_ANCILLAS
    n_sys = n - n_anc
    if n_sys < 1:
        raise ValueError(f"qpe_phase needs n > {n_anc}, got {n}")
    rng = _Rng(SEED + 13 * n)
    ops: list = []
    for a in range(n_anc):
        ops.append(("h", a))
    for s in range(n_anc, n):
        ops.append(("ry", s, rng.uniform(0.3, 1.2)))
    dt = 0.2
    for k in range(n_anc):
        scale = float(1 << k) * dt
        for a in range(n_anc, n - 1):
            _crzz(ops, k, a, a + 1, 2.0 * TFIM_J * scale)
        for s in range(n_anc, n):
            _crx(ops, k, s, 2.0 * TFIM_H * scale)
    # Inverse QFT on the ancilla register.
    for j in range(n_anc // 2):
        _swap(ops, j, n_anc - 1 - j)
    for j in reversed(range(n_anc)):
        for k in range(j + 1, n_anc):
            _cphase(ops, k, j, -math.pi / (1 << (k - j)))
        ops.append(("h", j))
    return n, ops


CIRCUITS = {
    "brickwork": brickwork,
    "qft": qft,
    "trotter_step": trotter_step,
    "vqe_energy": vqe_ansatz,
    "qpe_phase": qpe_circuit,
}


# --- Hamiltonians -----------------------------------------------------------


def hubbard_terms(n: int, fraction: float = 1.0) -> list:
    """Jordan-Wigner Fermi-Hubbard as [(((q, 'X'), ...), coeff), ...].

    openfermion builds it (bench-time input generation, never inside a timed
    kernel); term order is sorted so every stack receives an identical list.
    """
    from openfermion import fermi_hubbard, jordan_wigner

    if n not in HUBBARD_GRID:
        raise ValueError(f"no Fermi-Hubbard grid for n={n}; have {sorted(HUBBARD_GRID)}")
    x_dim, y_dim = HUBBARD_GRID[n]
    qubit_op = jordan_wigner(fermi_hubbard(x_dim, y_dim, HUBBARD_T, HUBBARD_U))
    terms = sorted(
        ((tuple(factors), complex(coeff)) for factors, coeff in qubit_op.terms.items()),
        key=lambda item: (len(item[0]), item[0]),
    )
    if fraction < 1.0:
        keep = max(1, int(round(len(terms) * fraction)))
        terms = terms[:keep]
    return terms


def tfim_terms(n_sys: int, offset: int) -> list:
    """TFIM on qubits [offset, offset + n_sys), as Pauli-term specs."""
    terms: list[tuple[tuple[tuple[int, str], ...], complex]] = []
    for a in range(n_sys - 1):
        terms.append((((offset + a, "Z"), (offset + a + 1, "Z")), complex(-TFIM_J)))
    for s in range(n_sys):
        terms.append((((offset + s, "X"),), complex(-TFIM_H)))
    return terms


MOLECULES = {4: "h2", 12: "lih", 14: "h2o"}


def molecular_terms(n: int) -> tuple[list, dict]:
    """Committed molecular Hamiltonian (data/<mol>.json) as Pauli-term specs.

    Generated offline by ``data/generate_molecules.py`` (pyscf + openfermion,
    self-validated against FCI before writing); bench runs only read JSON.
    Returns (terms, metadata) — metadata carries the FCI energy so `verify`
    can anchor the fixture against a published value.
    """
    import json
    import pathlib

    if n not in MOLECULES:
        raise ValueError(f"no molecule for n={n}; have {sorted(MOLECULES)}")
    path = pathlib.Path(__file__).resolve().parent / "data" / f"{MOLECULES[n]}.json"
    payload = json.loads(path.read_text())
    terms = [
        (tuple((int(q), p) for q, p in factors), complex(re, im))
        for factors, re, im in payload["terms"]
    ]
    return terms, payload
