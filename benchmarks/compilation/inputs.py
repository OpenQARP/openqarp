"""Fixture circuits, in the canonical op form every stack is fed.

Three families come from the routing matrix that chose OpenQARP's default router
and one comes from **MQT
Bench**, so the headline rows are not measured solely on fixtures we designed.
Generation is untimed input work: the MQT circuits are flattened to the common
basis with qiskit at optimization level 0, which unrolls the algorithm-level
instruction without optimizing anything away.

Every stack receives the identical op list, and every stack is asked to hit the
same target basis {h, rx, ry, rz, cx, swap} — comparing two-qubit counts across
compilers is meaningless otherwise.
"""

import math

SEED = 20260824
BASIS = ("h", "rx", "ry", "rz", "cx", "swap")
MQT_ALGORITHM = "qft"


class _Rng:
    """Deterministic LCG, so fixtures are identical on every host."""

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.state = (6364136223846793005 * self.state + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        return self.state >> 11

    def uniform(self, lo: float = 0.0, hi: float = 2 * math.pi) -> float:
        return lo + (hi - lo) * (self.next() / float(1 << 53))

    def shuffled(self, items: list) -> list:
        out = list(items)
        for i in range(len(out) - 1, 0, -1):
            j = self.next() % (i + 1)
            out[i], out[j] = out[j], out[i]
        return out


def _rzz(ops: list, a: int, b: int, theta: float) -> None:
    ops.append(("cx", a, b))
    ops.append(("rz", b, theta))
    ops.append(("cx", a, b))


def trotter(n: int) -> list:
    """All-pairs ZZ + X layer — every qubit pair interacts, so routing is forced."""
    rng = _Rng(SEED + n)
    ops: list = []
    for a in range(n):
        for b in range(a + 1, n):
            _rzz(ops, a, b, rng.uniform(0.1, 1.0))
    for q in range(n):
        ops.append(("rx", q, 0.3))
    return ops


def hea(n: int) -> list:
    """Hardware-efficient ansatz with a circular entangler (3 layers)."""
    rng = _Rng(SEED + 7 * n)
    ops: list = []
    for _ in range(3):
        for q in range(n):
            ops.append(("ry", q, rng.uniform()))
        for q in range(n):
            ops.append(("cx", q, (q + 1) % n))
    for q in range(n):
        ops.append(("ry", q, rng.uniform()))
    return ops


def qv(n: int) -> list:
    """Quantum-volume-like: random disjoint pairings, 4 layers."""
    rng = _Rng(SEED + 13 * n)
    ops: list = []
    for _ in range(4):
        order = rng.shuffled(list(range(n)))
        for k in range(0, n - 1, 2):
            a, b = order[k], order[k + 1]
            ops.append(("ry", a, rng.uniform()))
            ops.append(("ry", b, rng.uniform()))
            ops.append(("cx", a, b))
            ops.append(("rz", b, rng.uniform()))
            ops.append(("cx", a, b))
    return ops


def mqt(n: int) -> list:
    """An MQT Bench algorithm circuit, flattened to the common basis.

    Third-party fixture: the point is that the headline numbers are not all
    measured on circuits we wrote.  Measurements and barriers are dropped —
    this track compares compiled gate structure, not readout.
    """
    from mqt.bench import get_benchmark_alg
    from qiskit import transpile

    circuit = get_benchmark_alg(MQT_ALGORITHM, n).remove_final_measurements(inplace=False)
    flat = transpile(
        circuit, basis_gates=["h", "rx", "ry", "rz", "cx"], optimization_level=0, seed_transpiler=1
    )
    return from_qiskit(flat)


def from_qiskit(circuit) -> list:
    """qiskit QuantumCircuit -> canonical op list (basis gates only)."""
    index = {bit: i for i, bit in enumerate(circuit.qubits)}
    ops: list = []
    for instruction in circuit.data:
        name = instruction.operation.name
        if name in ("barrier", "measure", "delay"):
            continue
        qubits = [index[q] for q in instruction.qubits]
        params = [float(p) for p in instruction.operation.params]
        if name in ("cx", "swap"):
            ops.append((name, qubits[0], qubits[1]))
        elif name == "h":
            ops.append(("h", qubits[0]))
        elif name in ("rx", "ry", "rz"):
            ops.append((name, qubits[0], params[0]))
        else:
            raise ValueError(f"gate outside the target basis: {name}")
    return ops


def to_qiskit(n: int, ops: list):
    """Canonical op list -> qiskit QuantumCircuit (basis gates only)."""
    from qiskit import QuantumCircuit

    circuit = QuantumCircuit(n)
    for op in ops:
        name = op[0]
        if name == "h":
            circuit.h(op[1])
        elif name in ("cx", "swap"):
            getattr(circuit, name)(op[1], op[2])
        elif name in ("rx", "ry", "rz"):
            getattr(circuit, name)(op[2], op[1])
        else:
            raise ValueError(f"gate outside the target basis: {name}")
    return circuit


FIXTURES = {"trotter": trotter, "hea": hea, "qv": qv, "mqt": mqt}
