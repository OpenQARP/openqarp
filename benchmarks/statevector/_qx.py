"""qarp adapter: SimpleBlock + csim statevector, StateVector primitive for energies.

Gate builder signatures already match the op tuples (`b.rz(qubit, theta)`,
`b.cx(control, target)`) and qarp is LSB with the `exp(-i theta P / 2)`
convention (§1, §2), so nothing is translated here.
"""

import time

import numpy as np

from benchmarks.common import KERNEL_PROBE_QUBITS

MSB = False
LABEL = "qarpx"


def build_circuit(n: int, ops: list):
    from qarp.blocks import SimpleBlock

    block = SimpleBlock(n)
    for op in ops:
        if op[0] == "h":
            block.h(op[1])
        elif op[0] == "cx":
            block.cx(op[1], op[2])
        else:
            getattr(block, op[0])(op[1], op[2])
    block.build()
    return block


def run_state(n: int, circuit) -> np.ndarray:
    return np.asarray(circuit.statevector())


def build_energy(n: int, ops: list, terms: list):
    from qarp.algorithms import StateVector
    from qarp.engines import QarpEngine
    from qarp.operators import QubitOperator

    operator = QubitOperator()
    for factors, coeff in terms:
        operator += QubitOperator(tuple(factors), coeff)
    primitive = StateVector(operator=operator, ket=build_circuit(n, ops))
    primitive.build()
    engine = QarpEngine()
    engine.build([primitive])
    return engine


def run_energy(n: int, prepared) -> complex:
    return complex(prepared.run()[0])


def kernel_probe() -> float:
    """t(one rz) / t(one h) on QarpSimulator's raw csim dispatch at
    KERNEL_PROBE_QUBITS — the check qulacs exposes, so the layout guard in
    `_child` is symmetric across the two csim-backed stacks (its vendored
    copy has never tripped it: benchmarks.common.KERNEL_RATIO_LIMIT)."""
    import qarpx as qx
    from qarp.blocks import SimpleBlock

    n, reps = KERNEL_PROBE_QUBITS, 40
    sim = qx.QarpSimulator()
    sim.fusion_max_qubits = 0  # named kernels, no dense blocks

    def cost(add) -> float:
        block = SimpleBlock(n)
        for i in range(reps):
            add(block, i % n)
        block.build()
        cmds = block.flatten()
        sim.statevector(cmds, n)
        best = float("inf")
        for _ in range(3):
            t = time.perf_counter()
            sim.statevector(cmds, n)
            best = min(best, time.perf_counter() - t)
        return best

    return cost(lambda b, q: b.rz(q, 0.3)) / cost(lambda b, q: b.h(q))
