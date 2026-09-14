"""Random circuit generator.

Builds a :class:`qarp.blocks.SimpleBlock` populated with random gates from a
caller-supplied gate set of :class:`qarpx.GateType` values.
"""

import random
from typing import List

from numpy import pi

import qarpx as qx

from ..blocks import SimpleBlock

_GATESET_CTRL_ROTATIONS = {qx.GateType.CRx, qx.GateType.CRy, qx.GateType.CRz}
_GATESET_ROTATIONS = {qx.GateType.Rx, qx.GateType.Ry, qx.GateType.Rz}
_GATESET_TWO_QUBIT_NOPARAM = {qx.GateType.CX, qx.GateType.CZ, qx.GateType.SWAP}
_GATESET_ONE_QUBIT_NOPARAM = {
    qx.GateType.H,
    qx.GateType.X,
    qx.GateType.Y,
    qx.GateType.Z,
    qx.GateType.S,
    qx.GateType.Sdg,
    qx.GateType.T,
    qx.GateType.Tdg,
}


def generate_random_circuit(
    num_gates: int, num_qubits: int, gate_set: List[qx.GateType]
) -> SimpleBlock:
    """Build a random :class:`SimpleBlock` of ``num_gates`` gates.

    Args:
        num_gates: Number of gates to emit.
        num_qubits: Register size.
        gate_set: Allowed :class:`qarpx.GateType` values.  May mix
            single-qubit no-param, single-qubit parametric, two-qubit
            no-param, and controlled-rotation gates.

    Returns:
        A built :class:`SimpleBlock` with ``num_gates`` random gates.

    Raises:
        AssertionError: If any of ``num_gates``, ``num_qubits`` or
            ``len(gate_set)`` is not strictly positive.
        ValueError: If a gate in ``gate_set`` isn't in one of the supported
            categories above.
    """
    assert num_gates > 0, "The number of gates should be greater or equal to 1."
    assert num_qubits > 0, "The number of qubits should be greater or equal to 1."
    assert len(gate_set) > 0, "The gate set should be a list of qx.GateType greater or equal to 1."

    block = SimpleBlock(num_qubits, name="random")
    qubits = list(range(num_qubits))
    rotation_method = {
        qx.GateType.Rx: block.rx,
        qx.GateType.Ry: block.ry,
        qx.GateType.Rz: block.rz,
        qx.GateType.CRx: block.crx,
        qx.GateType.CRy: block.cry,
        qx.GateType.CRz: block.crz,
    }
    one_qubit_method = {
        qx.GateType.H: block.h,
        qx.GateType.X: block.x,
        qx.GateType.Y: block.y,
        qx.GateType.Z: block.z,
        qx.GateType.S: block.s,
        qx.GateType.Sdg: block.sdg,
        qx.GateType.T: block.t,
        qx.GateType.Tdg: block.tdg,
    }
    two_qubit_method = {
        qx.GateType.CX: block.cx,
        qx.GateType.CZ: block.cz,
        qx.GateType.SWAP: block.swap,
    }

    for _ in range(num_gates):
        random.shuffle(qubits)
        gate = random.choice(gate_set)
        if gate in _GATESET_CTRL_ROTATIONS:
            angle = random.uniform(0, 2 * pi)
            rotation_method[gate](qubits[0], qubits[1], angle)
        elif gate in _GATESET_TWO_QUBIT_NOPARAM:
            two_qubit_method[gate](qubits[0], qubits[1])
        elif gate in _GATESET_ONE_QUBIT_NOPARAM:
            one_qubit_method[gate](qubits[0])
        elif gate in _GATESET_ROTATIONS:
            angle = random.uniform(0, 2 * pi)
            rotation_method[gate](qubits[0], angle)
        else:
            raise ValueError(f"Unsupported gate {gate}.")

    block.build()
    return block
