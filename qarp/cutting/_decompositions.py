"""QPD gate decompositions.

Each function takes a prefix command list and returns 6 experiment command
lists plus a coefficient array.  Gate angles are in radians (qarpx convention).

Mid-circuit measurements use _measure_cmd(), which creates a properly formed
qx.Command with cbits set — routing through the qarpx trajectory simulation
path (the same infrastructure used by ConditionalBlock).
"""

import numpy as np
import sympy

import qarpx as qx

from ._internals import _cmd, _measure_cmd


def decompose_cx(
    prefix: list,
    n_qubits: int,
    idx: list,
    idx_to_be_used_bit: np.ndarray,
) -> tuple[list, np.ndarray]:
    """QPD decomposition of a CX gate into 6 single-qubit experiments.

    Args:
        prefix: Command list accumulated before this cut.
        n_qubits: Qubit count of the circuit.
        idx: [ctrl_qubit, tgt_qubit].
        idx_to_be_used_bit: Array of 6 ints; idx_to_be_used_bit[i] is the next
                            available QPD cbit index for experiment i.

    Returns:
        (experiments, coefficients) where experiments is a list of 6 command lists.
    """
    cbit = int(idx_to_be_used_bit[0])  # all 6 start at the same cbit

    cmds_1 = list(prefix) + [
        _cmd(qx.GateType.Sdg, idx[0]),
        _cmd(qx.GateType.H, idx[1]),
        _cmd(qx.GateType.Sdg, idx[1]),
        _cmd(qx.GateType.H, idx[1]),
    ]
    cmds_2 = list(prefix) + [
        _cmd(qx.GateType.S, idx[0]),
        _cmd(qx.GateType.H, idx[1]),
        _cmd(qx.GateType.S, idx[1]),
        _cmd(qx.GateType.H, idx[1]),
    ]
    cmds_3 = list(prefix) + [
        _cmd(qx.GateType.Sdg, idx[0]),
        _measure_cmd(idx[0], cbit),
    ]
    cmds_4 = list(prefix) + [
        _cmd(qx.GateType.Sdg, idx[0]),
        _measure_cmd(idx[0], cbit),
        _cmd(qx.GateType.H, idx[1]),
        _cmd(qx.GateType.Z, idx[1]),
        _cmd(qx.GateType.H, idx[1]),
    ]
    cmds_5 = list(prefix) + [
        _cmd(qx.GateType.H, idx[1]),
        _cmd(qx.GateType.Sdg, idx[1]),
        _measure_cmd(idx[1], cbit),
        _cmd(qx.GateType.H, idx[1]),
    ]
    cmds_6 = list(prefix) + [
        _cmd(qx.GateType.Z, idx[0]),
        _cmd(qx.GateType.H, idx[1]),
        _cmd(qx.GateType.Sdg, idx[1]),
        _measure_cmd(idx[1], cbit),
        _cmd(qx.GateType.H, idx[1]),
    ]

    subexperiments = [cmds_1, cmds_2, cmds_3, cmds_4, cmds_5, cmds_6]
    coefficients = np.array([0.5, 0.5, 0.5, -0.5, 0.5, -0.5])
    return subexperiments, coefficients


def decompose_rzz(
    prefix: list,
    n_qubits: int,
    idx: list,
    idx_to_be_used_bit: np.ndarray,
    parameter,
) -> tuple[list, np.ndarray]:
    """QPD decomposition of an RZZ gate.

    Args:
        prefix: Command list accumulated before this cut.
        n_qubits: Qubit count.
        idx: [q0, q1].
        idx_to_be_used_bit: Next available QPD cbit per experiment.
        parameter: RZZ rotation angle in radians (qarpx convention).
                   May be a float or sympy.Symbol for parametric circuits.

    Returns:
        (experiments, coefficients)
    """
    # Sign convention: qarpx RZZ(θ) = exp(-i θ/2 Z⊗Z).
    # The QPD formula uses the negated angle; coefficients are
    # cos²(θ/2), sin²(θ/2), ±sin(θ)/2 where θ = |parameter|.
    parameter_rad = -parameter
    cbit = int(idx_to_be_used_bit[0])

    a_1 = sympy.cos(parameter_rad / 2.0) ** 2
    a_2 = sympy.sin(parameter_rad / 2.0) ** 2
    a_3 = (-1) * sympy.sin(parameter_rad) / 2.0
    a_4 = sympy.sin(parameter_rad) / 2.0
    a_5 = a_3
    a_6 = a_4

    cmds_1 = list(prefix)
    cmds_2 = list(prefix) + [
        _cmd(qx.GateType.Z, idx[0]),
        _cmd(qx.GateType.Z, idx[1]),
    ]
    cmds_3 = list(prefix) + [
        _measure_cmd(idx[0], cbit),
        _cmd(qx.GateType.S, idx[1]),
    ]
    cmds_4 = list(prefix) + [
        _measure_cmd(idx[0], cbit),
        _cmd(qx.GateType.Sdg, idx[1]),
    ]
    cmds_5 = list(prefix) + [
        _cmd(qx.GateType.S, idx[0]),
        _measure_cmd(idx[1], cbit),
    ]
    cmds_6 = list(prefix) + [
        _cmd(qx.GateType.Sdg, idx[0]),
        _measure_cmd(idx[1], cbit),
    ]

    subexperiments = [cmds_1, cmds_2, cmds_3, cmds_4, cmds_5, cmds_6]
    coefficients = np.array([a_1, a_2, a_3, a_4, a_5, a_6])
    return subexperiments, coefficients


def decompose_cy(prefix, n_qubits, idx, idx_to_be_used_bit) -> tuple[list, np.ndarray]:
    raise NotImplementedError("CY gates are not yet supported with circuit cutting.")


def decompose_cz(prefix, n_qubits, idx, idx_to_be_used_bit) -> tuple[list, np.ndarray]:
    raise NotImplementedError("CZ gates are not yet supported with circuit cutting.")


def decompose_crx(prefix, n_qubits, idx, idx_to_be_used_bit) -> tuple[list, np.ndarray]:
    raise NotImplementedError("CRX gates are not yet supported with circuit cutting.")


def decompose_cry(prefix, n_qubits, idx, idx_to_be_used_bit) -> tuple[list, np.ndarray]:
    raise NotImplementedError("CRY gates are not yet supported with circuit cutting.")


def decompose_crz(prefix, n_qubits, idx, idx_to_be_used_bit) -> tuple[list, np.ndarray]:
    raise NotImplementedError("CRZ gates are not yet supported with circuit cutting.")
