"""PennyLane as an independent oracle for the parameter-shift rules.

The same circuit is built in both libraries (same gate names, same wire
labels — expectation values do not depend on endianness once the labels
agree), and ``qml.gradients.param_shift`` is compared with qarp's
``"parameter-shift"`` at a generic point.  ``CRX`` exercises PennyLane's own
four-term rule against ours.  Nightly ``[integrations]`` job.
"""

import numpy as np
import pytest

qml = pytest.importorskip("pennylane")

import qarpx as qx  # noqa: E402
from qarp.algorithms import StateVector  # noqa: E402
from qarp.blocks import SimpleBlock  # noqa: E402
from qarp.engines import QarpEngine  # noqa: E402

POINT = np.array([0.4, -0.7, 1.1, 0.35])


def _qarp_block():
    b = SimpleBlock(2)
    b.ry(0, qx.Param.symbol("p0"))
    b.rx(1, qx.Param.symbol("p1"))
    b.cx(0, 1)
    b.crx(0, 1, qx.Param.symbol("p2"))
    b.rz(0, qx.Param.symbol("p3"))
    b.ry(1, qx.Param.symbol("p0"))  # shared symbol
    b.build()
    return b


def _pennylane_tape(x):
    ops = [
        qml.RY(x[0], wires=0),
        qml.RX(x[1], wires=1),
        qml.CNOT(wires=[0, 1]),
        qml.CRX(x[2], wires=[0, 1]),
        qml.RZ(x[3], wires=0),
        qml.RY(x[0], wires=1),
    ]
    obs = qml.PauliX(0) + 0.5 * qml.PauliZ(1) + 0.3 * (qml.PauliY(0) @ qml.PauliZ(1))
    return qml.tape.QuantumScript(ops, [qml.expval(obs)], trainable_params=[0, 1, 2, 3, 4])


def _qarp_operator():
    return qx.QubitOperator("X0") + qx.QubitOperator("Z1", 0.5) + qx.QubitOperator("Y0 Z1", 0.3)


def test_parameter_shift_matches_pennylane():
    block = _qarp_block()
    engine = QarpEngine()
    engine.build([StateVector(ket=block, operator=_qarp_operator())])
    params = {f"p{k}": float(v) for k, v in enumerate(POINT)}
    ours = engine.run_gradient(params, method="parameter-shift")[0]

    dev = qml.device("default.qubit", wires=2)
    tape = _pennylane_tape(POINT)
    tapes, fn = qml.gradients.param_shift(tape)
    # One entry per trainable gate angle (CNOT has none); the observable's
    # coefficients are deliberately not trainable.
    per_op = np.asarray(fn(qml.execute(tapes, dev)), dtype=float)
    theirs = np.array([per_op[0] + per_op[4], per_op[1], per_op[2], per_op[3]])
    assert np.allclose(ours, theirs, atol=1e-8)

    # Values agree too, so the two circuits are the same circuit.
    value = float(np.real(engine.run(params)[0]))
    assert value == pytest.approx(float(qml.execute([tape], dev)[0]), abs=1e-10)
