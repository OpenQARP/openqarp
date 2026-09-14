"""Wrong-SDK objects are rejected by type, before any attribute access.

The 4×4 cross-product: each SDK's circuit object fed to each of the other
three absorbers raises ``CapabilityError`` naming the module the object
actually came from (previously an arbitrary ``AttributeError`` from deep
inside the parser).  The guard reads only ``type(obj).__module__``, so a
misshapen object cannot crash it.
"""

import pytest

from qarp.errors import CapabilityError

SDKS = ["qiskit", "pytket", "qulacs", "pennylane"]


def _make_circuit(sdk: str):
    if sdk == "qiskit":
        import qiskit

        qc = qiskit.QuantumCircuit(1)
        qc.h(0)
        return qc
    if sdk == "pytket":
        import pytket

        circ = pytket.Circuit(1)
        circ.H(0)
        return circ
    if sdk == "qulacs":
        import qulacs

        qcirc = qulacs.QuantumCircuit(1)
        qcirc.add_H_gate(0)
        return qcirc
    if sdk == "pennylane":
        import pennylane as qml

        return qml.tape.QuantumScript([qml.Hadamard(wires=0)])
    raise AssertionError(sdk)


def _absorber(sdk: str):
    from qarp import absorb

    return {
        "qiskit": absorb.QiskitAbsorber,
        "pytket": absorb.PytketAbsorber,
        "qulacs": absorb.QulacsAbsorber,
        "pennylane": absorb.PennylaneAbsorber,
    }[sdk]()


@pytest.mark.parametrize("source", SDKS)
@pytest.mark.parametrize("absorber", SDKS)
def test_wrong_sdk_object_raises_capability_error(source, absorber):
    pytest.importorskip(source)
    circuit = _make_circuit(source)
    if source == absorber:
        block = _absorber(absorber).absorb(circuit)
        assert block.n_qubits == 1
        return
    with pytest.raises(CapabilityError, match=source):
        _absorber(absorber).absorb(circuit)


def test_non_sdk_object_raises_capability_error():
    with pytest.raises(CapabilityError, match="builtins"):
        _absorber("qiskit").absorb("not a circuit")
