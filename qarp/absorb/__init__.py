"""QARPx SDK absorbers — convert an SDK-native circuit to a SimpleBlock.

    from qarp.absorb import QiskitAbsorber
    block = QiskitAbsorber().absorb(qiskit_circuit)

C++ parsers (``cpp/libqarpx/src/absorb/``) extract (commands, n_qubits, n_cbits);
the Python wrappers below construct the proper ``qarp.blocks.SimpleBlock``.
The QASM absorbers are thin Python wrappers over their C++ parsers.

Raising surface: an object from a *different* SDK raises
``qarp.errors.CapabilityError`` naming the module it came from (checked
before any attribute access); a missing SDK raises ``ImportError`` with a
``pip install`` hint; unparseable input stays ``RuntimeError``.

Public depth: flat.  Submodules are private.
"""

from ..errors import CapabilityError as _CapabilityError
from ._reconstruct import build_block as _build_block
from ._qasm3_absorber import QASM3Absorber
from ._qasm2_absorber import QASM2Absorber
import qarpx as _qx


def _require_sdk_object(obj, expected_roots: tuple, absorber: str):
    # type(obj).__module__ identifies the owning SDK without touching any
    # attribute of the (untrusted-shape) object itself.  Compiled SDKs may
    # expose their types from a binding module (qulacs → qulacs_core), hence
    # a tuple of accepted roots.
    root = type(obj).__module__.split(".")[0]
    if root not in expected_roots:
        raise _CapabilityError(
            f"{absorber} expected a {expected_roots[0]} object but received "
            f"an instance of {type(obj).__module__}.{type(obj).__qualname__} "
            f"(module root '{root}')"
        )


class QiskitAbsorber:
    def source_name(self) -> str:
        return "qiskit"

    def absorb(self, circuit):
        _require_sdk_object(circuit, ("qiskit",), "QiskitAbsorber")
        commands, n_qubits, n_cbits = _qx.QiskitAbsorber().absorb(circuit)
        return _build_block(commands, n_qubits, n_cbits)


class QulacsAbsorber:
    def source_name(self) -> str:
        return "qulacs"

    def absorb(self, circuit):
        _require_sdk_object(circuit, ("qulacs", "qulacs_core"), "QulacsAbsorber")
        commands, n_qubits, n_cbits = _qx.QulacsAbsorber().absorb(circuit)
        return _build_block(commands, n_qubits, n_cbits)


class PytketAbsorber:
    def source_name(self) -> str:
        return "pytket"

    def absorb(self, circuit):
        _require_sdk_object(circuit, ("pytket",), "PytketAbsorber")
        commands, n_qubits, n_cbits = _qx.PytketAbsorber().absorb(circuit)
        return _build_block(commands, n_qubits, n_cbits)


class PennylaneAbsorber:
    def source_name(self) -> str:
        return "pennylane"

    def absorb(self, tape):
        _require_sdk_object(tape, ("pennylane",), "PennylaneAbsorber")
        commands, n_qubits, n_cbits = _qx.PennylaneAbsorber().absorb(tape)
        return _build_block(commands, n_qubits, n_cbits)


__all__ = [
    "PennylaneAbsorber",
    "PytketAbsorber",
    "QASM2Absorber",
    "QASM3Absorber",
    "QiskitAbsorber",
    "QulacsAbsorber",
]
