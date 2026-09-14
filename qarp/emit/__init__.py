"""QARPx SDK emitters — convert a Block to an SDK-native circuit object.

    from qarp.emit import QiskitEmitter
    qc = QiskitEmitter().emit(block.flatten(), block.n_qubits)

All implementations are in C++ (``cpp/libqarpx/src/emit/``).  Every emitter
declares a ``gate_set()`` and ``capabilities()``; ``validate(commands)``
returns the first ``qarpx.Incompatibility`` (or ``None``) without importing
the SDK, and runs automatically at the start of every ``emit()``.

Raising surface: a circuit that cannot cross raises
``qarp.errors.CapabilityError`` (the offending command on its ``command``
attribute); a missing SDK raises ``ImportError`` with a ``pip install``
hint; parse/internal failures stay ``RuntimeError``.

Public depth: flat.  No submodules.
"""

from qarpx import (
    QiskitEmitter,
    PennylaneEmitter,
    PytketEmitter,
    QulacsEmitter,
    QASM2Emitter,
    QASM3Emitter,
    QIREmitter,
)

__all__ = [
    "PennylaneEmitter",
    "PytketEmitter",
    "QASM2Emitter",
    "QASM3Emitter",
    "QIREmitter",
    "QiskitEmitter",
    "QulacsEmitter",
]
