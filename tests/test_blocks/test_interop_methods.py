"""Convenience interop methods on blocks: ``to_<sdk>()`` / ``SimpleBlock.from_<sdk>()``.

Thin block-method wrappers over ``qarp.emit`` / ``qarp.absorb`` (the uniform
export/import surface). The QASM pair needs no SDK and is always exercised; the
four SDK pairs skip when their SDK is absent, matching ``tests/test_emit`` and
``tests/test_absorb``.
"""

import pytest

from qarp.blocks import SimpleBlock


def _benign_block() -> SimpleBlock:
    """A gate set every SDK maps natively — isolates the wrapper wiring from
    the per-SDK gate-mapping tests in test_emit/ and test_absorb/."""
    b = SimpleBlock(2, name="interop")
    b.h(0)
    b.cx(0, 1)
    b.rz(1, 0.7)
    b.build()
    return b


def test_to_qasm_from_qasm_round_trip():
    b = _benign_block()
    assert SimpleBlock.from_qasm3(b.to_qasm3()) == b


@pytest.mark.parametrize(
    "module, to_name, from_name",
    [
        ("qiskit", "to_qiskit", "from_qiskit"),
        ("qulacs", "to_qulacs", "from_qulacs"),
        ("pytket", "to_pytket", "from_pytket"),
        ("pennylane", "to_pennylane", "from_pennylane"),
    ],
)
def test_sdk_to_from_round_trip(module, to_name, from_name):
    """block.to_<sdk>() then SimpleBlock.from_<sdk>() reconstructs the same circuit —
    exercises that the wrappers delegate to the right emitter/absorber with
    ``(flatten(), n_qubits)``."""
    pytest.importorskip(module)
    b = _benign_block()
    circuit = getattr(b, to_name)()
    absorbed = getattr(SimpleBlock, from_name)(circuit)
    assert absorbed == b


def test_to_sdk_requires_built():
    """SDK export mirrors to_qasm3's built-block precondition (checked before
    the SDK is even imported, so this runs without any SDK installed)."""
    b = SimpleBlock(1)
    b.h(0)  # deliberately not built
    with pytest.raises(RuntimeError, match="not built"):
        b.to_qiskit()


def test_from_qasm_returns_python_simpleblock():
    b = _benign_block()
    assert isinstance(SimpleBlock.from_qasm3(b.to_qasm3()), SimpleBlock)
