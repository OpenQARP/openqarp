"""QASM3 absorber: QASM3 string → QARPx SimpleBlock.

Wraps the C++ ``qx.QASM3Absorber`` (``cpp/libqarpx/src/absorb/qasm3_absorber.cpp``).

Round-trip invariant:
    qasm = qx.QASM3Emitter().emit(cmds, n)
    assert qx.QASM3Emitter().emit(QASM3Absorber().absorb(qasm).flatten(), n) == qasm
"""

from __future__ import annotations

import re

from ._reconstruct import build_block


class QASM3Absorber:
    def source_name(self) -> str:
        return "qasm3"

    def absorb(self, qasm3_source: str):
        import qarpx as qx

        m = re.match(r"^//\s*(\S+)", qasm3_source.lstrip())
        circuit_name = m.group(1) if m else "Circuit"

        try:
            commands, n_qubits, n_cbits, _ = qx.QASM3Absorber().absorb(qasm3_source)
        except RuntimeError as e:
            raise RuntimeError(f"QASM3Absorber: parse error: {e}") from e

        return build_block(commands, n_qubits, n_cbits, name=circuit_name)
