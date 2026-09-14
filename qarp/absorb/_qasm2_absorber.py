"""QASM2 absorber: OpenQASM 2.0 string → QARPx SimpleBlock.

Wraps the C++ ``qx.QASM2Absorber`` (``cpp/libqarpx/src/absorb/qasm2_absorber.cpp``).

Accepts more than ``QASM2Emitter`` writes (§12.2) — multiple registers, the
whole-register ``measure q -> c;``, and the qiskit-extended ``qelib1.inc``
names — so ``from_qasm2`` works on files qarp did not produce.

Round-trip invariant:
    qasm = qx.QASM2Emitter().emit(cmds, n)
    assert qx.QASM2Emitter().emit(QASM2Absorber().absorb(qasm).flatten(), n) == qasm
"""

from __future__ import annotations

import re

from ._reconstruct import build_block


class QASM2Absorber:
    def source_name(self) -> str:
        return "qasm2"

    def absorb(self, qasm2_source: str):
        import qarpx as qx

        m = re.match(r"^//\s*(\S+)", qasm2_source.lstrip())
        circuit_name = m.group(1) if m else "Circuit"

        try:
            commands, n_qubits, n_cbits, _ = qx.QASM2Absorber().absorb(qasm2_source)
        except RuntimeError as e:
            raise RuntimeError(f"QASM2Absorber: parse error: {e}") from e

        return build_block(commands, n_qubits, n_cbits, name=circuit_name)
