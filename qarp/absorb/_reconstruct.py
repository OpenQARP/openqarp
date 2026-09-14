"""Shared reconstruction of a built ``SimpleBlock`` from an absorbed
``(commands, n_qubits, n_cbits)`` triple.

Every absorber — the four SDK wrappers in ``__init__`` and the QASM3 wrapper —
funnels through this one function, so the ``set_commands`` / ``set_built`` /
``_finalize`` reconstruction protocol lives in one place, shared by all backends.
"""

from __future__ import annotations

from typing import Any, Optional


def build_block(commands: Any, n_qubits: int, n_cbits: int, name: Optional[str] = None):
    """Wrap an absorbed command stream in a built ``SimpleBlock``.

    ``name=None`` defers to the ``SimpleBlock`` default ("SimpleBlock"); QASM3
    passes the program name parsed from the leading ``// <name>`` comment.
    """
    from qarp.blocks import SimpleBlock

    block = SimpleBlock(n_qubits, name=name)
    if n_cbits > 0:
        block.n_cbits = n_cbits
    block.set_commands(commands)
    block.mark_built()
    block._finalize()  # sets target_qubits and symbols
    return block
