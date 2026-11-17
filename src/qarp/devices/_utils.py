"""Device-side gate-set helpers.

The canonical helpers ``get_full_gate_set_1q`` / ``get_full_gate_set_2q``
return ``qx.GateSet`` objects sourced from the C++ side.
"""

import qarpx as qx


def get_full_gate_set_1q() -> "qx.GateSet":
    """All single-qubit gates, plus the ``Measure``/``Barrier``/``GPhase``
    markers a rebase target must admit so rebasing cannot drop them."""
    return qx.full_gateset_1q()


def get_full_gate_set_2q() -> "qx.GateSet":
    """All two-qubit gates."""
    return qx.full_gateset_2q()
