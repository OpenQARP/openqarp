"""Internal helpers for the cutting module. Not part of the public API."""

from dataclasses import dataclass

import qarpx as qx


@dataclass
class _CutMarker:
    """Sentinel that replaces a 2q gate in a command stream to mark a cut location.

    Placed by Reconstructer.reconstruct_swap_gates_by_customs.
    Identified downstream via isinstance(cmd, _CutMarker).
    Never exposed outside qarp.cutting.
    """

    name: str  # "cut_1", "cut_2", …
    qubits: list  # [q0, q1] global qubit indices
    gate_type_str: str  # "CX", "RZZ", … — key into decomposition parser dict
    params: list  # gate params as qx.Param objects (radians, qarpx convention)


def _cmd(gate: qx.GateType, q0: int, q1_or_param=None, param=None) -> qx.Command:
    """Create a qx.Command for any 1q or 2q gate, with or without a parameter.

    Handles all four combinations that appear in QPD decompositions:

    +------------------+----------------+---------------------------+
    | Gate kind        | Call signature | Example                   |
    +==================+================+===========================+
    | 1q, no param     | (gate, q0)     | _cmd(H, 0)                |
    | 1q, parametric   | (gate, q0, p)  | _cmd(Rx, 0, π/2)          |
    | 2q, no param     | (gate, q0, q1) | _cmd(CX, 0, 1)            |
    | 2q, parametric   | (gate,q0,q1,p) | _cmd(RZZ, 0, 1, θ)        |
    +------------------+----------------+---------------------------+

    The two-qubit parametric case (e.g. ``RZZ``) requires the explicit
    ``param`` keyword because ``q1_or_param`` already carries the second
    qubit index and nanobind dispatches ``Command(gate, int, int)`` as a
    2q no-param gate, not as a 1q parametric gate.

    Args:
        gate: Gate type (``qx.GateType`` enum value).
        q0: First (and for 1q gates, only) qubit index.
        q1_or_param: For 2q gates — the second qubit index (int).
                     For 1q parametric gates — the angle (float or
                     ``qx.Param``).  ``None`` → 1q no-param gate.
        param: Gate angle for 2q parametric gates (float or ``qx.Param``).
               Only meaningful when ``q1_or_param`` is an int (second qubit).

    Note:
        2q parametric gates (e.g. ``RZZ``) must pass the angle as ``param``;
        ``Reconstructer.reconstruct_filter_qubits`` relies on the 4-argument
        form to emit filtered subcircuit gates.
    """
    if q1_or_param is None:
        return qx.Command(gate, q0)
    if param is not None:
        # 2q parametric gate (e.g. RZZ): Command(gate, q0, q1, param)
        return qx.Command(gate, q0, q1_or_param, param)
    return qx.Command(gate, q0, q1_or_param)


def _measure_cmd(qubit: int, cbit: int) -> qx.Command:
    """Create a mid-circuit Measure command with proper cbits field.

    The direct Command constructor treats the second int as a qubit, not a
    cbit, so we use the SimpleBlock.measure() builder to get the correctly
    formed Command.
    """
    b = qx.SimpleBlock(qubit + 1, "_m")
    b.measure(qubit, cbit)
    b.set_built(True)
    return b.flatten()[0]


def _param_to_value(p):
    """Convert a qx.Param to a float (if concrete) or sympy.Symbol (if symbolic)."""
    import sympy

    if p.is_concrete():
        return p.evaluate({})
    syms = p.free_symbols()
    if len(syms) == 1:
        return sympy.Symbol(syms[0])
    # Multi-symbol linear param: reconstruct as sympy expression
    # For now handle the common single-symbol case; linear(coeff, sym, offset)
    # qx.Param.linear(coeff, name, offset) → coeff*sym + offset
    # We can't easily extract coeff/offset from qx.Param, so return a symbol
    return sympy.Symbol(syms[0])


def _count_qpd_measures(commands: list) -> int:
    """Count QPD Measure commands (qx.Command with gate==Measure) in a command list."""
    return sum(1 for c in commands if isinstance(c, qx.Command) and c.gate == qx.GateType.Measure)
