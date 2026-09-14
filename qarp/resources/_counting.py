"""Pure counting: commands / CircuitDAG → :class:`ResourceVector`.

Everything here is *counted* from the circuit as it exists — no cost models
(those live in :mod:`qarp.resources`).  Built on the existing C++
primitives ``CircuitDAG.count_ops()`` / ``depth()`` plus a per-command walk,
which is exact for variable-arity gates (``MCZ``, ``Custom``) where the
static gate table only records a minimum arity — the walk reads each
command's own qubit list, so a width-6 ``MCZ`` lands in ``n_3q_plus``
rather than being mis-bucketed by its table minimum of 2.

This module may import only ``qarpx`` and :mod:`qarp.resources`.
"""

from functools import lru_cache
from typing import Sequence

import qarpx as qx

from ._vector import Provenance, ResourceVector, Stage

# Non-physical commands (qx.gate_is_physical is the single classification —
# shared with Block.n_nqb_gates) minus Measure/Reset, which get their own
# counters; the remainder is excluded from n_gates and op_histogram.
_PSEUDO = frozenset(
    qx.gate_name(gt) for gt in qx.GateType.__members__.values() if not qx.gate_is_physical(gt)
) - {"Measure", "Reset"}
_MEASURE = "Measure"
_RESET = "Reset"
_OPAQUE = "Custom"  # unknown unitary: forces t_count to None
_MCZ = "MCZ"
_T_GATES = frozenset({"T", "Tdg"})


@lru_cache(maxsize=1)
def _gate_table() -> dict[str, tuple[int, int]]:
    """gate name → (num_qubits, num_params) for every bound GateType."""
    return {
        qx.gate_name(gt): (qx.gate_num_qubits(gt), qx.gate_num_params(gt))
        for gt in qx.GateType.__members__.values()
    }


def count_resources(
    source: "Sequence[qx.Command] | qx.CircuitDAG",
    *,
    provenance: Provenance,
    n_qubits: int | None = None,
) -> ResourceVector:
    """Count a circuit snapshot into a :class:`ResourceVector`.

    ``n_qubits`` overrides the inferred width (needed for idle qubits: the
    DAG infers max-wire+1, a block or device may be wider).  ``swap_count``
    is populated only when ``provenance.stage is Stage.ROUTED``; ``t_count``
    only when no parametric, opaque, or width >= 3 ``MCZ`` unitary remains.
    Gates inside branch regions count unconditionally (worst-case).
    """
    if isinstance(source, qx.CircuitDAG):
        dag = source
        cmds = dag.to_commands()
    else:
        cmds = list(source)
        dag = qx.CircuitDAG.from_commands(cmds)

    table = _gate_table()
    n_gates = n_1q = n_2q = n_3q_plus = n_meas = n_resets = t_count = 0
    exact_t = True
    for cmd in cmds:
        name = qx.gate_name(cmd.gate)
        if name in _PSEUDO:
            continue
        if name == _MEASURE:
            n_meas += 1
            continue
        if name == _RESET:
            n_resets += 1
            continue
        n_gates += 1
        arity = len(cmd.qubits)
        if arity == 1:
            n_1q += 1
        elif arity == 2:
            n_2q += 1
        else:
            n_3q_plus += 1
        if name in _T_GATES:
            t_count += 1
        if name == _OPAQUE or table[name][1] > 0:
            exact_t = False
        # A width >= 3 MCZ has no T-free exact form: like a parametric Rz,
        # its T-cost is unknown until lowered or synthesized (§19).  Width 2
        # (= CZ) is Clifford and stays countable.
        if name == _MCZ and arity >= 3:
            exact_t = False

    histogram = {k: v for k, v in dag.count_ops().items() if k not in _PSEUDO}

    return ResourceVector(
        n_qubits=dag.n_qubits if n_qubits is None else n_qubits,
        depth=dag.depth(),
        n_gates=n_gates,
        n_1q=n_1q,
        n_2q=n_2q,
        n_3q_plus=n_3q_plus,
        n_measurements=n_meas,
        n_resets=n_resets,
        t_count=t_count if exact_t else None,
        swap_count=(histogram.get("SWAP", 0) if provenance.stage is Stage.ROUTED else None),
        op_histogram=histogram,
        provenance=provenance,
    )
