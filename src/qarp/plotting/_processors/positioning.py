from collections.abc import Iterable
from typing import Dict


def flatten_qubits(qubit_input):
    """Flatten nested qubit structures."""
    flat = []
    for item in qubit_input:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            flat.extend(flatten_qubits(item))
        else:
            flat.append(item)
    return flat


def get_y_positions(qubits, invert_order: bool = False) -> Dict:
    """Calculate y positions for qubits."""
    n = len(qubits)
    return {q: (i if invert_order else n - 1 - i) for i, q in enumerate(qubits)}


def get_x_positions_list(commands, qubits):
    """Calculate x positions for gates with layer flattening.

    A Barrier draws as a thin dashed line rather than a gate-sized box, so it
    is given half a layer's width instead of a full one -- a full width reads
    as an empty gate-sized gap on either side of it.
    """
    qubit_x = {q: 0.0 for q in qubits}
    x_positions = []

    # mapping for all qubits, named or unnamed
    qubit_to_index = {q: i for i, q in enumerate(qubits)}

    for cmd in commands:
        if str(cmd.op).lower() == "measure":
            continue

        involved_qubits = sorted(set(flatten_qubits(cmd.qubits)), key=lambda q: qubit_to_index[q])
        is_barrier = str(cmd.op).lower() == "barrier"

        if not involved_qubits:
            if not is_barrier:
                # Skip other gates with no qubits (e.g., global phase gate).
                # These should already be filtered out, but handle defensively.
                continue
            # A qubit-less Barrier fences the whole register.
            q_span = list(qubits)
        else:
            min_idx = min(qubit_to_index[q] for q in involved_qubits)
            max_idx = max(qubit_to_index[q] for q in involved_qubits)
            q_span = [q for q in qubits if min_idx <= qubit_to_index[q] <= max_idx]

        width = 0.5 if is_barrier else 1.0
        x = max(qubit_x[q] for q in q_span)
        x_positions.append(x + width / 4)  # shift for nicer appearance

        for q in q_span:
            qubit_x[q] = x + width

    return x_positions
