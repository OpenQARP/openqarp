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
    """Calculate x positions for gates with layer flattening."""
    qubit_x = {q: 0 for q in qubits}
    x_positions = []

    # mapping for all qubits, named or unnamed
    qubit_to_index = {q: i for i, q in enumerate(qubits)}

    for cmd in commands:
        if str(cmd.op).lower() == "measure":
            continue

        involved_qubits = sorted(set(flatten_qubits(cmd.qubits)), key=lambda q: qubit_to_index[q])

        # Skip gates with no qubits (e.g., global Phase gate)
        # These should already be filtered out, but handle defensively
        if not involved_qubits:
            continue

        min_idx = min(qubit_to_index[q] for q in involved_qubits)
        max_idx = max(qubit_to_index[q] for q in involved_qubits)
        q_span = [q for q in qubits if min_idx <= qubit_to_index[q] <= max_idx]

        x = max(qubit_x[q] for q in q_span)
        x_positions.append(x + 1 / 4)  # shift for nicer appearance

        for q in q_span:
            qubit_x[q] = x + 1

    return x_positions
