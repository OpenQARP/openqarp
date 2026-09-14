"""Sampler workloads reuse the statevector track's circuits verbatim.

Same gate sequences, same seeds — so the two generated documents can be read
side by side as "exact amplitudes" vs "shots" on identical work.
"""

from benchmarks.sampler import spec
from benchmarks.statevector import inputs as circuits

_BASE = {
    "brickwork_sample": "brickwork",
    "trotter_sample": "trotter_step",
    "qpe_sample": "qpe_phase",
}


def workload(family: str, size: int) -> tuple[int, list, int]:
    """Return (n_qubits, ops, shots) for a family at its ladder point.

    For `shots_axis` the ladder value *is* the shot count, at a fixed circuit
    width — the qubit and shot axes never multiply into one another.
    """
    if family == "shots_axis":
        n_qubits, ops = circuits.brickwork(spec.SHOTS_AXIS_QUBITS)
        return n_qubits, ops, size
    n_qubits, ops = circuits.CIRCUITS[_BASE[family]](size)
    return n_qubits, ops, spec.SHOTS
