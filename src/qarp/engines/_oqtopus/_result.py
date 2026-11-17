"""OQTOPUS sampling payload → ``qx.SamplingResult``."""

from typing import Mapping, Optional, Sequence

import qarpx as qx


def counts_to_sampling_result(
    counts: Mapping[str, int],
    *,
    n_qubits: int,
    n_shots: int,
    wiring: Optional[Sequence[Optional[int]]] = None,
    mitigated: bool = False,
) -> qx.SamplingResult:
    """Wrap service counts in the §14 result type, keyed by logical qubit.

    Keys are the classical register as a bitstring with ``c[0]`` rightmost
    (the OQTOPUS user-API payload).  ``wiring[c]`` names the logical qubit
    whose measurement landed in ``c[c]`` (``None`` for an unused bit); the
    default is the identity over ``n_qubits``, under which ``int(key, 2)``
    is already qarp's LSB integer key (§1).  Spaces (multi-register
    spellings) are ignored.

    ``mitigated=True`` accepts counts summing to less than ``n_shots`` and
    reports their total as the shot count: the service's readout mitigation
    floors each corrected count to an integer.

    Raises:
        ValueError: a key whose width is not the register width, a wiring
            naming a qubit outside ``range(n_qubits)``, or counts that do not
            sum to ``n_shots`` (exceed it, when ``mitigated``).
    """
    if wiring is None:
        wiring = list(range(n_qubits))
    width = len(wiring)
    for q in wiring:
        if q is not None and not 0 <= q < n_qubits:
            raise ValueError(
                f"counts_to_sampling_result: wiring names qubit {q} outside range({n_qubits})"
            )
    out: dict[int, int] = {}
    for key, n in counts.items():
        bits = key.replace(" ", "")
        if len(bits) != width:
            raise ValueError(
                f"counts_to_sampling_result: key {key!r} has {len(bits)} bits, "
                f"expected the register width {width} (n_qubits={n_qubits})"
            )
        k = 0
        for c, q in enumerate(wiring):
            if q is not None and bits[width - 1 - c] == "1":
                k |= 1 << q
        out[k] = out.get(k, 0) + int(n)
    total = sum(out.values())
    if total > n_shots or (total != n_shots and not mitigated):
        raise ValueError(f"counts_to_sampling_result: counts sum to {total} but n_shots={n_shots}")
    return qx.SamplingResult(n_qubits, out, total)
