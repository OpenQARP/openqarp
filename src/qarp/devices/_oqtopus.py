"""``device_from_oqtopus``: an OQTOPUS ``device_info`` payload as a ``qx.Device``."""

from typing import Any, Mapping

import qarpx as qx


def device_from_oqtopus(info: Mapping[str, Any]) -> qx.Device:
    """Build the passive ``qx.Device`` an OQTOPUS device describes.

    ``info`` is the ``device_info`` JSON the service serves (the
    device-gateway ``device_topology.json`` shape): ``qubits[].id`` numbers
    the program register and ``couplings[].control/target`` reference those
    ids.  The gate set is ``sx_rz_gateset()`` and every coupling is a
    directed edge, which is what a QPU gateway reports.

    The register index is taken to be ``id``, not ``physical_id``; a QPU
    gateway has not confirmed this yet (the plan's open item 1).

    Raises:
        ValueError: a qubit id outside ``range(len(qubits))`` or a coupling
            naming an unknown qubit.
    """
    qubits = info["qubits"]
    n = len(qubits)
    ids = []
    for q in qubits:
        qid = int(q["id"])
        if not 0 <= qid < n:
            raise ValueError(f"device_from_oqtopus: qubit id {qid} outside range({n})")
        ids.append(qid)
    if len(set(ids)) != n:
        raise ValueError("device_from_oqtopus: duplicate qubit id")
    edges = []
    for c in info.get("couplings", []):
        control, target = int(c["control"]), int(c["target"])
        if control not in ids or target not in ids:
            raise ValueError(
                f"device_from_oqtopus: coupling ({control}, {target}) names an unknown qubit"
            )
        edges.append((control, target))
    name = str(info.get("device_id") or info.get("name") or "oqtopus")
    return qx.Device(
        n,
        architecture=qx.Architecture(n, edges, name),
        gate_set=qx.sx_rz_gateset(),
        directedness=True,
    )
