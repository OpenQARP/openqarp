"""``device_from_oqtopus``: the OQTOPUS ``device_info`` payload as a ``qx.Device``.

Oracle: the device-gateway example topology, transcribed verbatim in
``tests/oqtopus_device_topology.py`` with its source URL.  The builder reads
``qubits[].id`` and ``couplings[].control/target`` (plan open item 1); the
example's ``physical_id`` values differ from ``id``, so reading the wrong
field changes the qubit count or the edge list.
"""

import copy

import pytest

import qarpx as qx
from qarp.devices import device_from_oqtopus
from tests.oqtopus_device_topology import DEVICE_TOPOLOGY


def test_builds_the_example_topology():
    dev = device_from_oqtopus(DEVICE_TOPOLOGY)
    assert isinstance(dev, qx.Device)
    assert dev.n_qubits == 3
    assert dev.architecture.name == "anemone"
    assert dev.architecture.n_qubits == 3
    assert sorted(dev.architecture.edges) == [(0, 1), (2, 1)]
    assert dev.directedness is True
    assert dev.gate_set.name == "sx_rz"
    assert dev.gate_set.rules == "sx_rz"
    assert dev.noise_model is None


def test_directed_edges_keep_control_target_orientation():
    arch = device_from_oqtopus(DEVICE_TOPOLOGY).architecture
    assert arch.has_directed_edge(0, 1)
    assert not arch.has_directed_edge(1, 0)
    assert arch.has_directed_edge(2, 1)
    assert not arch.has_directed_edge(1, 2)


def test_rejects_coupling_naming_an_unknown_qubit():
    info = copy.deepcopy(DEVICE_TOPOLOGY)
    info["couplings"].append({"control": 1, "target": 7})
    with pytest.raises(ValueError, match="coupling"):
        device_from_oqtopus(info)


def test_rejects_qubit_id_outside_range():
    info = copy.deepcopy(DEVICE_TOPOLOGY)
    info["qubits"][2]["id"] = 5
    with pytest.raises(ValueError, match="qubit id"):
        device_from_oqtopus(info)


def test_falls_back_to_name_when_device_id_is_absent():
    info = copy.deepcopy(DEVICE_TOPOLOGY)
    del info["device_id"]
    assert device_from_oqtopus(info).architecture.name == "anemone"


def test_exported_from_qarp_devices():
    import qarp.devices

    assert "device_from_oqtopus" in qarp.devices.__all__
