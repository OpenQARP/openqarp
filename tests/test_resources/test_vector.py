"""Contract tests: the wire format is pinned here.

The golden dict below is what external resource-estimation tooling parses.
If a change breaks this test, it breaks them: bump SCHEMA_VERSION and
coordinate — never just update the dict.
"""

import dataclasses

import pytest

from qarp.resources import SCHEMA_VERSION, Provenance, ResourceVector, Stage


def _vector() -> ResourceVector:
    return ResourceVector(
        n_qubits=5,
        depth=12,
        n_gates=9,
        n_1q=3,
        n_2q=6,
        n_3q_plus=0,
        n_measurements=5,
        n_resets=0,
        t_count=None,
        swap_count=2,
        op_histogram={"Rz": 2, "SWAP": 2, "Measure": 5, "CX": 4, "H": 1},
        provenance=Provenance(
            stage=Stage.ROUTED,
            gateset="clifford_t_rz",
            opt_level="O1",
            router="Sabre",
            device="line-5",
            modeler=None,
        ),
    )


GOLDEN = {
    "schema_version": 1,
    "n_qubits": 5,
    "depth": 12,
    "n_gates": 9,
    "n_1q": 3,
    "n_2q": 6,
    "n_3q_plus": 0,
    "n_measurements": 5,
    "n_resets": 0,
    "t_count": None,
    "swap_count": 2,
    "op_histogram": {"Rz": 2, "SWAP": 2, "Measure": 5, "CX": 4, "H": 1},
    "provenance": {
        "stage": "routed",
        "gateset": "clifford_t_rz",
        "opt_level": "O1",
        "router": "Sabre",
        "device": "line-5",
        "modeler": None,
        "synthesis": None,
    },
    "t_count_modeled": None,
    "extras": {},
}


def test_wire_format_golden():
    assert _vector().to_dict() == GOLDEN


def test_schema_version_pinned():
    # v1 is the published baseline consumers pin against.
    assert SCHEMA_VERSION == 1


def test_arity_buckets_partition_n_gates():
    """The wire contract external tooling relies on: no gate is invisible to
    the arity breakdown, whatever its width."""
    v = _vector()
    assert v.n_1q + v.n_2q + v.n_3q_plus == v.n_gates


def test_from_dict_round_trip():
    v = _vector()
    assert ResourceVector.from_dict(v.to_dict()) == v


def test_from_dict_rejects_unknown_schema_version():
    d = _vector().to_dict()
    d["schema_version"] = 99
    with pytest.raises(ValueError, match="schema_version"):
        ResourceVector.from_dict(d)


def test_vector_is_frozen():
    v = _vector()
    with pytest.raises(dataclasses.FrozenInstanceError):
        v.n_qubits = 3  # type: ignore[misc]


def test_with_model_only_touches_modeled_fields():
    v = _vector()
    m = v.with_model(
        modeler="count_rz:t=1",
        t_count_modeled=30.0,
        extras={"n_rz_digital": 1.0},
    )
    assert m.t_count_modeled == 30.0
    assert m.extras == {"n_rz_digital": 1.0}
    assert m.provenance.modeler == "count_rz:t=1"
    # counted fields and the rest of provenance are untouched
    assert m.n_gates == v.n_gates and m.swap_count == v.swap_count
    assert m.provenance.stage is v.provenance.stage
    assert m.provenance.router == v.provenance.router
