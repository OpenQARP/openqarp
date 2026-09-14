"""NoiseModel wrapper plumbing: gate-set resolution shorthands, coercion
contracts, builder channel attachment, and composition. The channel physics
itself is pinned by the C++ suite; these tests pin the Python surface.
"""

import pytest

import qarpx as qx
from qarp.devices import NoiseModel
from qarp.devices._noise_model import _coerce_gate, _resolve_gates


def test_resolve_gates_shorthands():
    one_q = _resolve_gates("1q", "1q")
    two_q = _resolve_gates("2q", "2q")
    all_q = _resolve_gates("all", "all")
    assert qx.GateType.X in one_q and qx.GateType.CX not in one_q
    assert qx.GateType.CX in two_q and qx.GateType.X not in two_q
    assert set(one_q) | set(two_q) <= set(all_q)


def test_resolve_gates_none_uses_default():
    assert _resolve_gates(None, "1q") == _resolve_gates("1q", "1q")


def test_resolve_gates_unknown_shorthand_raises():
    with pytest.raises(ValueError, match="unknown gate_set shorthand"):
        _resolve_gates("3q", "1q")


def test_resolve_gates_gateset_object_passthrough():
    assert _resolve_gates(qx.full_gateset_1q(), "2q") == _resolve_gates("1q", "1q")


def test_resolve_gates_iterable_of_names_and_types():
    gates = _resolve_gates(["X", qx.GateType.CX], "1q")
    assert gates == [qx.GateType.X, qx.GateType.CX]


def test_coerce_gate_contracts():
    assert _coerce_gate("H") is qx.GateType.H
    with pytest.raises(ValueError, match="unknown GateType"):
        _coerce_gate("NotAGate")
    with pytest.raises(TypeError, match="cannot coerce"):
        _coerce_gate(3.14)


def test_pauli_builder_attaches_1q_channels():
    nm = NoiseModel.pauli(p_x=0.01, p_z=0.02)
    assert nm.has_any_channel()
    assert nm.has_channel(qx.GateType.X)
    assert not nm.has_channel(qx.GateType.CX)


def test_amplitude_damping_builder():
    nm = NoiseModel.amplitude_damping(0.05, gate_set=["H"])
    assert nm.has_channel(qx.GateType.H)
    assert not nm.has_channel(qx.GateType.X)


def test_composition_merges_channels():
    combined = NoiseModel.bit_flip(0.01, gate_set=["X"]) + NoiseModel.pauli(
        p_z=0.02, gate_set=["H"]
    )
    assert combined.has_channel(qx.GateType.X)
    assert combined.has_channel(qx.GateType.H)

    grown = NoiseModel.bit_flip(0.01, gate_set=["X"])
    grown += NoiseModel.pauli(p_z=0.02, gate_set=["H"])
    assert grown.has_channel(qx.GateType.X) and grown.has_channel(qx.GateType.H)


def test_enabled_round_trip_and_inner():
    nm = NoiseModel.bit_flip(0.01)
    assert nm.enabled
    nm.enabled = False
    assert not nm.enabled
    assert isinstance(nm.inner, qx.NoiseModel)


def test_shorthands_drop_non_physical_gates():
    """Regression: ``full_gateset_1q`` is a rebase target, not a noise list.

    It admits ``Measure``/``Barrier``/``GPhase`` so rebasing cannot drop them;
    the simulator injects no error on those and the factories refuse the arity.
    """
    for shorthand in ("1q", "2q", "all"):
        gates = _resolve_gates(shorthand, "1q")
        assert gates, shorthand
        assert all(qx.gate_is_physical(g) for g in gates), shorthand
        assert qx.GateType.GPhase not in gates, shorthand


@pytest.mark.parametrize(
    "builder, shorthand",
    [
        ("pauli", "1q"),
        ("bit_flip", "1q"),
        ("amplitude_damping", "1q"),
        ("depolarizing", "2q"),
        ("depolarizing", "all"),
    ],
)
def test_builders_accept_their_shorthands(builder, shorthand):
    kwargs = {"p_x": 0.01} if builder == "pauli" else {"p": 0.01}
    nm = getattr(NoiseModel, builder)(gate_set=shorthand, **kwargs)
    assert nm.has_any_channel()
