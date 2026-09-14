"""ResourceEstimator: stage staging, SWAP semantics, report behavior.

Routing counts assert invariants, not exact SABRE numbers — tie-break
stability across versions is not part of the contract.  The one equality we
do pin is pure rebase arithmetic: SWAP → 3 CX, so the TARGET 2q count is
the ROUTED one plus two CX per router SWAP.
"""

import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock
from qarp.devices import (
    Device,
    get_all_to_all_architecture,
    get_nearest_neighbour_architecture,
)
from qarp.resources import ResourceEstimator, Stage, estimate


def _fanout_cx_block(n: int = 5) -> SimpleBlock:
    """CX(0, k) fan-out: forces SWAPs on a line architecture."""
    b = SimpleBlock(n)
    b.h(0)
    for k in range(1, n):
        b.cx(0, k)
    b.measure([(q, q) for q in range(n)])
    return b


def _line_device(n: int = 5) -> Device:
    return Device(n, architecture=get_nearest_neighbour_architecture(1, n))


def test_logical_only_without_config():
    rep = estimate(_fanout_cx_block())
    assert rep.stages == (Stage.LOGICAL,)
    v = rep[Stage.LOGICAL]
    assert v.swap_count is None
    assert v.provenance.gateset is None and v.provenance.router is None


def test_full_pipeline_stage_semantics():
    rep = estimate(
        _fanout_cx_block(),
        gateset=qx.clifford_t_gateset(),
        device=_line_device(),
        device_label="line-5",
    )
    assert rep.stages == (
        Stage.LOGICAL,
        Stage.OPTIMIZED,
        Stage.ROUTED,
        Stage.TARGET,
    )
    routed, target = rep[Stage.ROUTED], rep[Stage.TARGET]

    # SWAPs exist only at ROUTED
    assert rep[Stage.LOGICAL].swap_count is None
    assert rep[Stage.OPTIMIZED].swap_count is None
    assert routed.swap_count is not None and routed.swap_count >= 1
    assert target.swap_count is None
    assert "SWAP" not in target.op_histogram

    # routing only adds 2q gates; rebase turns each SWAP into 3 CX
    assert routed.n_2q >= rep[Stage.OPTIMIZED].n_2q
    assert target.n_2q == routed.n_2q + 2 * routed.swap_count

    assert routed.provenance.router == "Sabre"
    assert routed.provenance.device == "line-5"
    assert target.provenance.gateset == "clifford_t"


def test_all_to_all_architecture_needs_no_swaps():
    rep = estimate(
        _fanout_cx_block(),
        gateset=qx.clifford_t_gateset(),
        device=Device(5, architecture=get_all_to_all_architecture(5)),
    )
    assert rep[Stage.ROUTED].swap_count == 0  # zero, not None: routing ran


def test_routed_stages_use_physical_width():
    rep = estimate(
        _fanout_cx_block(5),
        gateset=qx.clifford_t_gateset(),
        device=_line_device(6),
    )
    assert rep[Stage.LOGICAL].n_qubits == 5
    assert rep[Stage.ROUTED].n_qubits == 6
    assert rep[Stage.TARGET].n_qubits == 6


def test_device_requires_gateset():
    with pytest.raises(ValueError, match="gateset"):
        ResourceEstimator(device=_line_device())


def test_device_too_small_rejected():
    with pytest.raises(Exception):
        estimate(
            _fanout_cx_block(5),
            gateset=qx.clifford_t_gateset(),
            device=_line_device(3),
        )


def test_accepts_raw_commands():
    b = _fanout_cx_block()
    b.build()
    rep = estimate(b.flatten(), gateset=qx.clifford_t_gateset())
    assert rep.stages == (Stage.LOGICAL, Stage.OPTIMIZED)


def test_estimator_is_reusable():
    est = ResourceEstimator(gateset=qx.clifford_t_gateset(), device=_line_device())
    first = est.estimate(_fanout_cx_block())
    second = est.estimate(_fanout_cx_block())
    assert first[Stage.OPTIMIZED] == second[Stage.OPTIMIZED]


def test_report_mapping_and_serialization():
    rep = estimate(
        _fanout_cx_block(),
        gateset=qx.clifford_t_gateset(),
        device=_line_device(),
    )
    assert len(rep) == 4
    assert list(rep) == list(rep.stages)
    assert rep.final is rep[Stage.TARGET]
    d = rep.to_dict()
    assert d["schema_version"] == 1
    assert list(d["stages"]) == ["logical", "optimized", "routed", "target"]


def test_target_stage_is_custom_free_at_o1():
    """Rebase totality end-to-end: with O1 fusion active, the TARGET stage
    must re-open fused Customs (ZYZ) — every rotation visible, so a modeled
    T-count is a real number, not None (§19 Custom flow-through)."""
    from qarp.resources import Stage, estimate
    from tests.conftest import CountRzModeler

    b = SimpleBlock(2)
    b.h(0)
    b.t(0)
    b.rz(0, 0.7)
    b.s(0)
    b.cx(0, 1)
    b.h(1)
    b.tdg(1)
    b.build()

    report = estimate(
        b,
        gateset=qx.clifford_t_rz_gateset(),
        opt_level=qx.OptLevel.O1,
        device=Device(2, architecture=get_all_to_all_architecture(2)),
        modeler=CountRzModeler(),
    )
    v = report[Stage.TARGET]
    assert "Custom" not in v.op_histogram
    assert v.t_count_modeled is not None
