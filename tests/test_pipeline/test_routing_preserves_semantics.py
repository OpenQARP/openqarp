"""Phase 2b of the pipeline property suite: routing invariance.

For any counts-consuming primitive at EXACT readout, running on a routed
device (line / ring / grid, 1q/2q rebase target) returns the same result as
the architecture-free run of the same configuration: the router's SWAP
insertion plus the engine's logical←physical reindex must compose to the
identity on results.  The unrouted run is the oracle (independently
implemented path — no routing, no rebase, no reindex).

EXACT readout makes the comparison deterministic (no per-shot RNG whose
consumption pattern differs between the two compiled circuits), so the
tolerance is float roundoff, not statistics.  ``StateVector`` is excluded:
amplitude consumption over a *permuted* register is a documented rejection
(pinned in tests/test_engines/test_capability_validation.py), not an
invariance.  Every drawn configuration must RUN — a ``CapabilityError``
here is a failure.
"""

import pytest
from hypothesis import example, given

from tests.strategies import (
    PRIMITIVES,
    PipelineConfig,
    pipeline_configs,
    results_allclose,
    run_pipeline,
)

pytestmark = pytest.mark.property

# Amplitude consumption over a permuted register rejects by design; every
# counts path (finite or ExactResult) is reindexed and must be invariant.
COUNTS_PRIMITIVES = tuple(p for p in PRIMITIVES if p != "StateVector")


def _cfg(block_name: str, primitive: str, device: str, values_seed: int) -> PipelineConfig:
    return PipelineConfig(
        block_name=block_name,
        primitive=primitive,
        engine="qarp",
        device=device,
        opt_level=0,
        shots=None,
        noise=None,
        values_seed=values_seed,
    )


@example(cfg=_cfg("TrotterBlock-symbolic-time", "Sampler", "grid", 346))  # padded width
@example(cfg=_cfg("GivensBlock", "SWAPTest", "line", 0))  # CSWAP rebase + route
@example(cfg=_cfg("HEABlock", "PauliAveraging", "ring", 5))  # grouped, ring wrap
@given(
    cfg=pipeline_configs(
        primitives=COUNTS_PRIMITIVES,
        engines=("qarp",),
        devices=("line", "ring", "grid"),
        shots_axis=(None,),
        noise_axis=(None,),
    )
)
def test_routing_preserves_results(cfg):
    routed = run_pipeline(cfg)
    unrouted = run_pipeline(cfg._replace(device=None))
    assert results_allclose(routed, unrouted), (
        f"routed vs unrouted disagree for {cfg}: {routed!r} != {unrouted!r}"
    )


# ── Directed routing stays directed (pipeline_hardening_plan.md P1.18) ────
#
# Plain parametrised rows, not hypothesis: the oracle is `assert_directions`
# plus the pre-compile unitary under the final logical→physical map.

import numpy as np

import qarpx as qx
from qarp.blocks import SimpleBlock
from qarp.devices import Device
from qarp.errors import CapabilityError
from tests.test_emit.conftest import qarpx_unitary

_META = [qx.GateType.GPhase, qx.GateType.Measure, qx.GateType.Barrier, qx.GateType.Reset]
_SINGLES = [
    qx.GateType.H,
    qx.GateType.Rz,
    qx.GateType.Rx,
    qx.GateType.Ry,
    qx.GateType.X,
    qx.GateType.S,
    qx.GateType.T,
]


def _gate_set(name, two_qubit):
    gs = qx.GateSet()
    gs.name = name
    gs.allowed = set(_SINGLES) | set(two_qubit) | set(_META)
    return gs


def _directed_arch(kind, n):
    edges = [(i, i + 1) for i in range(n - 1)]
    if kind == "ring":
        edges.append((n - 1, 0))
    return qx.Architecture(n, edges, f"directed_{kind}")


def _logical_unitary(u_physical, initial, final, n):
    """Re-express a physical-register unitary in logical qubit order.

    Rows follow the final map, columns the initial placement (§14): a router
    may place qubits before the first gate and move them after, and the two
    coincide only when it inserts no SWAP.
    """

    def basis(l2p):
        return [sum(((j >> l) & 1) << l2p[l] for l in range(n)) for j in range(2**n)]

    return u_physical[np.ix_(basis(final), basis(initial))]


def _asymmetric_circuit():
    b = SimpleBlock(4)
    b.h(0).cx(3, 0).cy(2, 0).cx(0, 3).ecr(1, 3).rzz(0, 2, 0.3).cu(3, 1, 0.1, 0.2, 0.3, 0.0)
    b.build()
    return b


@pytest.mark.parametrize("target", ["cx", "cx_cz"])
@pytest.mark.parametrize("kind", ["line", "ring"])
def test_directed_compile_keeps_every_orientation(kind, target):
    two_qubit = [qx.GateType.CX] + ([qx.GateType.CZ] if target == "cx_cz" else [])
    arch = _directed_arch(kind, 4)
    dev = Device(4, architecture=arch, gate_set=_gate_set(target, two_qubit), directedness=True)
    b = _asymmetric_circuit()
    out = qx.compile_for_device(b.flatten(), 4, dev)
    qx.assert_directions(out.commands, arch)  # no reversed asymmetric gate
    assert all(c.gate != qx.GateType.SWAP for c in out.commands)
    u_logical = _logical_unitary(
        qarpx_unitary(list(out.commands), 4),
        list(out.initial_logical_to_physical),
        list(out.final_logical_to_physical),
        4,
    )
    np.testing.assert_allclose(u_logical, qarpx_unitary(list(b.flatten()), 4), atol=1e-10)


@pytest.mark.parametrize("gate", ["ecr", "ch", "csx", "csxdg", "cx"])
def test_reversed_asymmetric_gate_off_the_directed_edge_raises(gate):
    arch = qx.Architecture(2, [(0, 1)], "one_edge")
    b = SimpleBlock(2)
    getattr(b, gate)(1, 0)
    b.build()
    with pytest.raises(CapabilityError, match="against the directed architecture"):
        qx.assert_directions(b.flatten(), arch)
    # The router refuses to H-conjugate anything but CX; CX itself is flipped
    # or relaid — either way the compiled circuit passes the assertion.
    native = getattr(
        qx.GateType, {"ecr": "ECR", "ch": "CH", "csx": "CSX", "csxdg": "CSXdg", "cx": "CX"}[gate]
    )
    dev = Device(
        2,
        architecture=arch,
        gate_set=_gate_set("wide", [qx.GateType.CX, native]),
        directedness=True,
    )
    out = qx.compile_for_device(b.flatten(), 2, dev)
    qx.assert_directions(out.commands, arch)


def test_symmetric_gate_ignores_direction():
    arch = qx.Architecture(2, [(0, 1)], "one_edge")
    b = SimpleBlock(2)
    b.cz(1, 0)
    b.build()
    qx.assert_directions(b.flatten(), arch)
