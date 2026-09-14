"""End-to-end tests for the qarpx-native device pipeline.

Replaces the pre-pytket-removal skip-marked file.  Covers Device, Architecture,
NoiseModel, the router, and the rebase → route → re-rebase pipeline
exposed via ``qarp.devices.compile_for_device``.

The C++ side already has unit-level coverage (tests/cpp/test_architecture.cpp,
test_router.cpp, test_noise_model.cpp, test_device.cpp).  This file complements
those by exercising the Python surface end-to-end through Block construction,
the Engine, and the SamplingResult contract.
"""

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock
from qarp.devices import (
    Architecture,
    Device,
    NoiseModel,
    compile_for_device,
    get_all_to_all_architecture,
    get_nearest_neighbour_architecture,
)
from qarp.engines import QarpEngine
from qarp.errors import CapabilityError

# ── Architecture / GateSet helpers ──────────────────────────────────────────


def test_nearest_neighbour_grid_returns_qx_architecture():
    arch = get_nearest_neighbour_architecture(2, 2)
    assert isinstance(arch, qx.Architecture)
    assert arch.n_qubits == 4
    # 2×2 grid has 4 edges: (0,1) (0,2) (1,3) (2,3).
    assert len(arch.edges) == 4
    assert arch.is_connected(0, 1)
    assert arch.is_connected(0, 2)
    assert not arch.is_connected(0, 3)  # diagonal not adjacent


def test_all_to_all_returns_qx_architecture():
    arch = get_all_to_all_architecture(3)
    assert isinstance(arch, qx.Architecture)
    assert len(arch.edges) == 3  # C(3,2)


def test_edges_setter_rebuilds_the_connectivity_queries():
    """P2.9: the queries read cached indices; assigning ``edges`` must
    refresh them, and a rejected assignment must leave the old ones."""
    arch = qx.Architecture(4, [(0, 1)])
    assert arch.is_connected(0, 1) and arch.has_directed_edge(0, 1)
    assert not arch.has_directed_edge(1, 0)

    arch.edges = [(2, 3), (3, 1)]
    assert arch.edges == [(2, 3), (3, 1)]
    assert not arch.is_connected(0, 1)
    assert arch.is_connected(1, 3) and arch.has_directed_edge(3, 1)
    assert arch.neighbours(3) == [1, 2]
    assert arch.shortest_path(2, 1) == [2, 3, 1]

    with pytest.raises(ValueError):
        arch.edges = [(0, 4)]
    assert arch.edges == [(2, 3), (3, 1)]
    assert arch.is_connected(2, 3)


# ── Device ──────────────────────────────────────────────────────────────────


def test_default_device_is_thin_wrapper():
    d = Device(256)
    assert d.n_qubits == 256
    assert d.architecture is None
    assert d.noise_model is None
    assert d.gate_set is None
    assert d.directedness is False


def test_check_fits_raises_when_too_big():
    d = Device(2)
    with pytest.raises(CapabilityError, match="exposes only 2"):
        d.check_fits(3)


# ── compile_for_device pipeline ─────────────────────────────────────────────


def test_compile_for_device_pass_through_when_no_arch_no_gateset():
    b = SimpleBlock(n_qubits=2, name="bell")
    b.h(0)
    b.cx(0, 1)
    b.build()
    cmds = b.flatten()
    out = compile_for_device(cmds, 2, Device(2))
    # No transpilation or routing — same length, no SWAPs, identity mapping.
    assert len(out.commands) == 2
    assert not any(c.gate == qx.GateType.SWAP for c in out.commands)
    assert list(out.final_logical_to_physical) == [0, 1]


def test_compile_for_device_block_overload_matches_commands_overload():
    """``compile_for_device(block, device)`` must equal
    ``compile_for_device(block.flatten(), block.n_qubits, device)``."""
    b = SimpleBlock(n_qubits=3, name="long_cx")
    b.h(0)
    b.cx(0, 2)
    b.build()
    arch = get_nearest_neighbour_architecture(3, 1)
    dev = Device(3, architecture=arch)

    block_form = compile_for_device(b, dev)
    commands_form = compile_for_device(b.flatten(), b.n_qubits, dev)

    assert len(block_form.commands) == len(commands_form.commands)
    assert list(block_form.final_logical_to_physical) == list(
        commands_form.final_logical_to_physical
    )
    for a, c in zip(block_form.commands, commands_form.commands, strict=True):
        assert a.gate == c.gate
        assert list(a.qubits) == list(c.qubits)


def test_compile_for_device_inserts_swaps_on_disconnected_pair():
    b = SimpleBlock(n_qubits=3, name="long_cx")
    b.h(0)
    b.cx(0, 2)
    b.build()
    arch = get_nearest_neighbour_architecture(3, 1)  # 0-1-2 chain
    # Pin Lite: this observes the SWAP-insertion mechanic itself; the default
    # (Sabre) mapping search legitimately avoids the SWAP on this fixture.
    out = compile_for_device(
        b.flatten(), 3, Device(3, architecture=arch), router=qx.RouterKind.Lite
    )
    swap_count = sum(1 for c in out.commands if c.gate == qx.GateType.SWAP)
    cx_count = sum(1 for c in out.commands if c.gate == qx.GateType.CX)
    assert swap_count >= 1
    assert cx_count == 1

    # Default (Sabre): never more SWAPs than Lite — here, none.
    sabre_out = compile_for_device(b.flatten(), 3, Device(3, architecture=arch))
    assert sum(1 for c in sabre_out.commands if c.gate == qx.GateType.SWAP) == 0


def test_compile_for_device_decomposes_swap_when_target_gateset_excludes_it():
    # Force the second-rebase pass: arch needs SWAPs but the gate set doesn't
    # include SWAP, so the second `Transpiler` decomposes them into 3 CXs.
    b = SimpleBlock(n_qubits=3, name="long_cx")
    b.h(0)
    b.cx(0, 2)
    b.build()
    arch = get_nearest_neighbour_architecture(3, 1)
    gs = qx.GateSet()
    gs.name = "no_swap"
    gs.allowed = {
        qx.GateType.H,
        qx.GateType.X,
        qx.GateType.Y,
        qx.GateType.Z,
        qx.GateType.S,
        qx.GateType.Sdg,
        qx.GateType.T,
        qx.GateType.Tdg,
        qx.GateType.Rx,
        qx.GateType.Ry,
        qx.GateType.Rz,
        qx.GateType.CX,
        qx.GateType.Measure,
        qx.GateType.Barrier,
        qx.GateType.GPhase,
    }
    # Pin Lite: the point is the second-rebase-decomposes-SWAPs mechanic, and
    # the default (Sabre) mapping search avoids the SWAP entirely here.
    out = compile_for_device(
        b.flatten(),
        3,
        Device(3, architecture=arch, gate_set=gs),
        router=qx.RouterKind.Lite,
    )
    gates = [c.gate for c in out.commands]
    assert qx.GateType.SWAP not in gates
    # Original CX + 3-CX SWAP expansion = at least 4 CXs.
    assert gates.count(qx.GateType.CX) >= 4


def test_compile_for_device_directedness_respected():
    # Directed edge (0,1) only; CX(1,0) must come out direction-correct.  The
    # portfolio layout search may satisfy that either by H-conjugation or by
    # placing logical 1 on physical 0 so the gate lands on the edge directly
    # (1 command instead of 5) — both are valid; the contract is direction.
    arch = Architecture(2, [(0, 1)], "directed_edge")
    b = SimpleBlock(n_qubits=2, name="reversed_cx")
    b.cx(1, 0)
    b.build()
    out = compile_for_device(
        b.flatten(),
        2,
        Device(2, architecture=arch, directedness=True),
    )
    for c in out.commands:
        if c.gate == qx.GateType.CX:
            assert tuple(c.qubits) == (0, 1)
    assert sorted(out.final_logical_to_physical) == [0, 1]
    assert len(out.commands) <= 5  # never worse than plain H-conjugation


def test_route_directedness_h_conjugates_cx_under_pinned_mapping():
    """The H-conjugation mechanism itself, with the layout search disabled.

    Pinning `initial_mapping` to the identity forces CX(1, 0) against the
    directed edge, so the router must emit H(1) H(0) CX(0,1) H(1) H(0).
    """
    opts = qx.RoutingOptions()
    opts.arch = qx.Architecture(2, [(0, 1)], "directed_edge")
    opts.directedness = True
    opts.initial_mapping = [0, 1]
    opts.router = qx.RouterKind.Sabre

    b = SimpleBlock(n_qubits=2, name="reversed_cx")
    b.cx(1, 0)
    b.build()
    out = qx.route(b.flatten(), opts)
    gates = [c.gate for c in out.commands]
    assert gates.count(qx.GateType.H) == 4
    assert gates.count(qx.GateType.CX) == 1
    cx = next(c for c in out.commands if c.gate == qx.GateType.CX)
    assert tuple(cx.qubits) == (0, 1)


def test_compile_for_device_check_fits_raises():
    b = SimpleBlock(n_qubits=3, name="too_big")
    b.h(0)
    b.build()
    with pytest.raises(CapabilityError, match="exposes only 2"):
        compile_for_device(b.flatten(), 3, Device(2))


# ── End-to-end via QarpEngine ───────────────────────────────────────────────


def _bell_block():
    b = SimpleBlock(n_qubits=3, name="bell")
    b.h(0)
    b.cx(0, 2)  # not adjacent on a 0-1-2 chain
    b.build()
    return b


def test_engine_routes_circuit_and_reindexes_counts():
    """A Bell-style circuit routed on a linear chain must, after the engine's
    automatic reindex, produce the same outcome distribution as the unrouted
    baseline (within sampling tolerance)."""
    from qarp.algorithms import Sampler

    b = _bell_block()
    n_shots = 6000

    # Unrouted baseline
    sampler_a = Sampler(ket=b, n_shots=n_shots)
    e_unrouted = QarpEngine(seed=42)
    e_unrouted.build([sampler_a])
    counts_unrouted = e_unrouted.run()[0]

    # Routed via linear chain
    sampler_b = Sampler(ket=b, n_shots=n_shots)
    e_routed = QarpEngine(
        device=Device(3, architecture=get_nearest_neighbour_architecture(3, 1)),
        seed=42,
    )
    e_routed.build([sampler_b])
    counts_routed = e_routed.run()[0]

    # Bell-like distribution has two dominant outcomes; both engines should
    # concentrate probability on the same two.  Compare their tops directly.
    top_a = sorted(counts_unrouted.items(), key=lambda kv: -kv[1])[:2]
    top_b = sorted(counts_routed.items(), key=lambda kv: -kv[1])[:2]
    assert {k for k, _ in top_a} == {k for k, _ in top_b}
    for (k_a, p_a), (k_b, p_b) in zip(top_a, top_b, strict=True):
        assert k_a == k_b
        assert abs(p_a - p_b) < 0.05


def test_engine_with_noise_takes_trajectory_path():
    """Adding any noise channel must visibly redistribute outcomes — a smoke
    that the trajectory path is hit (since the fast path would produce
    identical counts to the noiseless run)."""
    from qarp.algorithms import Sampler

    b = SimpleBlock(n_qubits=2, name="ghz2")
    b.h(0)
    b.cx(0, 1)
    b.build()

    # Heavy 2q depolarizing → mixed
    nm = NoiseModel.depolarizing(0.5, "2q")
    e = QarpEngine(device=Device(2, noise_model=nm.inner), seed=0)
    sampler = Sampler(ket=b, n_shots=4000)
    e.build([sampler])
    result = e.run()[0]
    # Bell pair under full depolarizing → close to uniform over 4 outcomes,
    # but strictly: no single bit string carries ~1.0 of the mass.
    assert max(result.values()) < 0.85


def test_engine_noise_disabled_falls_back_to_fast_path():
    """noise_model.enabled=False keeps the simulator on the fast path —
    the resulting distribution should equal the no-noise baseline exactly."""
    from qarp.algorithms import Sampler

    b = SimpleBlock(n_qubits=2, name="bell2")
    b.h(0)
    b.cx(0, 1)
    b.build()

    nm = NoiseModel.depolarizing(0.5, "2q")
    nm.enabled = False
    e_off = QarpEngine(device=Device(2, noise_model=nm.inner), seed=99)
    e_on = QarpEngine(seed=99)

    sampler_off = Sampler(ket=b, n_shots=2000)
    sampler_on = Sampler(ket=b, n_shots=2000)
    e_off.build([sampler_off])
    e_on.build([sampler_on])
    r_off = e_off.run()[0]
    r_on = e_on.run()[0]
    # Same seed + same circuit + noise disabled → identical counts.
    assert r_off == r_on


# ── Wide MCZ survives rebase to a hardware basis ────────────────────────────
#
# `MCZ` is in `native_gateset()`, so the default simulation path never
# decomposes it and the C++ gate tests exercise only its kernel.  Rebasing to a
# device basis that lacks `MCZ` is the path that ran the rule — and it silently
# lowered every width >= 4 to `CCZ(q0, q1, target)`, dropping all controls past
# the first two.  Oracle is the analytic MCZ matrix, built here in numpy.


def _clifford_t_rz_gateset():
    gs = qx.GateSet()
    gs.name = "clifford_t_rz"
    gs.allowed = {
        qx.GateType.H,
        qx.GateType.S,
        qx.GateType.Sdg,
        qx.GateType.T,
        qx.GateType.Tdg,
        qx.GateType.X,
        qx.GateType.Y,
        qx.GateType.Z,
        qx.GateType.CX,
        qx.GateType.Rx,
        qx.GateType.Ry,
        qx.GateType.Rz,
        qx.GateType.P,
        qx.GateType.GPhase,
        qx.GateType.Barrier,
    }
    return gs


@pytest.mark.parametrize("width", [2, 3, 4, 5, 6, 7])
def test_wide_mcz_rebase_is_unitary_exact(width):
    """`MCZ` lowered to a basis without it must still be diag(-1) on all-ones."""
    block = SimpleBlock(n_qubits=width, name="mcz")
    block.mcz(list(range(width)))
    block.build()

    device = Device(width, None, None, _clifford_t_rz_gateset(), False)
    lowered = compile_for_device(block, device)
    got = np.array(qx.QarpSimulator().unitary_matrix(lowered.commands, width))

    expected = np.eye(2**width, dtype=complex)
    expected[-1, -1] = -1.0
    # Exact, not up-to-phase: the block is controllable, so its global phase is
    # observable (§13/§18).
    assert np.abs(got - expected).max() < 1e-9


def test_wide_mcz_rebase_leaves_middle_control_zero_states_untouched():
    """Regression pin for the specific corruption: with q0=q1=target=1 but a
    middle control 0, the broken rule applied a spurious -1."""
    width = 4
    block = SimpleBlock(n_qubits=width, name="mcz")
    block.mcz(list(range(width)))
    block.build()

    device = Device(width, None, None, _clifford_t_rz_gateset(), False)
    got = np.array(
        qx.QarpSimulator().unitary_matrix(compile_for_device(block, device).commands, width)
    )
    # 0b1011: q0=1, q1=1, q2=0, q3(target)=1 — not all controls set.
    assert got[11, 11].real == pytest.approx(1.0, abs=1e-9)
    assert got[15, 15].real == pytest.approx(-1.0, abs=1e-9)


# ── Wide MCZ noise: variadic depolarizing channel ───────────────────────────


def test_wide_mcz_depolarizing_scales_z_by_analytic_factor():
    """Uniform k-qubit depolarizing scales every non-identity Pauli expectation
    by 1 − p·4^k/(4^k−1) (Pauli-twirl algebra: the non-identity strings sum to
    −Q for traceless Q).  On |000⟩ an MCZ is trivial and ⟨Z_i⟩ = 1, so the
    measured ⟨Z_i⟩ must land on the factor itself — qubit-symmetric, so the
    check is independent of bitstring endianness."""
    from qarp.algorithms import Sampler

    k, p, n_shots = 3, 0.3, 20000
    b = SimpleBlock(n_qubits=k, name="mcz_noise")
    b.mcz(0, 1, 2)
    b.build()

    nm = NoiseModel.depolarizing(p, ["MCZ"])
    e = QarpEngine(device=Device(k, noise_model=nm.inner), seed=7)
    sampler = Sampler(ket=b, n_shots=n_shots)
    e.build([sampler])
    result = e.run()[0]

    factor = 1.0 - p * 4**k / (4**k - 1)
    total = sum(result.values())
    for i in range(k):
        z_i = sum((1.0 if key[i] == 0 else -1.0) * w for key, w in result.items()) / total
        assert z_i == pytest.approx(factor, abs=0.03)


def test_noisy_controlled_block_over_mcz_runs():
    """Regression: ControlledBlock.flatten() emits one wide MCZ, and the old
    fixed-2q depolarizing channel threw mid-shot ('requires exactly 2 target
    qubits') as soon as it fired on a wider command."""
    from qarp.algorithms import Sampler
    from qarp.blocks import ControlledBlock

    inner = SimpleBlock(n_qubits=3, name="ccz")
    inner.mcz(0, 1, 2)
    ctrl = ControlledBlock(inner, num_controls=2, ctrl_state=[True, True])
    ctrl.build()

    # p = 1 guarantees the channel fires every shot — the regression cannot
    # hide behind a low error rate.
    nm = NoiseModel.depolarizing(1.0, ["MCZ", "CX"])
    e = QarpEngine(device=Device(5, noise_model=nm.inner), seed=1)
    sampler = Sampler(ket=ctrl, n_shots=100)
    e.build([sampler])
    result = e.run()[0]
    assert sum(result.values()) > 0


# ── Wide MCZ survives routing ───────────────────────────────────────────────


def test_wide_mcz_routes_on_nearest_neighbour_device():
    """The router takes 0/1/2-qubit gates only, and a gateset carrying MCZ
    kept it native through the pre-route rebase — so this combination threw
    ("gate 'MCZ' has 4 qubits") for every MCZ-native gateset.  The compiled
    circuit must reproduce MCZ under the router's own permutation."""
    width = 4
    block = SimpleBlock(n_qubits=width, name="mcz")
    block.mcz(list(range(width)))
    block.build()

    device = Device(
        width,
        get_nearest_neighbour_architecture(2, 2),
        None,
        qx.native_gateset(),  # contains MCZ: the case that must not throw
        False,
    )
    compiled = compile_for_device(block, device)
    assert all(len(c.qubits) <= 2 for c in compiled.commands)

    got = np.array(qx.QarpSimulator().unitary_matrix(compiled.commands, width))

    # The routed circuit ends with qubits relocated by the SWAP network, so
    # it is MCZ composed with a basis permutation.  MCZ commutes with any
    # qubit permutation, which pins the result exactly: a signed permutation
    # matrix whose single -1 sits on the all-ones state (permutation-fixed).
    # The broken rule put the -1 on other states, so this catches it without
    # depending on the router's layout bookkeeping.
    all_ones = 2**width - 1
    for col in range(2**width):
        nz = np.flatnonzero(np.abs(got[:, col]) > 1e-9)
        assert len(nz) == 1, f"column {col} is not a permutation column"
        amp = got[nz[0], col]
        want = -1.0 if col == all_ones else 1.0
        assert amp == pytest.approx(want, abs=1e-9), f"column {col} carries {amp}"
    assert np.abs(got[all_ones, all_ones] - (-1.0)) < 1e-9


# ── Perfect-layout pre-pass (bugfix/sabre-grid-placement) ───────────────────


def _grid_edges(rows: int, cols: int) -> list[tuple[int, int]]:
    edges = []
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                edges.append((q, q + 1))
            if r + 1 < rows:
                edges.append((q, q + cols))
    return edges


def _ring_block(n: int, order: list[int] | None = None) -> SimpleBlock:
    """CX ring; `order` relabels the logical qubits without changing the graph."""
    order = order or list(range(n))
    block = SimpleBlock(n)
    for q in range(n):
        block.cx(order[q], order[(q + 1) % n])
    block.build()
    return block


def _swaps(compiled) -> int:
    return sum(1 for c in compiled.commands if c.gate == qx.GateType.SWAP)


@pytest.mark.parametrize("rows,cols", [(2, 4), (3, 4), (2, 7), (4, 4), (3, 6), (4, 5)])
def test_ring_on_grid_routes_without_swaps(rows, cols):
    """A ring that fits the device exactly must cost nothing to route.

    Oracle is combinatorial, not the router's own output: a grid with an even
    number of cells has a Hamiltonian cycle, so an n-qubit ring embeds in it
    and zero SWAPs is optimal.
    """
    n = rows * cols
    device = Device(n, architecture=Architecture(n, _grid_edges(rows, cols)))
    assert _swaps(compile_for_device(_ring_block(n), device)) == 0


def test_routing_cost_is_invariant_under_relabelling():
    """Renaming logical qubits is not a change to the problem.

    This is the property the old identity-seeded mapping search failed: the
    same ring on a 3x4 grid cost between 0 and 10 SWAPs depending only on how
    the qubits happened to be numbered.
    """
    n = 12
    device = Device(n, architecture=Architecture(n, _grid_edges(3, 4)))
    rng = np.random.default_rng(20260824)
    baseline = _swaps(compile_for_device(_ring_block(n), device))
    for _ in range(8):
        order = list(rng.permutation(n))
        assert _swaps(compile_for_device(_ring_block(n, order), device)) == baseline


def test_unembeddable_circuit_still_routes():
    """All-to-all cannot fit a grid; the pre-pass must give up, not hang."""
    n = 9
    block = SimpleBlock(n)
    for a in range(n):
        for b in range(a + 1, n):
            block.cx(a, b)
    block.build()
    device = Device(n, architecture=Architecture(n, _grid_edges(3, 3)))
    compiled = compile_for_device(block, device)
    assert _swaps(compiled) > 0
    assert sorted(compiled.final_logical_to_physical) == list(range(n))


# ── Compile errors are CapabilityError (pipeline_hardening_plan.md P1.21) ─


def test_disconnected_architecture_rejects_with_capability_error():
    """A triangle of interactions cannot embed in a single edge plus an
    isolated qubit; the router needs a path that does not exist."""
    b = SimpleBlock(n_qubits=3, name="triangle")
    b.cx(0, 1).cx(1, 2).cx(0, 2)
    b.build()
    dev = Device(3, architecture=Architecture(3, [(0, 1)], "edge_plus_island"))
    with pytest.raises(CapabilityError, match="disconnected"):
        compile_for_device(b.flatten(), 3, dev)


def test_bad_initial_mapping_rejects_with_capability_error():
    opts = qx.RoutingOptions()
    opts.arch = qx.Architecture(2, [(0, 1)])
    opts.initial_mapping = [0, 0]
    with pytest.raises(CapabilityError, match="not a permutation"):
        qx.route([qx.Command(qx.GateType.H, 0)], opts)


def test_wide_gate_without_gate_set_rejects_with_capability_error():
    b = SimpleBlock(n_qubits=3, name="ccx")
    b.ccx(0, 1, 2)
    b.build()
    dev = Device(3, architecture=Architecture(3, [(0, 1), (1, 2)], "line"))
    with pytest.raises(CapabilityError, match="0/1/2-qubit"):
        compile_for_device(b.flatten(), 3, dev)


# ── Two layouts: initial placement and final map (§14) ──────────────────────
#
# A routed circuit reports where each logical qubit is *placed* before the
# first gate and where it *ends* after the last.  The oracle is the full
# unitary, not a |0…0⟩ column: every permutation fixes |0…0⟩, so a wrong
# initial map is invisible to any check that starts there.


def _perm_matrix(n: int, logical_to_physical) -> np.ndarray:
    """Basis permutation moving logical bit ``l`` to bit ``logical_to_physical[l]``."""
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=complex)
    for b in range(dim):
        moved = 0
        for l in range(n):
            if (b >> l) & 1:
                moved |= 1 << logical_to_physical[l]
        out[moved, b] = 1.0
    return out


def _unitary(commands, n: int) -> np.ndarray:
    return np.array(qx.QarpSimulator().unitary_matrix(commands, n))


def _layout_oracle_holds(reference, routed, initial, final, n: int) -> bool:
    """``P_final⁻¹ · U_routed · P_initial == U_reference`` modulo one global phase."""
    lhs = _perm_matrix(n, final).T @ _unitary(routed, n) @ _perm_matrix(n, initial)
    rhs = _unitary(reference, n)
    idx = np.unravel_index(np.argmax(np.abs(rhs)), rhs.shape)
    phase = lhs[idx] / rhs[idx]
    return abs(abs(phase) - 1.0) < 1e-9 and np.abs(lhs - phase * rhs).max() < 1e-9


def test_route_reports_pinned_initial_mapping_and_final_map_separately():
    """A pinned placement is echoed as the initial map; SWAPs move the final one."""
    block = SimpleBlock(3)
    block.h(0)
    block.cx(0, 2)
    block.build()
    opts = qx.RoutingOptions()
    opts.arch = Architecture(3, [(0, 1), (1, 2)])
    opts.router = qx.RouterKind.Lite
    # Pin logical 0 and 2 to the two ends of the line so the CX must ferry —
    # a pin that lands them adjacent routes with no SWAP and final == initial.
    opts.initial_mapping = [2, 1, 0]
    routed = qx.route(block.flatten(), opts)
    initial = list(routed.initial_logical_to_physical)
    final = list(routed.final_logical_to_physical)
    assert initial == [2, 1, 0]
    assert sorted(final) == [0, 1, 2]
    assert final != initial
    assert _layout_oracle_holds(block.flatten(), routed.commands, initial, final, 3)
    # Load-bearing: pretending the placement was the identity — all a
    # final-map-only consumer can do — breaks the unitary identity.
    assert not _layout_oracle_holds(block.flatten(), routed.commands, [0, 1, 2], final, 3)


def test_compile_for_device_initial_placement_satisfies_unitary_oracle():
    """SABRE's own placement, on an all-pairs circuit it places non-trivially."""
    n = 5
    block = SimpleBlock(n)
    for a in range(n):
        for b in range(a + 1, n):
            block.cx(a, b)
    block.build()
    line = Architecture(n, [(i, i + 1) for i in range(n - 1)])
    compiled = compile_for_device(block, Device(n, architecture=line))
    initial = list(compiled.initial_logical_to_physical)
    final = list(compiled.final_logical_to_physical)
    assert sorted(initial) == list(range(n))
    assert sorted(final) == list(range(n))
    assert _layout_oracle_holds(block.flatten(), compiled.commands, initial, final, n)


def test_unrouted_device_reports_identity_for_both_maps():
    block = SimpleBlock(2)
    block.cx(0, 1)
    block.build()
    compiled = compile_for_device(block, Device(2))
    assert list(compiled.initial_logical_to_physical) == [0, 1]
    assert list(compiled.final_logical_to_physical) == [0, 1]
