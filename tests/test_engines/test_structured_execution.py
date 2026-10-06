"""Structured execution (§14): block structure run as typed kernels.

Oracles: ``unitary_matrix()`` (the §14 oracle path, never planned) and the
analytic images of basis states under permutation blocks."""

import numpy as np
import pytest
from sympy import Symbol

import qarp
import qarpx as qx
from qarp import _program
from qarp.algorithms import PauliAveraging, Sampler, StateVector
from qarp.blocks import (
    CompositeBlock,
    ConditionalBlock,
    ControlledBlock,
    ModularMultiplicationBlock,
    OrderFindingBlock,
    ResetBlock,
    SimpleBlock,
)
from qarp.devices import Device
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator


def _random_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    return psi / np.linalg.norm(psi)


class _Increment(SimpleBlock):
    """x → x + 1 mod 2^n on its register, as an mcx ladder."""

    def build_vanilla(self):
        for t in reversed(range(1, self.n_qubits)):
            self.mcx(*range(t), t)
        self.x(0)


class _Mix(SimpleBlock):
    def build_vanilla(self):
        self.h(0)
        self.cx(0, 1)
        self.ry(1, 0.3)
        self.rz(0, 0.2)


class _Rotations(SimpleBlock):
    def build_vanilla(self):
        for q in range(self.n_qubits):
            self.ry(q, 0.1 + 0.2 * q)


def _mixed_tree(n: int) -> CompositeBlock:
    parts = []
    for _ in range(2):
        parts.append(_Increment(4, target_qubits=[0, 2, 4, 6]))
        parts.append(
            ControlledBlock(ModularMultiplicationBlock(2, 5), 1, target_qubits=[n - 1, 1, 3, 5])
        )
        parts.append(_Mix(2, target_qubits=[n - 2, 7]))
    return CompositeBlock(parts, n_qubits=n)


def _increment_image(v: int, qubits: list[int]) -> int:
    work = sum(((v >> q) & 1) << b for b, q in enumerate(qubits))
    work = (work + 1) % (1 << len(qubits))
    for b, q in enumerate(qubits):
        v = (v & ~(1 << q)) | (((work >> b) & 1) << q)
    return v


# ── Block.statevector ───────────────────────────────────────────────────────


def test_program_matches_the_unitary_on_a_mixed_tree(monkeypatch):
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)
    n = 10
    block = _mixed_tree(n).build()
    kinds = _program.plan(block, n).program().kinds()
    assert {"permutation", "dense"} <= set(kinds)
    psi = _random_state(n, 1)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )


def test_a_permutation_tree_is_one_kernel():
    n = 12
    block = CompositeBlock(
        [_Increment(5, target_qubits=[0, 3, 6, 9, 11]), _Increment(4, target_qubits=[1, 2, 4, 5])]
        * 3,
        n_qubits=n,
    ).build()
    assert _program.plan(block, n).program().kinds() == ["permutation"]
    x = 0b101101100101
    psi = np.zeros(1 << n, dtype=complex)
    psi[x] = 1.0
    image = x
    for _ in range(3):
        image = _increment_image(image, [0, 3, 6, 9, 11])
        image = _increment_image(image, [1, 2, 4, 5])
    out = block.statevector(psi, structured=True)
    assert out[image] == 1.0 and np.count_nonzero(out) == 1


def test_no_structure_runs_the_gate_path_bit_for_bit():
    n = 12
    block = _Rotations(n).build()
    assert _program.plan(block, n) is None
    psi = _random_state(n, 2)
    np.testing.assert_array_equal(
        block.statevector(psi, structured=True), block.statevector(psi, structured=False)
    )


def test_narrow_registers_are_never_planned():
    block = _Increment(_program.MIN_QUBITS - 1).build()
    assert _program.plan(block, block.n_qubits) is None


class _MeasureThenFlip(SimpleBlock):
    def build_vanilla(self):
        self.measure(0, 0)
        self.x(0)


def test_a_mid_circuit_measurement_keeps_the_gate_path(monkeypatch):
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)
    block = CompositeBlock([_Increment(4), _MeasureThenFlip(4), _Increment(4)], n_qubits=4).build()
    assert _program.plan(block, 4) is None


def _measure_then_condition(n: int) -> CompositeBlock:
    """A recorded measurement feeding a conditioned X on the same qubit."""
    prelude = SimpleBlock(1, target_qubits=[0])
    prelude.measure(0, 0)
    body = SimpleBlock(1, target_qubits=[0])
    body.x(0)
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=body.build())
    cond.build()
    cond.target_cbits = [0]
    return CompositeBlock(
        [_Increment(5, target_qubits=[0, 1, 2, 3, 4])] * 2 + [prelude, cond], n_qubits=n
    )


@pytest.mark.parametrize("tail", ["reset", "condition"])
def test_a_reset_or_a_classical_condition_keeps_the_gate_path(tail):
    # Both need per-shot trajectories, which only the gate path runs; the
    # permutation before them must not be planned around them.
    n = 12
    if tail == "reset":
        ket = CompositeBlock(
            [_Increment(5, target_qubits=[0, 1, 2, 3, 4])] * 2 + [ResetBlock(0)], n_qubits=n
        )
    else:
        ket = _measure_then_condition(n)
    assert _program.plan(ket.build(), n) is None
    engine = QarpEngine(seed=5, n_shots=100, structured=True)
    sampler = Sampler(ket)
    engine.build([sampler])
    assert _programs(engine, sampler) == [None]
    assert sum(engine.run()[0].probabilities) == pytest.approx(1.0)


def test_structured_default_follows_the_environment(monkeypatch):
    for raw, expected in [("0", False), ("false", False), ("1", True), ("junk", True)]:
        monkeypatch.setenv("QARP_STRUCTURED", raw)
        _program.structured_default.cache_clear()
        assert _program.resolve(None) is expected
    monkeypatch.delenv("QARP_STRUCTURED")
    _program.structured_default.cache_clear()
    assert _program.resolve(None) is True
    assert _program.resolve(False) is False


# ── QarpEngine ──────────────────────────────────────────────────────────────


def _programs(engine, prim):
    return engine._programs[id(prim)]


@pytest.mark.parametrize("n_shots", [qarp.EXACT, 256])
def test_engine_samples_a_permuted_basis_state_deterministically(n_shots):
    n = 12
    qubits = [0, 3, 6, 9, 11]
    ket = CompositeBlock([_Increment(5, target_qubits=qubits)] * 2, n_qubits=n)
    x = 0b010011010001
    psi = np.zeros(1 << n, dtype=complex)
    psi[x] = 1.0
    sampler = Sampler(ket, n_shots=n_shots, initial_state=psi)
    engine = QarpEngine(seed=7, structured=True)
    engine.build([sampler])
    assert _programs(engine, sampler)[0] is not None
    dist = engine.run()[0]
    image = _increment_image(_increment_image(x, qubits), qubits)
    assert dist.outcomes.tolist() == [image]
    assert dist.probabilities.tolist() == [1.0]


def test_engine_exact_distribution_matches_the_unitary():
    n = 12
    ket = CompositeBlock(
        [_Rotations(3, target_qubits=[0, 1, 2]), _Increment(6, target_qubits=[0, 1, 2, 3, 4, 5])],
        n_qubits=n,
    )
    engine = QarpEngine(structured=True)
    sampler = Sampler(ket, n_shots=qarp.EXACT)
    engine.build([sampler])
    assert _programs(engine, sampler)[0] is not None
    dist = engine.run()[0]
    small = CompositeBlock(
        [_Rotations(3, target_qubits=[0, 1, 2]), _Increment(6, target_qubits=[0, 1, 2, 3, 4, 5])],
        n_qubits=6,
    ).build()
    probs6 = np.abs(np.asarray(small.unitary_matrix())[:, 0]) ** 2
    expected = {k: p for k, p in enumerate(probs6) if p > 1e-12}
    assert dict(
        zip(dist.outcomes.tolist(), dist.probabilities.tolist(), strict=True)
    ) == pytest.approx(expected, abs=1e-12)


def _bound_ket(angle, n: int = 12) -> CompositeBlock:
    class _Param(SimpleBlock):
        def build_vanilla(self):
            self.ry(0, angle)
            self.cx(0, 1)
            self.ry(1, angle)

    return CompositeBlock(
        [_Param(2, target_qubits=[0, 1]), _Increment(5, target_qubits=[0, 1, 2, 3, 4])] * 2,
        n_qubits=n,
    )


def _born(angle: float) -> np.ndarray:
    """|U|0⟩|² of the bound ket, from its unitary on the five qubits it touches."""
    return np.abs(np.asarray(_bound_ket(angle, n=5).build().unitary_matrix())[:, 0]) ** 2


def _dense(dist, size: int) -> np.ndarray:
    out = np.zeros(size)
    out[dist.outcomes] = dist.probabilities
    return out


def test_engine_binds_parameters_into_program_gates():
    theta = Symbol("theta")
    engine = QarpEngine(structured=True)
    sampler = Sampler(_bound_ket(theta), n_shots=qarp.EXACT)
    engine.build([sampler])
    assert "permutation" in _programs(engine, sampler)[0].kinds()
    for angle in (0.7, 1.9):
        dist = engine.run({theta: angle})[0]
        np.testing.assert_allclose(_dense(dist, 1 << 12)[:32], _born(angle), atol=1e-12)


def test_batch_run_exact_readout_uses_the_programs():
    theta = Symbol("theta")
    engine = QarpEngine(structured=True)
    sampler = Sampler(_bound_ket(theta), n_shots=qarp.EXACT)
    angles = [0.3, 1.2]
    swept = engine.batch_run([sampler], [{theta: a} for a in angles])
    assert "permutation" in _programs(engine, sampler)[0].kinds()
    for angle, results in zip(angles, swept, strict=True):
        np.testing.assert_allclose(_dense(results[0], 1 << 12)[:32], _born(angle), atol=1e-12)
    # rebuild=False keeps the programs of the first build.
    again = engine.batch_run([sampler], [{theta: 0.5}], rebuild=False)
    assert "permutation" in _programs(engine, sampler)[0].kinds()
    np.testing.assert_allclose(_dense(again[0][0], 1 << 12)[:32], _born(0.5), atol=1e-12)


class _Rotate(SimpleBlock):
    def build_vanilla(self):
        self.ry(0, Symbol("phi"))


@pytest.mark.parametrize("method", ["parameter-shift", "finite-diff"])
def test_gradients_through_a_planned_circuit_match_the_analytic_value(method):
    # Ry(φ) on qubit 0, then a classical shift carries it to qubit 1:
    # ⟨Z₁⟩ = cos φ, so the gradient is −sin φ.
    n = 12
    shift = SimpleBlock(n)
    shift.swap(0, 1).swap(2, 3).swap(3, 2)
    ket = CompositeBlock([_Rotate(1, target_qubits=[0]), shift], n_qubits=n)
    estimator = PauliAveraging(ket=ket, operator=QubitOperator("Z1"), n_shots=qarp.EXACT)
    engine = QarpEngine(structured=True)
    engine.build([estimator])
    assert any(p is not None for p in _programs(engine, estimator))
    phi = 0.8
    assert engine.run({Symbol("phi"): phi})[0] == pytest.approx(np.cos(phi), abs=1e-12)
    grad = engine.run_gradient({Symbol("phi"): phi}, method=method)[0]
    assert grad[0] == pytest.approx(-np.sin(phi), abs=1e-6)


def test_the_oracle_and_the_adjoint_gradient_never_plan(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("planned")

    n = 12
    block = CompositeBlock([_Increment(5, target_qubits=[0, 1, 2, 3, 4])] * 2, n_qubits=n).build()
    ket = CompositeBlock(
        [_Rotate(1, target_qubits=[0]), _Increment(5, target_qubits=[1, 2, 3, 4, 5])], n_qubits=n
    )
    estimator = StateVector(operator=QubitOperator("Z0"), ket=ket)
    engine = QarpEngine(structured=True)
    monkeypatch.setattr(_program, "plan", refuse)
    block.unitary_matrix()
    engine.build([estimator])
    grad = engine.run_gradient({Symbol("phi"): 0.8}, method="adjoint")[0]
    assert grad[0] == pytest.approx(-np.sin(0.8), abs=1e-10)


def test_engine_keeps_the_gate_path_with_a_device_or_amplitude_primitive():
    n = 12
    ket = CompositeBlock([_Increment(5, target_qubits=[0, 1, 2, 3, 4])] * 2, n_qubits=n)
    with_device = QarpEngine(device=Device(n), structured=True)
    sampler = Sampler(ket, n_shots=qarp.EXACT)
    with_device.build([sampler])
    assert _programs(with_device, sampler) == [None]
    engine = QarpEngine(structured=True)
    amplitudes = StateVector(operator=QubitOperator("Z0"), ket=ket)
    engine.build([amplitudes])
    assert _programs(engine, amplitudes) == [None]


class _Collide(SimpleBlock):
    """A 3-qubit site collision: not a permutation."""

    def build_vanilla(self):
        self.h(0)
        self.cx(0, 1)
        self.cry(1, 2, 0.4)
        self.cx(2, 0)
        self.ry(1, 0.25)


def test_collision_sites_run_as_shared_dense_kernels(monkeypatch):
    # Streaming shifts between identical per-site collisions: one permutation
    # per shift, one dense kernel per site, and the sites share one matrix.
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)
    n = 9
    sites = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    parts = [_Increment(3, target_qubits=[0, 3, 6])]
    parts += [_Collide(3, target_qubits=site) for site in sites]
    parts.append(_Increment(3, target_qubits=[2, 5, 8]))
    block = CompositeBlock(parts, n_qubits=n).build()
    plan = _program.plan(block, n)
    dense = [k for k in plan.kernels if isinstance(k, _program.Dense)]
    assert len(dense) == len(sites)
    assert all(k.matrix is dense[0].matrix for k in dense)
    assert plan.program().kinds().count("permutation") == 2
    psi = _random_state(n, 3)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )


def test_fusion_width_follows_the_simulator():
    # §14 Simulation fusion: blocks of fusion_max_qubits from fusion_min_qubits
    # qubits, the single-qubit pass below that, nothing when off.
    sim = qx.QarpSimulator()
    sim.fusion_max_qubits = 3
    sim.fusion_min_qubits = 12
    assert _program.fusion_width_of(sim, 12) == 3
    assert _program.fusion_width_of(sim, 11) == 1
    sim.fusion_max_qubits = 1
    assert _program.fusion_width_of(sim, 12) == 1
    sim.fusion_max_qubits = 0
    assert _program.fusion_width_of(sim, 12) == 0


def _two_qubit_terms(n: int) -> CompositeBlock:
    return CompositeBlock([_Mix(2, target_qubits=[q, q + 1]) for q in range(n - 1)], n_qubits=n)


def test_a_dense_kernel_no_wider_than_the_fusion_width_stays_gates():
    # Fusion covers a span of its own width or less, merging it with its
    # neighbours and with its repeats; a dense kernel would fence it off.
    n = 12
    terms = _two_qubit_terms(n).build()
    assert _program.plan(terms, n, fusion_width=3) is None
    assert _program.plan(terms, n, fusion_width=2) is None
    assert set(_program.plan(terms, n, fusion_width=1).kinds()) == {"dense"}
    sites = CompositeBlock(
        [_Collide(3, target_qubits=[q, q + 1, q + 2]) for q in range(0, n, 3)] * 5, n_qubits=n
    ).build()
    assert _program.plan(sites, n, fusion_width=3) is None
    assert _program.plan(sites, n, fusion_width=2).kinds() == ["dense"] * 20
    psi = _random_state(n, 9)
    for block in (terms, sites):
        np.testing.assert_array_equal(
            block.statevector(psi, structured=True), block.statevector(psi, structured=False)
        )


def test_the_engine_plans_at_its_simulator_fusion_width():
    n = 12
    engine = QarpEngine(structured=True)
    sampler = Sampler(_two_qubit_terms(n), n_shots=qarp.EXACT)
    engine.build([sampler])
    assert _programs(engine, sampler) == [None]
    engine._sim.fusion_max_qubits = 1
    engine.build([sampler])
    assert set(_programs(engine, sampler)[0].kinds()) == {"dense"}


def test_order_finding_reads_its_order_through_permutation_kernels():
    # a = 2 mod 15 has order 4, which divides the counting register's 2^8, so
    # the counting marginal is exactly 1/4 at 0, 64, 128 and 192 (analytic).
    block = OrderFindingBlock(2, 15).build()
    n = block.n_qubits
    assert n == 12 and "permutation" in _program.plan(block, n).program().kinds()
    probs = np.abs(block.statevector(structured=True)) ** 2
    counting = probs.reshape(1 << block.n_work_qubits, 1 << block.n_counting_qubits).sum(axis=0)
    expected = np.zeros(1 << block.n_counting_qubits)
    expected[[0, 64, 128, 192]] = 0.25
    np.testing.assert_allclose(counting, expected, atol=1e-12)


class _MeasureAndReuse(SimpleBlock):
    def build_vanilla(self):
        self.measure(0, 0)
        self.h(0)
        self.measure(0, 1)


def test_a_measurement_reused_in_the_last_kernel_keeps_the_gate_path():
    # Terminal measurements are sampled once; a reused measured qubit needs
    # per-shot trajectories, which only the gate path runs.
    n = 12
    ket = CompositeBlock(
        [_Increment(6, target_qubits=[0, 1, 2, 3, 4, 5])] * 2 + [_MeasureAndReuse(1)], n_qubits=n
    )
    assert _program.plan(ket.build(), n) is None
    engine = QarpEngine(seed=3, n_shots=200, structured=True)
    sampler = Sampler(ket)
    engine.build([sampler])
    assert _programs(engine, sampler) == [None]
    assert sum(engine.run()[0].probabilities) == pytest.approx(1.0)


def _mix_then_increment(n: int, unrecorded_measure: bool) -> CompositeBlock:
    mix = SimpleBlock(2, target_qubits=[n - 2, n - 1])
    mix.h(0).cx(0, 1).ry(1, 0.3).rz(0, 0.2)
    if unrecorded_measure:
        mix.set_commands([*mix.commands(), qx.Command(qx.GateType.Measure, 0)])
    return CompositeBlock([mix, _Increment(5, target_qubits=[0, 1, 2, 3, 4])], n_qubits=n).build()


def test_an_unrecorded_measurement_keeps_its_span_as_gates(monkeypatch):
    # A Measure without a cbit leaves the state alone, so the oracle is the
    # unitary of the same tree without it.
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)
    n = 8
    block = _mix_then_increment(n, unrecorded_measure=True)
    assert _program.plan(block, n).program().kinds() == ["gates", "permutation"]
    psi = _random_state(n, 8)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(_mix_then_increment(n, unrecorded_measure=False).unitary_matrix()) @ psi,
        atol=1e-12,
    )


def test_a_planned_block_deep_copies_without_its_program():
    import copy

    n = 12
    block = CompositeBlock([_Increment(5, target_qubits=[0, 1, 2, 3, 4])] * 2, n_qubits=n).build()
    psi = _random_state(n, 4)
    expected = block.statevector(psi, structured=True)
    clone = copy.deepcopy(block)
    np.testing.assert_array_equal(clone.statevector(psi, structured=True), expected)


# ── Placement through the tree ──────────────────────────────────────────────


class _Asym(SimpleBlock):
    """Three qubits with no symmetry between them; not a permutation."""

    def build_vanilla(self):
        self.h(0)
        self.cx(0, 2)
        self.ry(1, 0.37)
        self.cry(2, 1, 0.9)
        self.rz(0, 0.21)
        self.t(2)
        self.cx(1, 0)
        self.rx(2, 1.1)
        self.gphase(0.45)


def _spread(n: int) -> _Rotations:
    return _Rotations(n, target_qubits=list(range(n)))


def _assert_matches_the_unitary(block, kind: str) -> list[str]:
    block.build()
    n = block.n_qubits
    plan = _program.plan(block, n)
    assert plan is not None
    kinds = plan.program().kinds()
    assert kind in kinds
    psi = _random_state(n, 5)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )
    return kinds


@pytest.fixture
def plan_small_registers(monkeypatch):
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)


def test_a_placed_composite_places_its_children(plan_small_registers):
    # Two placed levels, each too wide for one dense kernel, above a declared
    # and a controlled permutation: both must land through both placements.
    inner = CompositeBlock(
        [
            _spread(9),
            ModularMultiplicationBlock(7, 15, target_qubits=[6, 0, 3, 8]),
            ControlledBlock(_Increment(3, target_qubits=[2, 0, 1]), 1, target_qubits=[4, 7, 1, 5]),
        ],
        n_qubits=9,
    )
    inner.target_qubits = [7, 2, 9, 0, 5, 3, 8, 1, 6]
    outer = CompositeBlock([inner, _Increment(2, target_qubits=[3, 1])], n_qubits=10)
    outer.target_qubits = [4, 8, 1, 6, 0, 10, 3, 5, 2, 7]
    root = CompositeBlock([_spread(11), outer], n_qubits=11)
    kinds = _assert_matches_the_unitary(root, "permutation")
    assert "dense" not in kinds


def test_a_controlled_permutation_places_its_inner_block(plan_small_registers):
    # §6.1 with two controls, a mixed control state and a placed inner block,
    # once derived from gates and once declared.
    derived = ControlledBlock(
        _Increment(4, target_qubits=[3, 1, 0, 2]),
        2,
        ctrl_state=[True, False],
        target_qubits=[7, 2, 0, 5, 1, 6],
    )
    declared = ControlledBlock(
        ModularMultiplicationBlock(7, 15, target_qubits=[2, 0, 3, 1]),
        2,
        ctrl_state=[False, True],
        target_qubits=[1, 6, 0, 4, 2, 5],
    )
    for controlled in (derived, declared):
        tree = CompositeBlock([_spread(10), controlled], n_qubits=10)
        _assert_matches_the_unitary(tree, "permutation")


# ── Controlled-power ladders ────────────────────────────────────────────────


def _ladder(steps, inner, ctrl_state=True) -> CompositeBlock:
    kids = [_spread(10)]
    for control, targets in steps:
        kids.append(
            ControlledBlock(inner(), 1, ctrl_state=[ctrl_state], target_qubits=[control, *targets])
        )
    return CompositeBlock(kids, n_qubits=10)


_TARGETS = [6, 3, 5]


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(
            [(0, _TARGETS)] + [(1, _TARGETS)] * 2 + [(2, _TARGETS)] * 4, id="powers of two"
        ),
        pytest.param([(0, _TARGETS), (1, _TARGETS)] * 3 + [(1, _TARGETS)] * 2, id="interleaved"),
    ],
)
def test_a_ladder_of_one_controlled_block_is_one_kernel(plan_small_registers, steps):
    kinds = _assert_matches_the_unitary(_ladder(steps, lambda: _Asym(3)), "controlled_powers")
    assert kinds.count("controlled_powers") == 1


def test_a_ladder_places_its_inner_block(plan_small_registers):
    steps = [(0, _TARGETS)] * 6 + [(1, _TARGETS)] * 6
    _assert_matches_the_unitary(
        _ladder(steps, lambda: _Asym(3, target_qubits=[2, 0, 1])), "controlled_powers"
    )


def test_ladders_on_different_targets_do_not_merge(plan_small_registers):
    steps = [(0, [6, 3, 5])] * 6 + [(0, [3, 6, 5])] * 6
    kinds = _assert_matches_the_unitary(_ladder(steps, lambda: _Asym(3)), "controlled_powers")
    assert kinds.count("controlled_powers") == 2


def test_a_ladder_too_short_to_pay_for_its_kernel_stays_gates(plan_small_registers):
    # One controlled phase per step: three steps on three controls are three
    # gates against six passes, seven steps are seven against six.
    def phase():
        u = SimpleBlock(1)
        u.p(0, 0.7)
        return u

    short = _ladder([(0, [6]), (1, [6]), (2, [6])], phase).build()
    assert _program.plan(short, short.n_qubits) is None
    long = _ladder([(0, [6])] + [(1, [6])] * 2 + [(2, [6])] * 4, phase)
    assert _assert_matches_the_unitary(long, "controlled_powers").count("controlled_powers") == 1


def test_a_ladder_controlled_on_zero_matches_the_unitary(plan_small_registers):
    block = _ladder([(0, _TARGETS)] * 8, lambda: _Asym(3), ctrl_state=False).build()
    psi = _random_state(10, 6)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )


# ── The cached program follows the block ────────────────────────────────────


def test_a_block_changed_after_planning_is_planned_again():
    n = 12
    block = SimpleBlock(n)
    for t in reversed(range(1, 6)):
        block.mcx(*range(t), t)
    block.build()
    psi = _random_state(n, 7)
    block.statevector(psi, structured=True)
    block.x(7)
    block.swap(2, 9)
    block.x(0)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )


# ── Kernels through the bindings, above the parallel threshold ──────────────


def _offsets(qubits: list[int]) -> np.ndarray:
    local = np.arange(1 << len(qubits))
    out = np.zeros_like(local)
    for b, q in enumerate(qubits):
        out |= ((local >> b) & 1) << q
    return out


def _apply_local(psi, matrix, qubits, control=None):
    """``matrix`` on ``qubits`` wherever ``control`` reads 1, by explicit gather."""
    mask = int(sum(1 << q for q in qubits))
    base = np.arange(psi.size)
    base = base[(base & mask) == 0]
    if control is not None:
        base = base[(base >> control) & 1 == 1]
    cols = base[:, None] | _offsets(qubits)[None, :]
    out = psi.copy()
    out[cols] = psi[cols] @ matrix.T
    return out


def _haar(k: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((1 << k, 1 << k)) + 1j * rng.standard_normal((1 << k, 1 << k))
    return np.linalg.qr(a)[0]


_WIDE = 17  # qarp forks its OpenMP team from 16 qubits


def test_permutation_kernel_matches_numpy_indexing():
    rng = np.random.default_rng(11)
    psi = _random_state(_WIDE, 12)
    sim = qx.QarpSimulator()
    for qubits in ([16, 3, 11, 0, 7], list(range(_WIDE))):
        table = rng.permutation(1 << len(qubits)).astype(np.int64)
        program = qx.Program()
        program.add_permutation(qubits, table)
        index = np.arange(psi.size)
        local = sum(((index >> q) & 1) << b for b, q in enumerate(qubits))
        kept = index & ~int(sum(1 << q for q in qubits))
        expected = np.empty_like(psi)
        expected[kept | _offsets(qubits)[table[local]]] = psi
        np.testing.assert_array_equal(
            sim.program_statevector(program, _WIDE, initial_state=psi), expected
        )


def test_dense_and_controlled_powers_kernels_match_numpy():
    psi = _random_state(_WIDE, 13)
    sim = qx.QarpSimulator()
    u = _haar(3, 14)
    program = qx.Program()
    program.add_dense([15, 2, 8], u)
    np.testing.assert_allclose(
        sim.program_statevector(program, _WIDE, initial_state=psi),
        _apply_local(psi, u, [15, 2, 8]),
        atol=1e-12,
    )
    v = _haar(2, 15)
    program = qx.Program()
    program.add_controlled_powers([0, 13], [3, 2], [4, 16], v)
    expected = _apply_local(psi, np.linalg.matrix_power(v, 3), [4, 16], control=0)
    expected = _apply_local(expected, np.linalg.matrix_power(v, 2), [4, 16], control=13)
    np.testing.assert_allclose(
        sim.program_statevector(program, _WIDE, initial_state=psi), expected, atol=1e-12
    )


def test_invalid_kernels_are_refused_at_the_binding():
    program = qx.Program()
    with pytest.raises(ValueError, match="bijection"):
        program.add_permutation([0, 1], np.array([0, 0, 1, 2], dtype=np.int64))
    with pytest.raises(ValueError, match=">= 0"):
        program.add_permutation([0, 1], np.array([0, -1, 1, 2], dtype=np.int64))
    program.add_permutation([5], np.array([1, 0], dtype=np.int64))
    with pytest.raises(ValueError, match="qubit 5"):
        qx.QarpSimulator().program_statevector(program, 3)
