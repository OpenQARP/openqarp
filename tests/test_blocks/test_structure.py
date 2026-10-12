"""The ``structure`` protocol (§13) and the two readers of a declaration
(§14 *Structured execution*): the planner on a built block, and
``plan_structure`` on one that is never built.

Oracles: ``unitary_matrix()`` (the §14 oracle path, never planned); the
block's own ``flatten()`` is the contract check of a declaration, not the
oracle."""

import inspect

import numpy as np
import pytest

import qarp
import qarp.blocks as qb
import qarpx as qx
from qarp import _program
from qarp.algorithms import Sampler, StateVector
from qarp.blocks import (
    CompositeBlock,
    ComputationalBasisStateBlock,
    ControlledBlock,
    DOSQPEBlock,
    HnBlock,
    ModularMultiplicationBlock,
    QFTBlock,
    QPEBlock,
    Repeat,
    SimpleBlock,
)
from qarp.devices import Device
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator


def _random_state(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    psi = rng.standard_normal(1 << n) + 1j * rng.standard_normal(1 << n)
    return psi / np.linalg.norm(psi)


def _phase_u(phi: float = 0.375) -> SimpleBlock:
    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * phi)
    return u


class _Asym(SimpleBlock):
    """Two qubits, no symmetry, a global phase: a U that is not a permutation."""

    def build_vanilla(self):
        self.h(0)
        self.cx(0, 1)
        self.ry(1, 0.37)
        self.rz(0, 0.21)
        self.t(1)
        self.gphase(0.45)


# Every block class overriding structure(), with small unmeasured instances
# (unitary_matrix() is the oracle, and it has no column for a measurement).
DECLARING = {
    QPEBlock: [
        lambda: QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1, measure=False),
        lambda: QPEBlock(ComputationalBasisStateBlock([0, 1]), _Asym(2), 4, 2, measure=False),
    ],
    DOSQPEBlock: [
        lambda: DOSQPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1, measure=False),
        lambda: DOSQPEBlock(ComputationalBasisStateBlock([1, 0]), _Asym(2), 3, 2, measure=False),
    ],
}


def _overrides_structure(cls) -> bool:
    method = getattr(cls, "structure", None)
    return method is not None and not getattr(method, "_qarp_default", False)


def test_every_declaring_block_class_is_registered():
    declaring = {
        cls
        for _, cls in inspect.getmembers(qb, inspect.isclass)
        if issubclass(cls, (SimpleBlock, qb.CompositeBlockBase)) and _overrides_structure(cls)
    }
    assert declaring == set(DECLARING)


def _expanded(parts) -> list:
    out: list = []
    for part in parts:
        if isinstance(part, Repeat):
            out.extend(list(part.block.flatten()) * part.count)
        else:
            out.extend(part.flatten())
    return out


@pytest.mark.parametrize("factory", [f for fs in DECLARING.values() for f in fs])
def test_the_parts_expanded_are_the_built_blocks_gate_stream(factory):
    block = factory()
    parts = block.structure()
    assert parts is not None and not block.is_built
    assert _expanded(parts) == list(block.build().flatten())


@pytest.mark.parametrize("factory", [f for fs in DECLARING.values() for f in fs])
def test_an_unbuilt_declaration_lowers_to_its_unitary(factory):
    block = factory()
    n = block.n_qubits
    plan = _program.plan_structure(block, n)
    assert plan is not None and not block.is_built
    assert "controlled_powers" in plan.kinds()
    psi = _random_state(n, 1)
    out = qx.QarpSimulator().program_statevector(plan.program(), n, initial_state=psi)
    np.testing.assert_allclose(
        np.asarray(out), np.asarray(factory().build().unitary_matrix()) @ psi, atol=1e-12
    )


def test_a_built_declaring_block_plans_from_its_declaration(monkeypatch):
    # Twelve qubits: the planner runs on the built block, and the declaration
    # gives the same kernels as the unbuilt plan without any ladder detection.
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 11, 1)
    unbuilt = _program.plan_structure(block, block.n_qubits)
    calls = []
    steps = _program._Planner._ladder_step

    def counted(self, kid, kid_map, *rest):
        calls.append(kid)
        return steps(self, kid, kid_map, *rest)

    monkeypatch.setattr(_program._Planner, "_ladder_step", counted)
    built = _program.plan(block.build(), block.n_qubits)
    assert built is not None and built.kinds() == unbuilt.kinds()
    assert len(calls) <= 2 * block.n_ancilla


def _ladder_step(control: int, inner, targets) -> ControlledBlock:
    cu = ControlledBlock(inner, num_controls=1, ctrl_state=[True])
    cu.build()
    cu.target_qubits = [control, *targets]
    return cu


class _Lies(qb.CompositeBlockBase):
    """Declares one more application than it builds."""

    def __init__(self, cu):
        self._cu = cu
        super().__init__(n_qubits=5)

    def structure(self):
        return [Repeat(self._cu, 4)]

    def build_vanilla(self):
        for _ in range(3):
            self.add_child(self._cu)


def test_a_declaration_that_does_not_add_up_is_ignored(monkeypatch):
    # A built composite cannot change, so the only drift is a wrong
    # declaration; its exponent would read 4 where the gates apply U three
    # times, and the planner must plan from the gates instead.
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)
    block = _Lies(_ladder_step(0, _Asym(2).build(), [3, 4])).build()
    psi = _random_state(5, 2)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )


class _AsymOther(_Asym):
    """The same gate count as ``_Asym`` with other angles."""

    def build_vanilla(self):
        self.h(0)
        self.cx(0, 1)
        self.ry(1, 0.91)
        self.rz(0, 0.64)
        self.t(1)
        self.gphase(0.12)


class _LiesInContent(qb.CompositeBlockBase):
    """Declares a ladder over one U and builds one of equal length over another."""

    def __init__(self, declared, used):
        self._declared = declared
        self._used = used
        super().__init__(n_qubits=5)

    def structure(self):
        return [Repeat(self._declared, 4)]

    def build_vanilla(self):
        for _ in range(4):
            self.add_child(self._used)


def test_a_declaration_of_the_right_length_but_other_gates_is_ignored(monkeypatch):
    # Same gate count, so only the digest of the parts tells the declaration
    # from the gates; planned from the declaration this gave wrong amplitudes.
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)
    declared = _ladder_step(0, _Asym(2).build(), [3, 4])
    used = _ladder_step(0, _AsymOther(2).build(), [3, 4])
    assert len(declared.flatten()) == len(used.flatten())
    block = _LiesInContent(declared, used).build()
    psi = _random_state(5, 4)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )


def test_a_declaration_naming_an_unbuilt_part_is_ignored(monkeypatch):
    monkeypatch.setattr(_program, "MIN_QUBITS", 0)
    unbuilt = ControlledBlock(_Asym(2), num_controls=1, ctrl_state=[True])
    used = _ladder_step(0, _Asym(2).build(), [3, 4])
    block = _LiesInContent(unbuilt, used).build()
    assert not unbuilt.is_built
    psi = _random_state(5, 5)
    np.testing.assert_allclose(
        block.statevector(psi, structured=True),
        np.asarray(block.unitary_matrix()) @ psi,
        atol=1e-12,
    )


@pytest.mark.parametrize("cls", [QPEBlock, DOSQPEBlock])
def test_a_shared_eigenstate_block_is_placed_as_a_copy(cls):
    # Two blocks over one state block: the caller's object keeps its placement
    # and the first block's unitary is the one built from an unshared state.
    shared = ComputationalBasisStateBlock([1])
    alone = cls(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1, measure=False).build()
    first = cls(shared, _phase_u(), 3, 1, measure=False).build()
    second = cls(shared, _phase_u(), 4, 1, measure=False).build()
    assert shared.target_qubits is None or list(shared.target_qubits) == [0]
    np.testing.assert_allclose(
        np.asarray(first.unitary_matrix()), np.asarray(alone.unitary_matrix()), atol=1e-12
    )
    assert second.n_qubits == alone.n_qubits + 1


def test_repeat_refuses_a_count_below_one():
    with pytest.raises(ValueError, match="count"):
        Repeat(_phase_u().build(), 0)
    with pytest.raises(ValueError, match="count"):
        Repeat(_phase_u().build(), True)


class _Declares(qb.CompositeBlockBase):
    """A test block whose declaration is whatever it was given."""

    def __init__(self, parts, n_qubits):
        self._parts = parts
        super().__init__(n_qubits=n_qubits)

    def structure(self):
        return self._parts

    def build_vanilla(self):
        for part in self._parts:
            for _ in range(part.count if isinstance(part, Repeat) else 1):
                self.add_child(part.block if isinstance(part, Repeat) else part)


def test_an_unbuilt_plan_refuses_what_it_cannot_lower():
    u = _Asym(2).build()
    ladder = [Repeat(_ladder_step(a, u, [3, 4]), 2**a) for a in range(3)]
    assert _program.plan_structure(_Declares(ladder, 5), 5) is not None

    # A repeat that is not a ladder step.
    plain = SimpleBlock(2, target_qubits=[3, 4])
    plain.h(0).cx(0, 1).ry(1, 0.3)
    assert _program.plan_structure(_Declares([Repeat(plain.build(), 3)], 5), 5) is None

    # A parametric part.
    from sympy import Symbol

    rot = SimpleBlock(1, target_qubits=[0])
    rot.ry(0, Symbol("phi"))
    assert _program.plan_structure(_Declares([rot.build(), *ladder], 5), 5) is None

    # A recorded measurement before the last part.
    meas = SimpleBlock(1, target_qubits=[0])
    meas.measure(0, 0)
    assert _program.plan_structure(_Declares([meas.build(), *ladder], 5), 5) is None
    tail = SimpleBlock(1, target_qubits=[0])
    tail.measure(0, 0)
    assert _program.plan_structure(_Declares([*ladder, tail.build()], 5), 5) is not None


def _order_finding_declared(n_counting: int = 8) -> _Declares:
    """Order finding for a = 2 mod 15 with one C-M(2) per counting qubit,
    repeated 2^i times; the work register (|1⟩) follows the counting one."""
    counting = list(range(n_counting))
    work = list(range(n_counting, n_counting + 4))
    one = SimpleBlock(4, target_qubits=work, name="PrepareWorkOne")
    one.x(0)
    mult = ModularMultiplicationBlock(2, 15).build()
    iqft = QFTBlock(n_counting).dagger().build()
    iqft.target_qubits = counting
    parts = [
        HnBlock(n_counting, target_qubits=counting).build(),
        one.build(),
        *(Repeat(_ladder_step(c, mult, work), 2**i) for i, c in enumerate(counting)),
        iqft,
    ]
    return _Declares(parts, n_counting + 4)


def _counting_marginal(psi, n_counting: int) -> np.ndarray:
    probs = np.abs(np.asarray(psi)) ** 2
    return probs.reshape(1 << 4, 1 << n_counting).sum(axis=0)


# a = 2 mod 15 has order 4, which divides 2^8: the counting marginal is exactly
# 1/4 at 0, 64, 128 and 192 (analytic).
_ORDER_FOUR = np.zeros(256)
_ORDER_FOUR[[0, 64, 128, 192]] = 0.25


def test_a_repeated_controlled_permutation_lowers_unbuilt_to_one_table():
    block = _order_finding_declared()
    plan = _program.plan_structure(block, 12)
    assert plan is not None and not block.is_built
    kinds = plan.program().kinds()
    assert "permutation" in kinds and "controlled_powers" not in kinds
    zero = np.zeros(1 << 12, dtype=complex)
    zero[0] = 1.0
    out = qx.QarpSimulator().program_statevector(plan.program(), 12, initial_state=zero)
    np.testing.assert_allclose(_counting_marginal(out, 8), _ORDER_FOUR, atol=1e-12)


def test_a_repeated_controlled_permutation_is_lifted_once_per_entry_when_built(monkeypatch):
    lifts = []
    single = _program._Planner._single

    def counted(self, kid, span, qmap, *rest):
        lifts.append(kid)
        return single(self, kid, span, qmap, *rest)

    monkeypatch.setattr(_program._Planner, "_single", counted)
    block = _order_finding_declared().build()
    plan = _program.plan(block, 12)
    assert plan is not None and "controlled_powers" not in plan.program().kinds()
    # One lift per repeated entry (255 copies in all); Repeat(·, 1) walks as a plain part.
    assert len(lifts) == sum(1 for p in block.structure() if isinstance(p, Repeat) and p.count > 1)
    np.testing.assert_allclose(
        _counting_marginal(block.statevector(structured=True), 8), _ORDER_FOUR, atol=1e-12
    )


def test_the_table_power_is_the_repeated_composition():
    table = np.random.default_rng(6).permutation(64).astype(np.int64)
    for count in (1, 2, 13, 32):
        naive = np.arange(64, dtype=np.int64)
        for _ in range(count):
            naive = table[naive]
        np.testing.assert_array_equal(_program._power(table, count), naive)


def test_a_repeat_naming_an_unbuilt_block_refuses_the_unbuilt_plan():
    unbuilt = ControlledBlock(_Asym(2), num_controls=1, ctrl_state=[True])
    assert _program.plan_structure(_Declares([Repeat(unbuilt, 3)], 5), 5) is None


def test_plan_structure_descends_an_unbuilt_composite():
    """A declaring child of an unbuilt composite is lowered from its
    declaration without being built; the plain tail is built on its own.
    Oracle: ``unitary_matrix()`` of a separately built copy."""

    def tree():
        qpe = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1, measure=False)
        return qpe, CompositeBlock([qpe, HnBlock(4)], n_qubits=4)

    qpe, composite = tree()
    found = _program.plan_structure(composite, 4)
    assert found is not None and found.kinds().count("controlled_powers") == 1
    assert not qpe.is_built and not composite.is_built
    psi = np.asarray(qx.QarpSimulator().program_statevector(found.program(), 4))
    expected = np.asarray(tree()[1].build().unitary_matrix())[:, 0]
    np.testing.assert_allclose(psi, expected, atol=1e-10)


def test_engines_without_the_path_build_the_block():
    def builds(engine, prim):
        engine.build([prim])
        return prim.sub_blocks[0].is_built and engine._programs[id(prim)] == [None]

    for engine in (QarpEngine(device=Device(4)), QarpEngine(structured=False)):
        block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1)
        assert builds(engine, Sampler(block))
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1, measure=False)
    assert builds(QarpEngine(), StateVector(operator=QubitOperator("Z0"), ket=block))


def test_the_engine_samples_a_declared_block_without_building_it():
    # QPE on P(2πφ) from |1⟩ reads φ = 3/8 exactly (analytic), below the
    # planner's register minimum and without building the block.
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1)
    sampler = Sampler(block, n_shots=qarp.EXACT, measured_qubits=[0, 1, 2])
    engine = QarpEngine()
    engine.build([sampler])
    assert not block.is_built
    assert engine._programs[id(sampler)][0].kinds().count("controlled_powers") == 1
    dist = engine.run()[0]
    assert dict(dist) == pytest.approx({(1, 1, 0): 1.0}, abs=1e-10)
    assert sampler.result is dist

    # The primitive marginalises as on the ordinary path: no measured_qubits
    # on a sampler reads the whole register.
    whole = Sampler(
        QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1), n_shots=qarp.EXACT
    )
    engine.build([whole])
    assert dict(engine.run()[0]) == pytest.approx({(1, 1, 0, 1): 1.0}, abs=1e-10)


def test_a_built_declaring_block_through_a_primitive_plans_its_ladder():
    engine = QarpEngine(structured=True)
    sampler = Sampler(QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 11, 1))
    engine.build([sampler])
    program = engine._programs[id(sampler)][0]
    assert program is not None and "controlled_powers" in program.kinds()


class _FlipAll(CompositeBlock):
    """Populates itself at build, as a composite subclass may."""

    def __init__(self, n: int):
        super().__init__([], n_qubits=n, name="FlipAll")

    def build_vanilla(self):
        flip = SimpleBlock(self.n_qubits, name="X")
        for q in range(self.n_qubits):
            flip.x(q)
        self.add_child(flip.build())
        super().build_vanilla()


@pytest.mark.parametrize("structured", [True, False])
def test_a_composite_populated_at_build_is_a_plain_part(structured):
    # QPE reads φ = 3/8 as ancillas (1, 1, 0) beside the eigenstate |1⟩
    # (analytic); flipping every qubit then gives (0, 0, 1, 0).
    qpe = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1, measure=False)
    readout = SimpleBlock(4, name="m")
    readout.measure([(q, q) for q in range(4)])
    tree = CompositeBlock([qpe, _FlipAll(4), readout], n_qubits=4)
    sampler = Sampler(tree, n_shots=qarp.EXACT)
    engine = QarpEngine(structured=structured)
    engine.build([sampler])
    assert dict(engine.run()[0]) == pytest.approx({(0, 0, 1, 0): 1.0}, abs=1e-10)


class _QPEThenFlip(QPEBlock):
    """Inherits the declaration but builds one more gate than it names."""

    def build_vanilla(self):
        super().build_vanilla()
        flip = SimpleBlock(1, target_qubits=[self.n_ancilla], name="X")
        flip.x(0)
        self.add_child(flip.build())


@pytest.mark.parametrize("built", [False, True])
def test_an_inherited_declaration_is_not_trusted_over_another_build(built):
    # The ancillas read φ = 3/8 as (1, 1, 0); the extra X turns the
    # eigenstate |1⟩ into |0⟩ after the kickback (analytic).
    block = _QPEThenFlip(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1, measure=False)
    if built:
        block.build()
    sampler = Sampler(block, n_shots=qarp.EXACT)
    engine = QarpEngine(structured=True)
    engine.build([sampler])
    assert dict(engine.run()[0]) == pytest.approx({(1, 1, 0, 0): 1.0}, abs=1e-10)


def test_a_built_declaring_block_is_planned_from_its_stream():
    # Built, the block is planned from its tree, its declaration checked by
    # digest against the stream it was built with.
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 11, 1).build()
    sampler = Sampler(block)
    engine = QarpEngine(structured=True)
    engine.build([sampler])
    assert sampler.compiled_circuits[0]
    assert "controlled_powers" in engine._programs[id(sampler)][0].kinds()


def test_a_lowered_circuit_sweeps_from_an_initial_state():
    # From |1⟩ on the state qubit the preparation's X leaves |0⟩, an eigenstate
    # of P with phase 0: every shot reads all zeros (analytic).
    psi = np.zeros(16, dtype=complex)
    psi[0b1000] = 1.0
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1)
    sampler = Sampler(block, n_shots=64, initial_state=psi)
    engine = QarpEngine(seed=3)
    (per_set,) = engine.batch_run([sampler], [{}])
    assert not block.is_built
    assert dict(per_set[0]) == {(0, 0, 0, 0): 1.0}
