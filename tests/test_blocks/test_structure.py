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
from qarp._types import Consumes
from qarp.algorithms import PauliAveraging, Sampler, StateVector
from qarp.blocks import (
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
from tests.conftest import _StubEngine


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


def test_engines_without_the_path_offer_no_structured_run():
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1)
    assert _StubEngine().prepare_structured(block, Sampler()) is None
    assert QarpEngine(device=Device(4)).prepare_structured(block, Sampler()) is None
    assert QarpEngine(structured=False).prepare_structured(block, Sampler()) is None
    amplitudes = StateVector(operator=QubitOperator("Z0"), ket=block)
    assert QarpEngine().prepare_structured(block, amplitudes) is None
    # Counts, but of derived circuits: the run would hand it the bare block.
    averaging = PauliAveraging(operator=QubitOperator("X0"), n_shots=qarp.EXACT)
    assert averaging.consumes is Consumes.COUNTS
    assert QarpEngine().prepare_structured(block, averaging) is None
    assert QarpEngine().prepare_structured(block, _CountsWithoutTheFlag()) is None
    assert not block.is_built


class _CountsWithoutTheFlag:
    """A runnable from before ``samples_block``: counts, no declaration."""

    consumes = Consumes.COUNTS
    n_shots = None
    initial_state = None
    supports_exact = True


def test_a_structured_run_samples_the_declared_block():
    # QPE on P(2πφ) from |1⟩ reads φ = 3/8 exactly (analytic), below the
    # planner's register minimum and without building the block.
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1)
    sampler = Sampler(n_shots=qarp.EXACT, measured_qubits=[0, 1, 2])
    run = QarpEngine().prepare_structured(block, sampler)
    assert run is not None and not block.is_built
    dist = run.sample()
    assert dict(dist) == pytest.approx({(1, 1, 0): 1.0}, abs=1e-10)
    assert sampler.result is dist

    # The primitive marginalises as on the ordinary path: no measured_qubits
    # on an unbuilt sampler reads the whole register.
    whole = Sampler(n_shots=qarp.EXACT)
    dist = QarpEngine().prepare_structured(block, whole).sample()
    assert dict(dist) == pytest.approx({(1, 1, 0, 1): 1.0}, abs=1e-10)


def test_a_built_declaring_block_through_a_primitive_plans_its_ladder():
    engine = QarpEngine(structured=True)
    sampler = Sampler(QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 11, 1))
    engine.build([sampler])
    program = engine._programs[id(sampler)][0]
    assert program is not None and "controlled_powers" in program.kinds()
