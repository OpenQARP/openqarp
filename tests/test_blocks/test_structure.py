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
    ComputationalBasisStateBlock,
    ControlledBlock,
    DOSQPEBlock,
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

    def counted(self, kid, kid_map):
        calls.append(kid)
        return steps(self, kid, kid_map)

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


def test_engines_without_the_path_offer_no_structured_run():
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1)
    assert _StubEngine().prepare_structured(block, Sampler()) is None
    assert QarpEngine(device=Device(4)).prepare_structured(block, Sampler()) is None
    assert QarpEngine(structured=False).prepare_structured(block, Sampler()) is None
    amplitudes = StateVector(operator=QubitOperator("Z0"), ket=block)
    assert QarpEngine().prepare_structured(block, amplitudes) is None
    assert not block.is_built


def test_a_structured_run_samples_the_declared_block():
    # QPE on P(2πφ) from |1⟩ reads φ = 3/8 exactly (analytic), below the
    # planner's register minimum and without building the block.
    block = QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1)
    sampler = Sampler(n_shots=qarp.EXACT, measured_qubits=[0, 1, 2])
    run = QarpEngine().prepare_structured(block, sampler)
    assert run is not None and not block.is_built
    dist = run.sample()
    assert dict(dist) == pytest.approx({(1, 1, 0): 1.0}, abs=1e-10)


def test_a_built_declaring_block_through_a_primitive_plans_its_ladder():
    engine = QarpEngine(structured=True)
    sampler = Sampler(QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 11, 1))
    engine.build([sampler])
    program = engine._programs[id(sampler)][0]
    assert program is not None and "controlled_powers" in program.kinds()
