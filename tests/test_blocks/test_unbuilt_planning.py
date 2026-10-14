"""``plan_structure`` on a block that is never built (§14): daggered parts,
nested placements, who may declare, and classical bits.

Oracles: analytic states and distributions; the built stream
(``structured=False``, or the block's own ``flatten()`` run directly) is a
cross-check of the same answer, never the oracle on its own."""

import math
from typing import Mapping, cast

import numpy as np
import pytest

import qarp
import qarpx as qx
from qarp import _program
from qarp.algorithms import MirrorTest, Sampler
from qarp.blocks import (
    CompositeBlock,
    ComputationalBasisStateBlock,
    ControlledBlock,
    QPEBlock,
    Repeat,
    SimpleBlock,
)
from qarp.engines import QarpEngine


def _increment(n: int) -> SimpleBlock:
    """|x⟩ → |x+1 mod 2^n⟩, LSB on qubit 0: a permutation, not self-inverse."""
    block = SimpleBlock(n, name="inc")
    for top in range(n - 1, 0, -1):
        block.mcx(*range(top + 1))
    block.x(0)
    return block


def _phase_u(phi: float = 0.375) -> SimpleBlock:
    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * phi)
    return u


def _sampled(block, **engine_kwargs) -> tuple[dict, list]:
    sampler = Sampler(block, n_shots=qarp.EXACT)
    engine = QarpEngine(**engine_kwargs)
    engine.build([sampler])
    return dict(cast(Mapping, engine.run()[0])), engine.kernels(sampler)[0]


# ── B1: a daggered part is planned as its dagger ───────────────────────────


def test_a_run_of_daggered_controlled_rotations_rotates_backwards():
    """Four C(RY(θ))† on a control in |1⟩ and a target in |+⟩ rotate by −4θ:
    P(target = 0) = (1 + 1/√2)/2 for θ = π/16 (analytic); the forward
    kernel would give (1 − 1/√2)/2."""
    theta = math.pi / 16

    def circuit():
        prep = SimpleBlock(2)
        prep.x(0)
        prep.h(1)
        parts = [prep]
        for _ in range(4):
            ry = SimpleBlock(1)
            ry.ry(0, theta)
            parts.append(ControlledBlock(ry, 1, target_qubits=[0, 1]).dagger())
        return CompositeBlock(parts, n_qubits=2)

    expected = (1 + 1 / math.sqrt(2)) / 2
    planned, kernels = _sampled(circuit())
    assert "controlled_powers" in kernels
    assert planned[(1, 0)] == pytest.approx(expected, abs=1e-10)
    built, _ = _sampled(circuit(), structured=False)
    assert built[(1, 0)] == pytest.approx(expected, abs=1e-10)


def test_a_daggered_controlled_permutation_is_inverted():
    """MirrorTest of |q1 = 1⟩ against C(increment) active on |0⟩: the bra's
    dagger decrements q1..q3 from 1 to 0, so the overlap is exactly 1; the
    forward table would read 0."""

    def primitive():
        ket = SimpleBlock(12)
        ket.x(1)
        bra = ControlledBlock(_increment(3), 1, ctrl_state=[False])
        return MirrorTest(bra=bra, ket=ket, n_shots=qarp.EXACT)

    for structured, expected_kernel in ((True, "permutation"), (False, "gates")):
        mirror = primitive()
        engine = QarpEngine(structured=structured)
        engine.build([mirror])
        assert expected_kernel in engine.kernels(mirror)[0]
        assert engine.run()[0] == pytest.approx(1.0, abs=1e-10)


def test_a_daggered_composite_is_walked_through_its_reversed_children():
    """(C-INC on q0..q3, RY(θ)^⊗8 on q4..q11)† from |0⟩: the rotations put
    each of q4..q11 in |1⟩ with probability sin²(θ/2) independently, then the
    decrement (control q0 in |0⟩) sets q1..q3 (analytic product distribution)."""
    theta = 0.7

    def circuit():
        cinc = ControlledBlock(_increment(3), 1, ctrl_state=[False])
        layer = SimpleBlock(8, target_qubits=list(range(4, 12)))
        for q in range(8):
            layer.ry(q, theta)
        inner = CompositeBlock([cinc, layer], n_qubits=12).dagger()
        return CompositeBlock([inner], n_qubits=12)

    p1 = math.sin(theta / 2) ** 2
    expected = {}
    for bits in range(1 << 8):
        tail = tuple((bits >> q) & 1 for q in range(8))
        expected[(0, 1, 1, 1, *tail)] = math.prod(p1 if b else 1 - p1 for b in tail)
    planned, kernels = _sampled(circuit())
    assert kernels != ["gates"]
    assert planned == pytest.approx(expected, abs=1e-10)
    built, _ = _sampled(circuit(), structured=False)
    assert built == pytest.approx(expected, abs=1e-10)


# ── B3: a nested Repeat keeps its placement ────────────────────────────────


class _XLayerThrice(SimpleBlock):
    """X on each of its qubits, three times, declared as one repeated layer."""

    def build_vanilla(self):
        for _ in range(3):
            for q in range(self.n_qubits):
                self.x(q)

    def structure(self):
        layer = SimpleBlock(self.n_qubits)
        for q in range(self.n_qubits):
            layer.x(q)
        return [Repeat(layer.build(), 3)]


def test_a_nested_repeat_is_placed_in_the_root_frame():
    """The declaring block sits on qubits 2..4 of a 5-qubit composite: X³ = X
    on each sets exactly those (analytic), never qubits 0..2."""

    def circuit():
        return CompositeBlock([_XLayerThrice(3, target_qubits=[2, 3, 4])], n_qubits=5)

    planned, kernels = _sampled(circuit())
    assert "permutation" in kernels
    assert planned == pytest.approx({(0, 0, 1, 1, 1): 1.0}, abs=1e-12)
    built, _ = _sampled(circuit(), structured=False)
    assert built == pytest.approx({(0, 0, 1, 1, 1): 1.0}, abs=1e-12)


# ── B4: an unbuilt declaration is read only from the class that builds ─────


class _Annotated(CompositeBlock):
    """Builds its children; declares something else."""

    def structure(self):
        x = SimpleBlock(1)
        x.x(0)
        return [Repeat(x.build(), 1)]


def test_a_declaration_under_an_inherited_build_is_not_trusted_unbuilt():
    """``CompositeBlock.build_vanilla`` wires the children and never reads
    ``structure()``, so the empty child is what runs: |0⟩ on both paths."""
    assert _program.plan_structure(_Annotated([SimpleBlock(1)], n_qubits=1), 1) is None
    for structured in (True, False):
        dist, _ = _sampled(_Annotated([SimpleBlock(1)], n_qubits=1), structured=structured)
        assert dist == pytest.approx({(0,): 1.0}, abs=1e-12)


def test_a_declaration_from_the_building_class_is_still_lowered():
    """Control: a class defining both methods is read unbuilt."""
    block = _XLayerThrice(3)
    assert _program.plan_structure(block, 3) is not None
    assert not block.is_built


# ── B5: planning declines when the built stream would move cbits ───────────


def _built_cbits(block) -> list[int]:
    built = block.build()
    result = qx.QarpSimulator().run(list(built.flatten()), built.n_qubits, 1, 1)
    return [int(b) for b in result.cbit_history[0]]


def _planned_cbits(plan, n_qubits: int) -> list[int]:
    result = qx.QarpSimulator().program_run(plan.program(), n_qubits, 1, 1)
    return [int(b) for b in result.cbit_history[0]]


def _qpe_three_eighths() -> QPEBlock:
    # Phase 3/8 = 0.011₂ lands bits (1, 1, 0) in cbits 0, 1, 2 (analytic).
    return QPEBlock(ComputationalBasisStateBlock([1]), _phase_u(), 3, 1)


def test_a_root_relabelling_its_cbits_is_not_planned():
    """``target_cbits = [2, 1, 0]`` on the root reverses QPE's readout to
    (0, 1, 1); unbuilt commands carry QPE's own cbits, so the root builds."""

    def circuit():
        root = CompositeBlock([_qpe_three_eighths()], n_qubits=4)
        root.target_cbits = [2, 1, 0]
        return root

    assert _program.plan_structure(circuit(), 4) is None
    assert _built_cbits(circuit()) == [0, 1, 1]


def test_a_cbit_owner_before_another_part_is_not_planned():
    """A part declaring two cbits shifts QPE's readout from cbits 0..2 to
    2..4 in the built stream."""

    def circuit():
        first = SimpleBlock(4)
        first.n_cbits = 2
        return CompositeBlock([first, _qpe_three_eighths()], n_qubits=4)

    assert _program.plan_structure(circuit(), 4) is None
    assert _built_cbits(circuit()) == [0, 0, 1, 1, 0]


def test_a_nested_composite_relabelling_its_cbits_is_not_planned():
    def circuit():
        inner = CompositeBlock([_qpe_three_eighths()], n_qubits=4)
        inner.target_cbits = [2, 1, 0]
        return CompositeBlock([inner], n_qubits=4)

    assert _program.plan_structure(circuit(), 4) is None
    assert _built_cbits(circuit()) == [0, 1, 1]


def test_a_nested_cbit_owner_before_the_readout_is_not_planned():
    def circuit():
        first = SimpleBlock(4)
        first.n_cbits = 2
        inner = CompositeBlock([first], n_qubits=4)
        inner.n_cbits = 2
        return CompositeBlock([inner, _qpe_three_eighths()], n_qubits=4)

    assert _program.plan_structure(circuit(), 4) is None
    assert _built_cbits(circuit()) == [0, 0, 1, 1, 0]


def test_a_cbit_owner_ending_the_composition_is_planned():
    """Control: the last part's cbits start at 0 in the built stream too, so
    QPE's readout (1, 1, 0) is planned without building the root."""

    def circuit():
        prep = SimpleBlock(5)
        prep.x(4)
        return CompositeBlock([prep, _qpe_three_eighths()], n_qubits=5)

    plan = _program.plan_structure(circuit(), 5)
    assert plan is not None and "controlled_powers" in plan.kinds()
    assert _planned_cbits(plan, 5) == [1, 1, 0]
    assert _built_cbits(circuit()) == [1, 1, 0]
