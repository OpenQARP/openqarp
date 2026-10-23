"""Primitives derive their circuits from unbuilt inputs (§14 *Structured
execution*): ``build()`` builds nothing, the engine plans what it is handed
and builds only what planning refuses.

Oracles are analytic, on ``|ψ_a⟩ = RY(a)|0⟩ ⊗ |0⟩``, which is no eigenstate of
the operators used, so a dropped or misplaced operator changes the value:
``⟨X0⟩ = sin a``, ``⟨Z0⟩ = cos a``, ``⟨ψ_b|ψ_a⟩ = cos((a − b)/2)``,
``⟨ψ_b|Z0|ψ_a⟩ = cos((a + b)/2)``, and the uniform amplitudes of ``H⊗H``.
"""

import math

import numpy as np
import pytest

import qarp
from qarp.algorithms import (
    BasisRotationAveraging,
    GroupedTransitionHadamardTest,
    HadamardTest,
    InterferometricTest,
    MirrorTest,
    PauliAveraging,
    Sampler,
    StateVector,
    SWAPTest,
    TermwiseHadamardTest,
    TermwiseSWAPTest,
)
from qarp.blocks import ComputationalBasisStateBlock, HnBlock, SimpleBlock, XnBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator


def _x_on_0() -> SimpleBlock:
    u = SimpleBlock(2, name="X0")
    u.x(0)
    return u


def _two_bras():
    return [HnBlock(2), XnBlock(2)]


_A, _B = 0.7, 0.2


def _ry(angle: float) -> SimpleBlock:
    block = SimpleBlock(2, name="RY0")
    block.ry(0, angle)
    return block


# name -> (primitive over unbuilt inputs, the inputs that must stay unbuilt)
def _cases() -> dict:
    ket, bra, op = HnBlock(2), HnBlock(2), _x_on_0()
    bras = _two_bras()
    basis = ComputationalBasisStateBlock([1, 0])
    cases = {
        "Sampler": (Sampler(ket=ket), [ket]),
        "StateVector": (StateVector(ket=ket, operator=QubitOperator("Z0")), [ket]),
        "StateVector overlap": (StateVector(ket=ket, bra=bra), [ket, bra]),
        "PauliAveraging": (PauliAveraging(ket=ket, operator=QubitOperator("X0")), [ket]),
        "HadamardTest": (HadamardTest(ket=ket, operator=op), [ket, op]),
        "SWAPTest": (SWAPTest(bra=bra, ket=ket, operator=op), [bra, ket, op]),
        "MirrorTest": (MirrorTest(bra=bra, ket=ket, operator=op), [bra, ket, op]),
        "TermwiseHadamardTest": (
            TermwiseHadamardTest(ket=ket, operator=QubitOperator("X0") + QubitOperator("Z1")),
            [ket],
        ),
        "TermwiseSWAPTest": (TermwiseSWAPTest(bra=bras, ket=ket), [*bras, ket]),
        "GroupedTransitionHadamardTest": (
            GroupedTransitionHadamardTest(bra=bra, ket=ket, operator=QubitOperator("Z0")),
            [bra, ket],
        ),
        "InterferometricTest": (InterferometricTest(ket=basis, operator=op), [basis, op]),
        "BasisRotationAveraging": (
            BasisRotationAveraging(
                ket=ket, integrals=(np.array([[0.5]]), np.full((1, 1, 1, 1), 0.2))
            ),
            [ket],
        ),
    }
    return cases


_CASES = (
    "Sampler",
    "StateVector",
    "StateVector overlap",
    "PauliAveraging",
    "HadamardTest",
    "SWAPTest",
    "MirrorTest",
    "TermwiseHadamardTest",
    "TermwiseSWAPTest",
    "GroupedTransitionHadamardTest",
    "InterferometricTest",
    "BasisRotationAveraging",
)


def test_every_case_is_listed():
    assert set(_cases()) == set(_CASES)


@pytest.mark.parametrize("name", _CASES)
def test_build_leaves_the_inputs_unbuilt(name):
    prim, inputs = _cases()[name]
    prim.build()
    assert prim.sub_blocks
    assert not any(block.is_built for block in inputs)


def _x0() -> SimpleBlock:
    block = SimpleBlock(2, name="X0 bra")
    block.x(0)
    return block


# name -> (primitive over unbuilt RY inputs, its analytic value)
def _evaluated(name: str):
    a, b = _A, _B
    if name == "Sampler":
        return Sampler(ket=HnBlock(2)), {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25}
    if name == "StateVector":
        return StateVector(ket=_ry(a), operator=QubitOperator("Z0")), math.cos(a)
    if name == "StateVector overlap":
        return StateVector(ket=_ry(a), bra=_ry(b)), math.cos((a - b) / 2)
    if name == "PauliAveraging":
        return PauliAveraging(ket=_ry(a), operator=QubitOperator("X0")), math.sin(a)
    if name == "HadamardTest":
        return HadamardTest(ket=_ry(a), operator=_x_on_0()), math.sin(a)
    if name == "SWAPTest":
        return SWAPTest(bra=_ry(a), ket=_ry(a), operator=_x_on_0()), math.sin(a) ** 2
    if name == "MirrorTest":
        return MirrorTest(bra=_ry(a), ket=_ry(a), operator=_x_on_0()), math.sin(a) ** 2
    if name == "TermwiseHadamardTest":
        op = QubitOperator("X0") + QubitOperator("Z1")
        return TermwiseHadamardTest(ket=_ry(a), operator=op), math.sin(a) + 1.0
    if name == "TermwiseSWAPTest":
        # Σ_i |⟨bra_i|ψ_a⟩|²: ⟨ψ_b|ψ_a⟩ and ⟨1,0|ψ_a⟩ = sin(a/2).
        expected = math.cos((a - b) / 2) ** 2 + math.sin(a / 2) ** 2
        return TermwiseSWAPTest(bra=[_ry(b), _x0()], ket=_ry(a)), expected
    if name == "GroupedTransitionHadamardTest":
        prim = GroupedTransitionHadamardTest(bra=_ry(b), ket=_ry(a), operator=QubitOperator("Z0"))
        return prim, math.cos((a + b) / 2)
    if name == "InterferometricTest":
        # The test reads ⟨ψ|U|ψ⟩ against the reference |00⟩, which U must fix:
        # P(a) does, and ⟨1|P(a)|1⟩ = exp(i a).
        u = SimpleBlock(2, name="P0")
        u.p(0, a)
        interferometric = InterferometricTest(ket=ComputationalBasisStateBlock([1, 0]), operator=u)
        return interferometric, complex(math.cos(a), math.sin(a))
    assert name == "BasisRotationAveraging"
    # One spatial orbital, h = 0.5, (00|00) = 0.2: H = 0.5 (n↑ + n↓) + 0.2 n↑n↓,
    # and RY(a)|00⟩ occupies spin orbital 0 with probability sin²(a/2).
    integrals = (np.array([[0.5]]), np.full((1, 1, 1, 1), 0.2))
    return BasisRotationAveraging(ket=_ry(a), integrals=integrals), 0.5 * math.sin(a / 2) ** 2


@pytest.mark.parametrize(
    "name",
    [
        "Sampler",
        "StateVector",
        "StateVector overlap",
        "PauliAveraging",
        "HadamardTest",
        "SWAPTest",
        "MirrorTest",
        "TermwiseHadamardTest",
        "TermwiseSWAPTest",
        "GroupedTransitionHadamardTest",
        "InterferometricTest",
        "BasisRotationAveraging",
    ],
)
def test_the_engine_builds_what_the_primitive_left_unbuilt(name):
    prim, expected = _evaluated(name)
    engine = QarpEngine(n_shots=qarp.EXACT)
    engine.build([prim])
    result = engine.run()[0]
    if isinstance(expected, dict):
        assert dict(result) == pytest.approx(expected, abs=1e-12)
    else:
        assert complex(result) == pytest.approx(expected, abs=1e-10)
