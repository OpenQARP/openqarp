"""Primitives derive their circuits from unbuilt inputs (§14 *Structured
execution*): ``build()`` builds nothing, the engine plans what it is handed
and builds only what planning refuses.

Oracles are analytic: uniform amplitudes of ``H⊗H``, ``⟨+|X|+⟩ = 1`` and
``⟨+|Z|+⟩ = 0``, a state's overlap with itself.
"""

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


@pytest.mark.parametrize("name", list(_cases()))
def test_build_leaves_the_inputs_unbuilt(name):
    prim, inputs = _cases()[name]
    prim.build()
    assert prim.sub_blocks
    assert not any(block.is_built for block in inputs)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Sampler", {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25}),
        ("PauliAveraging", 1.0),
        ("HadamardTest", 1.0),
        ("SWAPTest", 1.0),
        ("MirrorTest", 1.0),
        ("StateVector", 0.0),
    ],
)
def test_the_engine_builds_what_the_primitive_left_unbuilt(name, expected):
    prim, inputs = _cases()[name]
    engine = QarpEngine(n_shots=qarp.EXACT)
    engine.build([prim])
    result = engine.run()[0]
    if isinstance(expected, dict):
        assert dict(result) == pytest.approx(expected, abs=1e-12)
    else:
        assert result == pytest.approx(expected, abs=1e-10)
