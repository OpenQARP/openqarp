"""Tests for the InterferometricTest primitive.

The primitive inherits from ``PrimitiveAlgorithm`` and exposes
``sub_blocks`` plus a list-based ``run(results)`` signature returning the
post-processed expectation directly.  The constructor / validation contract
is tested below; the physical-result test exercises the full build+run
pipeline through ``QarpEngine``.
"""

import pytest

from qarp.algorithms import InterferometricTest, MirrorTest, SWAPTest, Target
from qarp.blocks import (
    ComputationalBasisStateBlock,
    IdentityBlock,
)
from qarp.engines import QarpEngine


@pytest.fixture
def basic_blocks():
    """bra/operator/ket triple for use in constructor tests."""
    bra = ComputationalBasisStateBlock([0, 1])
    ket = ComputationalBasisStateBlock([0, 1])
    op = IdentityBlock(n_qubits=2)
    return bra, op, ket


# ── Constructor / parameter wiring ──────────────────────────────────────


def test_interferometric_test_init_default_parameters(basic_blocks):
    bra, op, ket = basic_blocks
    test = InterferometricTest(bra=bra, operator=op, ket=ket)

    assert test.bra is bra
    assert test.operator is op
    assert test.ket is ket
    assert test.real is True
    assert test.imaginary is True
    assert isinstance(test.sampling_algorithm, MirrorTest)
    assert test.n_shots is None
    assert test.result_real is None
    assert test.result_imaginary is None
    assert test.result is None
    assert isinstance(test.reference, ComputationalBasisStateBlock)
    assert test.reference.basis_state == [0, 0]


def test_interferometric_test_init_custom_parameters(basic_blocks):
    bra, op, ket = basic_blocks
    swap_test = SWAPTest()
    test = InterferometricTest(
        bra=bra,
        operator=op,
        ket=ket,
        real=False,
        imaginary=True,
        sampling_algorithm=swap_test,
        n_shots=1000,
    )
    assert test.real is False
    assert test.imaginary is True
    assert test.sampling_algorithm is swap_test
    assert test.n_shots == 1000


def test_interferometric_test_init_only_ket_required():
    ket = ComputationalBasisStateBlock([0, 1])
    test = InterferometricTest(ket=ket)
    assert test.ket is ket


def test_interferometric_test_reference_state_creation():
    """Reference is the all-zero state matching the ket's qubit count."""
    ket = ComputationalBasisStateBlock([1, 0, 1])
    test = InterferometricTest(ket=ket)
    assert isinstance(test.reference, ComputationalBasisStateBlock)
    assert test.reference.basis_state == [0, 0, 0]


def test_interferometric_test_n_qubits_property():
    ket = ComputationalBasisStateBlock([1, 0, 1])
    test = InterferometricTest(ket=ket)
    assert test.n_qubits == 3


# ── Validation paths ────────────────────────────────────────────────────


def test_validate_bra_must_be_basis_state():
    with pytest.raises(TypeError, match=r"bra must be a ComputationalBasisStateBlock"):
        InterferometricTest(bra="invalid", ket=ComputationalBasisStateBlock([0, 1]))


def test_validate_operator_must_be_block():
    with pytest.raises(TypeError, match=r"operator must be a Block"):
        InterferometricTest(operator="invalid", ket=ComputationalBasisStateBlock([0, 1]))


def test_validate_ket_must_be_basis_state():
    with pytest.raises(TypeError, match=r"ket must be a ComputationalBasisStateBlock"):
        InterferometricTest(ket="invalid")


def test_validate_real_or_imaginary_required():
    with pytest.raises(ValueError, match="At least one of real or imaginary"):
        InterferometricTest(ket=ComputationalBasisStateBlock([0, 1]), real=False, imaginary=False)


def test_validate_sampling_algorithm_type():
    with pytest.raises(ValueError, match=r"sampling_algorithm must be MirrorTest"):
        InterferometricTest(
            ket=ComputationalBasisStateBlock([0, 1]),
            sampling_algorithm="InvalidAlgorithm",
        )


def test_validate_ket_not_all_zero():
    with pytest.raises(ValueError, match="ket must not be the all-zero state"):
        InterferometricTest(ket=ComputationalBasisStateBlock([0, 0]))


# ── expectation_type / repr ─────────────────────────────────────────────


def test_expectation_type_property(basic_blocks):
    bra, op, ket = basic_blocks
    assert InterferometricTest(bra=bra, operator=op, ket=ket).expectation_type == "complex"
    assert (
        InterferometricTest(
            bra=bra, operator=op, ket=ket, real=True, imaginary=False
        ).expectation_type
        == "real"
    )
    assert (
        InterferometricTest(
            bra=bra, operator=op, ket=ket, real=False, imaginary=True
        ).expectation_type
        == "imaginary"
    )


def test_repr(basic_blocks):
    bra, op, ket = basic_blocks
    test = InterferometricTest(bra=bra, operator=op, ket=ket, n_shots=1000)
    test.target = Target.EXPECTATION_VALUE

    r = repr(test)
    assert "InterferometricTest" in r
    assert "n_shots=1000" in r


# ── End-to-end physical result via QarpEngine ───────────────────────────


@pytest.mark.parametrize("bs", [[0, 1], [1, 1]])
def test_correct_physical_results_mirror(bs):
    """⟨ψ|I|ψ⟩ = 1 for any non-zero |ψ⟩, MirrorTest sampler.

    MirrorTest is shot-based so the result carries sampling noise; bump the
    shot count to 1e6 to get a tight (~1e-3) tolerance on the imaginary part.
    """
    ket = ComputationalBasisStateBlock(bs)
    test = InterferometricTest(
        bra=ket,
        operator=IdentityBlock(n_qubits=2),
        ket=ket,
        real=True,
        imaginary=True,
        sampling_algorithm=MirrorTest(),
    )
    engine = QarpEngine(n_shots=1_000_000)
    engine.build([test])
    engine.run()

    assert abs(test.result_real - 1) < 1e-2
    assert abs(test.result_imaginary - 0) < 1e-2


def test_correct_physical_results_swap():
    """⟨ψ|I|ψ⟩ = 1 via SWAPTest sampler.  Shot-based, so use a high shot
    count + relaxed tolerance (per the testing guidelines in the plan)."""
    n_qubits = 2
    ket = ComputationalBasisStateBlock([0, 1])
    test = InterferometricTest(
        bra=ket,
        operator=IdentityBlock(n_qubits=n_qubits),
        ket=ket,
        real=True,
        imaginary=True,
        sampling_algorithm=SWAPTest(),
    )
    engine = QarpEngine(n_shots=1_000_000)
    engine.build([test])
    engine.run()

    assert abs(test.result_real - 1) < 1e-2
    assert abs(test.result_imaginary - 0) < 1e-2
