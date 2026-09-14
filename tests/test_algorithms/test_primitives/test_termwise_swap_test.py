"""Unit + integration tests for the ``TermwiseSWAPTest`` composition wrapper.

``TermwiseSWAPTest`` builds one child :class:`SWAPTest` per ``bra`` against a
single ``ket`` and sums the coefficient-weighted overlap estimates.  These
tests exercise the Python layer: constructor coercion/wiring, build()-time
validation, sub_blocks/sub_algorithms orchestration, run() weighting math
(via synthetic results fed to each child), error paths, and accessors.  One
integration test drives the full pipeline through a seeded ``QarpEngine``.
"""

from types import SimpleNamespace

import pytest

from qarp.algorithms import SWAPTest, TermwiseSWAPTest
from qarp.blocks import ComputationalBasisStateBlock
from qarp.engines import QarpEngine

# ── Constructor ─────────────────────────────────────────────────────────


def test_constructor_coerces_single_bra_to_list():
    bra = ComputationalBasisStateBlock(basis_state=[0])
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bra, ket=ket)
    assert isinstance(ts.bra, list)
    assert ts.bra == [bra]


def test_constructor_stores_bra_list_ket_coefficients():
    bra0 = ComputationalBasisStateBlock(basis_state=[0])
    bra1 = ComputationalBasisStateBlock(basis_state=[1])
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=[bra0, bra1], ket=ket, coefficients=[1.5, 2.5])
    assert ts.bra == [bra0, bra1]
    assert ts.ket is ket
    assert ts.coefficients == [1.5, 2.5]


def test_constructor_rejects_missing_bra():
    ket = ComputationalBasisStateBlock(basis_state=[0])
    with pytest.raises(ValueError, match="Both bra and ket must be provided"):
        TermwiseSWAPTest(bra=None, ket=ket)


def test_constructor_rejects_missing_ket():
    bra = ComputationalBasisStateBlock(basis_state=[0])
    with pytest.raises(ValueError, match="Both bra and ket must be provided"):
        TermwiseSWAPTest(bra=bra, ket=None)


# ── Validation (build) ──────────────────────────────────────────────────


def test_build_rejects_empty_bra_list():
    ket = ComputationalBasisStateBlock(basis_state=[0])
    with pytest.raises(ValueError, match="bra list cannot be empty"):
        TermwiseSWAPTest(bra=[], ket=ket).build()


def test_build_rejects_non_block_bra_entry():
    bra = ComputationalBasisStateBlock(basis_state=[0])
    ket = ComputationalBasisStateBlock(basis_state=[0])
    with pytest.raises(TypeError, match=r"bra\[1\] must be a Block instance"):
        TermwiseSWAPTest(bra=[bra, "not a block"], ket=ket).build()


def test_build_rejects_non_block_ket():
    bra = ComputationalBasisStateBlock(basis_state=[0])
    with pytest.raises(TypeError, match="ket must be a Block instance"):
        TermwiseSWAPTest(bra=[bra], ket="not a block").build()


def test_build_rejects_coefficients_length_mismatch():
    bra = ComputationalBasisStateBlock(basis_state=[0])
    ket = ComputationalBasisStateBlock(basis_state=[0])
    with pytest.raises(ValueError, match="coefficients must match the length of bra list"):
        TermwiseSWAPTest(bra=[bra, bra], ket=ket, coefficients=[1.0]).build()


# ── build() orchestration ───────────────────────────────────────────────


def test_build_sub_blocks_count_matches_bras():
    bras = [
        ComputationalBasisStateBlock(basis_state=[0]),
        ComputationalBasisStateBlock(basis_state=[1]),
        ComputationalBasisStateBlock(basis_state=[0]),
    ]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    ts.build()
    assert len(ts.sub_blocks) == 3
    assert len(ts.sub_algorithms) == 3


def test_build_default_coefficients_are_ones():
    bras = [
        ComputationalBasisStateBlock(basis_state=[0]),
        ComputationalBasisStateBlock(basis_state=[1]),
    ]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    ts.build()
    assert ts.coefficients == [1.0, 1.0]


def test_build_children_are_swap_tests():
    bras = [ComputationalBasisStateBlock(basis_state=[0])]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    ts.build()
    assert all(isinstance(c, SWAPTest) for c in ts.sub_algorithms)


# ── run() weighting math (synthetic results) ────────────────────────────


def test_run_weighted_sum():
    """Two bras, coefficients [2.0, 3.0].  child0 overlap=1.0, child1=0.0 →
    weighted [2.0, 0.0], sum 2.0."""
    bras = [
        ComputationalBasisStateBlock(basis_state=[1]),
        ComputationalBasisStateBlock(basis_state=[0]),
    ]
    ket = ComputationalBasisStateBlock(basis_state=[1])
    ts = TermwiseSWAPTest(bra=bras, ket=ket, coefficients=[2.0, 3.0])
    ts.build()

    # child0 reads bit0: all-zero counts → 2·P0 − 1 = 1.0.
    # child1: 50/50 split → 0.0.
    r0 = SimpleNamespace(counts={0: 1000}, n_shots=1000, n_qubits=1)
    r1 = SimpleNamespace(counts={0: 500, 1: 500}, n_shots=1000, n_qubits=1)

    total = ts.run([r0, r1])
    assert ts.result_list == pytest.approx([2.0, 0.0])
    assert ts.result_sum == pytest.approx(2.0)
    assert total == pytest.approx(2.0)


def test_run_before_build_raises():
    bras = [ComputationalBasisStateBlock(basis_state=[0])]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    with pytest.raises(ValueError, match="must be built before running"):
        ts.run([SimpleNamespace(counts={0: 1000}, n_shots=1000, n_qubits=1)])


def test_run_wrong_results_length_raises():
    bras = [
        ComputationalBasisStateBlock(basis_state=[0]),
        ComputationalBasisStateBlock(basis_state=[1]),
    ]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    ts.build()
    with pytest.raises(ValueError, match="Expected 2 results"):
        ts.run([SimpleNamespace(counts={0: 1000}, n_shots=1000, n_qubits=1)])


# ── Properties / accessors ──────────────────────────────────────────────


def test_len_and_n_bra():
    bras = [
        ComputationalBasisStateBlock(basis_state=[0]),
        ComputationalBasisStateBlock(basis_state=[1]),
    ]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    assert len(ts) == 2
    assert ts.n_bra == 2


def test_n_sub_algorithms_property():
    bras = [
        ComputationalBasisStateBlock(basis_state=[0]),
        ComputationalBasisStateBlock(basis_state=[1]),
    ]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    assert ts.n_sub_algorithms == 0  # not built yet
    ts.build()
    assert ts.n_sub_algorithms == 2


def test_expectation_type_is_real():
    bras = [ComputationalBasisStateBlock(basis_state=[0])]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    assert ts.expectation_type == "real"


def test_get_swap_test_returns_child():
    bras = [
        ComputationalBasisStateBlock(basis_state=[0]),
        ComputationalBasisStateBlock(basis_state=[1]),
    ]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    ts.build()
    child = ts.get_swap_test(1)
    assert child is ts.sub_algorithms[1]
    assert isinstance(child, SWAPTest)


def test_get_swap_test_before_build_raises():
    bras = [ComputationalBasisStateBlock(basis_state=[0])]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    with pytest.raises(ValueError, match="must be built first"):
        ts.get_swap_test(0)


# ── __repr__ ─────────────────────────────────────────────────────────────


def test_repr_contains_name_and_n_bra():
    bras = [
        ComputationalBasisStateBlock(basis_state=[0]),
        ComputationalBasisStateBlock(basis_state=[1]),
    ]
    ket = ComputationalBasisStateBlock(basis_state=[0])
    ts = TermwiseSWAPTest(bra=bras, ket=ket)
    r = repr(ts)
    assert "TermwiseSWAPTest" in r
    assert "n_bra=2" in r


# ── Integration (full build+run via QarpEngine) ─────────────────────────


def test_integration_weighted_sum():
    """Two bras [|1⟩, |0⟩] vs ket |1⟩, coefficients [2, 3] →
    2·1 + 3·0 = 2.0 (with sampling noise)."""
    ket = ComputationalBasisStateBlock(basis_state=[1])
    bras = [
        ComputationalBasisStateBlock(basis_state=[1]),
        ComputationalBasisStateBlock(basis_state=[0]),
    ]
    ts = TermwiseSWAPTest(bra=bras, ket=ket, coefficients=[2.0, 3.0], n_shots=2000)
    ts.build()
    eng = QarpEngine(n_shots=2000, seed=42)
    eng.build([ts])
    out = eng.run()[0]
    assert abs(out - 2.0) < 0.1
