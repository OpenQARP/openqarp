"""Unit + integration tests for the MirrorTest primitive.

MirrorTest estimates ``|⟨bra|U|ket⟩|²`` by appending the mirror sequence
``ket · U · bra†`` to ``|0…0⟩`` and reading ``P(all zeros)``.  These tests
cover the Python layer only: constructor wiring, input validation,
``build()`` orchestration, ``run()`` post-processing, ``__repr__`` and a few
end-to-end physical checks through ``QarpEngine``.
"""

from types import SimpleNamespace

import pytest

from qarp.algorithms import MirrorTest, Target
from qarp.blocks import ComputationalBasisStateBlock, HnBlock
from qarp.engines import QarpEngine

# ── Constructor ─────────────────────────────────────────────────────────


def test_mirror_test_constructor_stores_inputs():
    bra = ComputationalBasisStateBlock(basis_state=[0])
    ket = ComputationalBasisStateBlock(basis_state=[1])
    op = HnBlock(1)
    mt = MirrorTest(bra=bra, operator=op, ket=ket, n_shots=512)

    assert mt.target == Target.OVERLAP
    assert mt.bra is bra
    assert mt.ket is ket
    assert mt.operator is op
    assert mt.n_shots == 512


# ── Validation (via build()) ────────────────────────────────────────────


def test_mirror_test_rejects_non_block_bra():
    with pytest.raises(TypeError, match="bra must be a Block instance"):
        MirrorTest(bra="not a block", ket=ComputationalBasisStateBlock(basis_state=[0])).build()


def test_mirror_test_rejects_non_block_ket():
    with pytest.raises(TypeError, match="ket must be a Block instance"):
        MirrorTest(bra=ComputationalBasisStateBlock(basis_state=[0]), ket="not a block").build()


def test_mirror_test_rejects_non_block_operator():
    with pytest.raises(TypeError, match="operator must be a Block instance or None"):
        MirrorTest(
            bra=ComputationalBasisStateBlock(basis_state=[0]),
            ket=ComputationalBasisStateBlock(basis_state=[0]),
            operator="not a block",
        ).build()


# ── build() orchestration ───────────────────────────────────────────────


def test_mirror_test_build_single_composite_subblock():
    bra = ComputationalBasisStateBlock(basis_state=[1, 0, 1])
    ket = ComputationalBasisStateBlock(basis_state=[1, 0, 1])
    mt = MirrorTest(bra=bra, ket=ket, n_shots=100)
    mt.build()

    assert len(mt.sub_blocks) == 1
    composite = mt.sub_blocks[0]
    assert mt.n_qubits == 3
    assert composite.n_cbits == 3


def test_mirror_test_build_n_qubits_is_max_of_widths():
    """n_qubits is the max over ket / bra / operator widths."""
    bra = ComputationalBasisStateBlock(basis_state=[0, 0])
    ket = ComputationalBasisStateBlock(basis_state=[0, 0])
    op = HnBlock(4)  # widest input
    mt = MirrorTest(bra=bra, ket=ket, operator=op, n_shots=100)
    mt.build()

    assert mt.n_qubits == 4
    assert mt.sub_blocks[0].n_cbits == 4


def test_mirror_test_build_with_operator_completes():
    """ket · op · bra† wiring builds without error and yields one composite."""
    bra = ComputationalBasisStateBlock(basis_state=[0])
    ket = ComputationalBasisStateBlock(basis_state=[0])
    op = HnBlock(1)
    mt = MirrorTest(bra=bra, ket=ket, operator=op, n_shots=100)
    mt.build()

    assert len(mt.sub_blocks) == 1
    assert mt.n_qubits == 1


# ── run() unit (synthetic results) ──────────────────────────────────────


def test_mirror_test_run_returns_p_all_zeros():
    mt = MirrorTest(
        bra=ComputationalBasisStateBlock(basis_state=[0, 0]),
        ket=ComputationalBasisStateBlock(basis_state=[0, 0]),
        n_shots=1000,
    )
    sr = SimpleNamespace(counts={0: 900, 3: 100}, n_shots=1000)
    assert mt.run([sr]) == pytest.approx(0.9)
    assert mt.result == pytest.approx(0.9)


def test_mirror_test_run_zero_when_no_all_zeros_outcome():
    mt = MirrorTest(
        bra=ComputationalBasisStateBlock(basis_state=[0, 0]),
        ket=ComputationalBasisStateBlock(basis_state=[0, 0]),
        n_shots=1000,
    )
    sr = SimpleNamespace(counts={1: 400, 2: 600}, n_shots=1000)
    assert mt.run([sr]) == pytest.approx(0.0)


# ── __repr__ ─────────────────────────────────────────────────────────────


def test_mirror_test_repr():
    mt = MirrorTest(n_shots=256)
    assert "MirrorTest" in repr(mt)


# ── Integration through QarpEngine ───────────────────────────────────────


def test_mirror_test_same_state_overlap_one():
    """|1,0,1⟩ vs itself → P(all zeros) ≈ 1 (overlap² = 1)."""
    state = [1, 0, 1]
    mt = MirrorTest(
        bra=ComputationalBasisStateBlock(basis_state=state),
        ket=ComputationalBasisStateBlock(basis_state=state),
        n_shots=1000,
    )
    mt.build()
    eng = QarpEngine(n_shots=1000, seed=42)
    eng.build([mt])
    assert eng.run()[0] == pytest.approx(1.0)


def test_mirror_test_orthogonal_states_overlap_zero():
    mt = MirrorTest(
        bra=ComputationalBasisStateBlock(basis_state=[0, 1, 0]),
        ket=ComputationalBasisStateBlock(basis_state=[1, 0, 1]),
        n_shots=1000,
    )
    mt.build()
    eng = QarpEngine(n_shots=1000, seed=42)
    eng.build([mt])
    assert eng.run()[0] == pytest.approx(0.0)


def test_mirror_test_partial_overlap_hn():
    """|0,0,0,0⟩ vs Hn(4) → P(all zeros) = overlap² = 1/2^4 = 1/16."""
    mt = MirrorTest(
        bra=ComputationalBasisStateBlock(basis_state=[0, 0, 0, 0]),
        ket=HnBlock(4),
        n_shots=8000,
    )
    mt.build()
    eng = QarpEngine(n_shots=8000, seed=42)
    eng.build([mt])
    out = eng.run()[0]
    assert abs(out - 1.0 / 16) < 0.03, f"got {out}, expected {1.0 / 16}"


def test_mirror_test_composite_bra_dagger_and_deepcopy():
    """A CompositeBlock bra must be correctly daggered and deepcopy-safe.

    Guards the composite-dagger trap: a composite's daggered top-level
    command buffer is empty (gates live in the children), so a
    mis-materialised bra† silently becomes identity — here that would give
    ``P(all zeros) = |⟨0|X₀H₁|00⟩|² = 0`` instead of 1.  Also checks the
    built primitive stays deepcopy-safe (no raw C++-only child).
    """
    import copy

    from qarp.blocks import CompositeBlock, SimpleBlock

    def make_state():
        x = SimpleBlock(1, name="x")
        x.x(0)
        x.build()
        x.target_qubits = [0]
        h = SimpleBlock(1, name="h")
        h.h(0)
        h.build()
        h.target_qubits = [1]
        return CompositeBlock([x, h], n_qubits=2)

    mt = MirrorTest(bra=make_state(), ket=make_state(), n_shots=1000)
    mt.build()
    eng = QarpEngine(n_shots=1000, seed=42)
    eng.build([mt])
    assert eng.run()[0] == pytest.approx(1.0)

    mt_copy = copy.deepcopy(mt.sub_blocks[0])
    assert len(list(mt_copy.flatten())) == len(list(mt.sub_blocks[0].flatten()))
