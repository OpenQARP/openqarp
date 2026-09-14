"""Tests for the PauliAveraging primitive (general-commuting grouping path).

PauliAveraging estimates ``⟨ψ|H|ψ⟩`` for a Hermitian ``QubitOperator`` ``H``
by partitioning its Pauli terms into mutually-commuting groups (by *general*
multi-qubit commutation, not just qubit-wise) and running one
``ket → diagonalising-Clifford → measure-all`` circuit per group.  The
simultaneous diagonalisation is delegated to the C++ Clifford synthesiser, so
these tests exercise only the Python layer: input validation, operator
decomposition, the general-commuting grouping bookkeeping, the parity
post-processor in ``run()``, properties and ``__repr__`` — plus a couple of
end-to-end engine integrations that lock in the diagonalised general group.
"""

from types import SimpleNamespace

import pytest

from qarp.algorithms import PauliAveraging
from qarp.blocks import ComputationalBasisStateBlock, SimpleBlock
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator

# ── Validation (build()) ────────────────────────────────────────────────


def test_validate_ket_must_be_block():
    pa = PauliAveraging(ket="not-a-block", operator=QubitOperator("Z0"))
    with pytest.raises(TypeError, match="ket must be a Block instance"):
        pa.build()


def test_validate_operator_must_be_qubit_operator():
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator="not-an-operator",
    )
    with pytest.raises(TypeError, match="operator must be a QubitOperator instance"):
        pa.build()


# ── Operator decomposition ──────────────────────────────────────────────


def test_decompose_folds_identity_into_constant():
    """``H = 0.5·X0 + 0.3·Z0 + 0.2·I`` → 2 Pauli terms, constant tracks 0.2."""
    H = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.3) + QubitOperator("", 0.2)
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=H,
    )
    pa.build()
    assert pa.n_terms == 2
    assert pa._constant == pytest.approx(0.2)


# ── General-commuting grouping (key capability) ─────────────────────────


def test_general_commuting_grouping_beats_qwc():
    """``H = Z0 + (Y0 Z1 X2 X3 Y4) + (X0 X1)``.

    These three terms are pairwise *general*-commuting but NOT qubit-wise
    commuting, so general-commuting grouping packs them into 2 sets where QWC
    would force 3.  Locks in the general-commuting default.
    """
    H = QubitOperator("Z0") + QubitOperator("Y0 Z1 X2 X3 Y4") + QubitOperator("X0 X1")
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0, 0, 0, 0, 0]),
        operator=H,
    )
    pa.build()
    assert pa.n_terms == 3
    assert pa.n_groups == 2


def test_xx_yy_zz_all_commute_single_group():
    """``X0 X1 + Y0 Y1 + Z0 Z1`` mutually commute but are pairwise non-QWC.

    General commutation collapses all three into a single group (QWC would
    give 3).
    """
    H = QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0 Z1")
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0, 0]),
        operator=H,
    )
    pa.build()
    assert pa.n_terms == 3
    assert pa.n_groups == 1


def test_xx_yy_zz_on_bell_state_integration():
    """End-to-end Clifford-diagonalised general group on |Φ+⟩.

    On the Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2:
        ⟨XX⟩ = 1, ⟨YY⟩ = -1, ⟨ZZ⟩ = 1  →  ⟨H⟩ = 1.
    All three live in one general-commuting group, so this validates the
    single-Clifford diagonalisation end-to-end (including the -1 sign on YY).
    """
    bell = SimpleBlock(2)
    bell.h(0)
    bell.cx(0, 1)

    H = QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0 Z1")
    pa = PauliAveraging(ket=bell, operator=H, n_shots=8000)
    pa.build()
    assert pa.n_groups == 1

    eng = QarpEngine(n_shots=8000, seed=42)
    eng.build([pa])
    assert abs(eng.run()[0] - 1.0) < 0.1


# ── run() parity post-processing (synthetic) ────────────────────────────


def _build_z0(operator):
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=operator,
    )
    pa.build()
    return pa


def test_run_parity_all_zero_outcome_gives_plus_one():
    """Single Z0 group, all shots in outcome 0 → even parity → +1.0."""
    pa = _build_z0(QubitOperator("Z0"))
    assert pa.n_groups == 1
    sr = SimpleNamespace(counts={0: 1000}, n_shots=1000, n_qubits=1)
    assert pa.run([sr]) == pytest.approx(1.0)


def test_run_parity_all_one_outcome_gives_minus_one():
    """Single Z0 group, all shots in outcome 1 → odd parity → -1.0."""
    pa = _build_z0(QubitOperator("Z0"))
    sr = SimpleNamespace(counts={1: 1000}, n_shots=1000, n_qubits=1)
    assert pa.run([sr]) == pytest.approx(-1.0)


def test_run_parity_includes_constant():
    """``Z0 + 0.5·I``, all shots outcome 0 → 1.0 + 0.5 = 1.5."""
    pa = _build_z0(QubitOperator("Z0") + QubitOperator("", 0.5))
    assert pa._constant == pytest.approx(0.5)
    sr = SimpleNamespace(counts={0: 1000}, n_shots=1000, n_qubits=1)
    assert pa.run([sr]) == pytest.approx(1.5)


# ── Pure-constant operator ──────────────────────────────────────────────


def test_pure_constant_operator():
    """``H = 3·I`` — no Pauli terms, no groups, no circuits; run([]) = 3.0."""
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=QubitOperator("", 3.0),
    )
    pa.build()
    assert pa.n_terms == 0
    assert pa.n_groups == 0
    assert pa.sub_blocks == []
    assert pa.run([]) == pytest.approx(3.0)


# ── run() error handling ────────────────────────────────────────────────


def test_run_wrong_results_length_raises():
    pa = _build_z0(QubitOperator("Z0"))
    with pytest.raises(ValueError, match="group results"):
        pa.run([])


# ── Properties / __repr__ ───────────────────────────────────────────────


def test_properties_and_repr():
    H = QubitOperator("X0") + QubitOperator("X1") + QubitOperator("Z0")
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0, 0]),
        operator=H,
        n_shots=1234,
    )
    pa.build()
    assert pa.n_terms == 3
    assert isinstance(pa.n_groups, int)
    r = repr(pa)
    assert "PauliAveraging" in r
    assert "n_terms=3" in r
    assert "n_shots=1234" in r


# ── End-to-end engine integration (ported from e2e) ─────────────────────


def test_z_on_zero_is_plus_one():
    """``⟨0|Z0|0⟩ = 1`` (Z diagonal — exact)."""
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=QubitOperator("Z0"),
        n_shots=1000,
    )
    pa.build()
    eng = QarpEngine(n_shots=1000, seed=42)
    eng.build([pa])
    assert eng.run()[0] == pytest.approx(1.0)


def test_z_on_one_is_minus_one():
    """``⟨1|Z0|1⟩ = -1``."""
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[1]),
        operator=QubitOperator("Z0"),
        n_shots=1000,
    )
    pa.build()
    eng = QarpEngine(n_shots=1000, seed=42)
    eng.build([pa])
    assert eng.run()[0] == pytest.approx(-1.0)


def test_x_z_two_qubit_grouping_integration():
    """``H = X0 + X1 + Z0 + Z1`` on |00⟩ → 2 groups, ⟨H⟩ = 2.0."""
    H = QubitOperator("X0") + QubitOperator("X1") + QubitOperator("Z0") + QubitOperator("Z1")
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0, 0]),
        operator=H,
        n_shots=4000,
    )
    pa.build()
    assert pa.n_terms == 4
    assert pa.n_groups == 2
    eng = QarpEngine(n_shots=4000, seed=42)
    eng.build([pa])
    assert abs(eng.run()[0] - 2.0) < 0.1


def test_constant_term_integration():
    """``H = 0.5·X0 + 0.3·Z0 + 0.2·I`` on |0⟩ → 0.0 + 0.3 + 0.2 = 0.5."""
    H = QubitOperator("X0", 0.5) + QubitOperator("Z0", 0.3) + QubitOperator("", 0.2)
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=H,
        n_shots=4000,
    )
    pa.build()
    eng = QarpEngine(n_shots=4000, seed=42)
    eng.build([pa])
    assert abs(eng.run()[0] - 0.5) < 0.05


# ── Grouping-strategy injection ─────────────────────────────────────────


def test_default_grouping_is_fully_commuting():
    from qarp.operators import FullyCommuting

    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0]),
        operator=QubitOperator("Z0"),
    )
    assert isinstance(pa.grouping, FullyCommuting)
    assert "FullyCommuting" in repr(pa)


def test_no_grouping_one_circuit_per_term():
    from qarp.operators import NoGrouping

    H = QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0 Z1")
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0, 0]),
        operator=H,
        grouping=NoGrouping(),
    )
    pa.build()
    assert pa.n_groups == pa.n_terms == 3


def test_qwc_strategy_group_count():
    """XX+YY+ZZ: FullyCommuting → 1 group, QWC → 3 (pairwise non-QWC)."""
    from qarp.operators import QubitWiseCommuting

    H = QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0 Z1")
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0, 0]),
        operator=H,
        grouping=QubitWiseCommuting(),
    )
    pa.build()
    assert pa.n_groups == 3


def test_strategies_agree_on_bell_expectation():
    """⟨H⟩ on |Φ+⟩ is strategy-independent (only circuit count changes)."""
    from qarp.operators import FullyCommuting, NoGrouping, QubitWiseCommuting

    H = QubitOperator("X0 X1") + QubitOperator("Y0 Y1") + QubitOperator("Z0 Z1")
    for strategy in (FullyCommuting(), QubitWiseCommuting(), NoGrouping()):
        bell = SimpleBlock(2)
        bell.h(0)
        bell.cx(0, 1)
        pa = PauliAveraging(ket=bell, operator=H, n_shots=8000, grouping=strategy)
        pa.build()
        eng = QarpEngine(n_shots=8000, seed=42)
        eng.build([pa])
        assert abs(eng.run()[0] - 1.0) < 0.1, repr(strategy)


# ── P2.4: vectorised parity post-processing ─────────────────────────────


def _reference_run(pa, results):
    """The pre-P2.4 scalar loop, kept as the oracle for the vectorised
    ``run``: per term, per outcome, ``bin(outcome & mask).count("1")``."""
    energy = pa._constant
    for grp_idx, grp in enumerate(pa._groups):
        sr = results[grp_idx]
        for k, term_idx in enumerate(grp):
            mask = pa._group_masks[grp_idx][k]
            sign = -1.0 if pa._group_signs[grp_idx][k] else 1.0
            signed = sum(
                count if bin(outcome & mask).count("1") % 2 == 0 else -count
                for outcome, count in sr.counts.items()
            )
            energy += pa._coeffs[term_idx] * sign * signed / sr.n_shots
    return energy


def _random_operator(rng, n_qubits, n_terms, locality=4):
    op = QubitOperator()
    for _ in range(n_terms):
        qubits = rng.choice(n_qubits, locality, replace=False)
        op += QubitOperator(" ".join(f"{'XYZ'[rng.integers(3)]}{q}" for q in qubits), rng.normal())
    return op


@pytest.mark.parametrize("seed", range(3))
def test_run_matches_scalar_parity_loop_on_random_counts(seed):
    """Random masks, signs and outcome multisets on 12 qubits: the
    XOR-fold matrix reproduces the scalar popcount loop to 1e-12."""
    import numpy as np

    rng = np.random.default_rng(seed)
    n_qubits = 12
    pa = PauliAveraging(
        ket=ComputationalBasisStateBlock(basis_state=[0] * n_qubits),
        operator=_random_operator(rng, n_qubits, 60),
    )
    pa.build()
    results = []
    for _ in pa._groups:
        outcomes = rng.integers(0, 2**n_qubits, size=500)
        counts = {}
        for o in outcomes.tolist():
            counts[o] = counts.get(o, 0) + 1
        results.append(SimpleNamespace(counts=counts, n_shots=500, n_qubits=n_qubits))
    assert pa.run(results) == pytest.approx(_reference_run(pa, results), abs=1e-12)


def test_parity_helper_matches_popcount_up_to_bit_62():
    """The fold must reach every bit a 63-bit mask can set."""
    import numpy as np

    from qarp.algorithms._primitives.pauli_averaging import _parity

    rng = np.random.default_rng(0)
    values = rng.integers(0, 2**62, size=2000, dtype=np.int64)
    values[:63] = 1 << np.arange(63, dtype=np.int64)  # every single bit
    expected = [bin(int(v)).count("1") % 2 for v in values]
    assert _parity(values).tolist() == expected


def test_exact_readout_equals_statevector_expectation():
    """``qarp.EXACT`` counts through the vectorised parity sum equal the
    dense ⟨ψ|H|ψ⟩ from the statevector and ``H.sparse_matrix()`` at 1e-10."""
    import numpy as np

    import qarp
    import qarpx as qx
    from qarp.blocks import HEABlock

    rng = np.random.default_rng(3)
    n_qubits = 6
    H = _random_operator(rng, n_qubits, 30, locality=3)
    ket = HEABlock(n_qubits, 2, real=False, linear=True, circular=False, use_cz=False)
    ket.build()
    ket = ket.set_symbols(
        {s: float(v) for s, v in zip(ket.symbols, rng.normal(size=len(ket.symbols)), strict=True)}
    )
    psi = np.asarray(qx.QarpSimulator().statevector(ket.flatten(), n_qubits)).flatten()
    expected = np.vdot(psi, H.sparse_matrix(n_qubits) @ psi).real

    pa = PauliAveraging(ket=ket, operator=H, n_shots=qarp.EXACT)
    engine = QarpEngine(n_shots=qarp.EXACT)
    engine.build([pa])
    assert engine.run()[0] == pytest.approx(expected, abs=1e-10)


@pytest.mark.bench
def test_post_processing_is_a_small_fraction_of_the_run():
    """P2.4 target: parity post-processing ≤ 10 % of ``engine.run`` at
    16 qubits / 1000 terms / 10k shots (was 44 % with the scalar loop)."""
    import time
    from unittest import mock

    import numpy as np

    from qarp.blocks import HEABlock

    rng = np.random.default_rng(0)
    n_qubits, n_shots = 16, 10_000
    H = _random_operator(rng, n_qubits, 1000)
    ket = HEABlock(n_qubits, 2, real=True, linear=True, circular=False, use_cz=False)
    ket.build()
    ket = ket.set_symbols({s: 0.3 for s in ket.symbols})
    pa = PauliAveraging(ket=ket, operator=H, n_shots=n_shots)
    engine = QarpEngine(n_shots=n_shots, seed=1)
    engine.build([pa])

    captured = {}
    original = PauliAveraging.run

    def spy(self, results):
        captured["results"] = results
        return original(self, results)

    with mock.patch.object(PauliAveraging, "run", spy):
        t = time.perf_counter()
        energy = engine.run()[0]
        t_run = time.perf_counter() - t
    t = time.perf_counter()
    assert pa.run(captured["results"]) == energy
    t_post = time.perf_counter() - t
    assert t_post < 0.10 * t_run, f"post-processing {t_post:.3f}s of {t_run:.3f}s"
