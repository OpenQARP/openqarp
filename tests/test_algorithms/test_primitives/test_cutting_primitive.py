"""Tests for the CuttingPrimitive (QPD circuit cutting).

CuttingPrimitive wraps the ``qarp.cutting`` pipeline (EAPartitioning →
QPDDecomposition) as a standard ``PrimitiveAlgorithm``: ``build()`` partitions
the circuit and produces ``sub_blocks`` for the engine, and ``run(results)``
reconstructs the expectation value via QWC-bitmask post-processing.

These tests cover input validation, the ``build()`` pipeline (subcircuit
count, cut count, ``_qpd`` wiring), the ``config.max_number_of_cuts`` guard,
``experiment_fraction``/``rng_seed`` pass-through, and end-to-end engine
integration on small circuits with known expectation values.
"""

import pytest

import qarpx as qx
from qarp import config
from qarp.algorithms import CuttingPrimitive, PauliAveraging, StateVector
from qarp.blocks import CompositeBlock, HnBlock, LinearEntanglingBlock, SimpleBlock
from qarp.cutting import QPDDecomposition
from qarp.engines import QarpEngine
from qarp.operators import QubitOperator

# ── Circuit builders ────────────────────────────────────────────────────────


def _separable_pairs_block() -> SimpleBlock:
    """Two independent Bell pairs on (0,1) and (2,3) — no cross-pair gates.

    Already separable into two 2-qubit components, so with
    ``max_subcircuit_qubits=2`` (subcircuit capacity [2, 2]) the automatic
    partitioner needs zero cuts.
    """
    b = SimpleBlock(4, name="pairs")
    b.h(0)
    b.cx(0, 1)
    b.h(2)
    b.cx(2, 3)
    return b


def _ghz_chain_block(n_qubits: int = 4) -> SimpleBlock:
    """Linear-chain GHZ preparation: a single connected component."""
    b = SimpleBlock(n_qubits, name="ghz")
    b.h(0)
    for i in range(n_qubits - 1):
        b.cx(i, i + 1)
    return b


# ── Validation (build()) ─────────────────────────────────────────────────


def test_validate_ket_required():
    cp = CuttingPrimitive(ket=None, operator=QubitOperator("Z0"), max_subcircuit_qubits=2)
    with pytest.raises(RuntimeError, match="ket must be set"):
        cp.build()


def test_validate_operator_required():
    cp = CuttingPrimitive(ket=_ghz_chain_block(), operator=None, max_subcircuit_qubits=2)
    with pytest.raises(RuntimeError, match="operator must be set"):
        cp.build()


def test_rejects_openfermion_qubit_operator():
    """`operator` must be a qarp.operators.QubitOperator, not openfermion's —
    CuttingPrimitive forwards it straight into QPDDecomposition, which
    validates (see test_qpd_decomposition.test_rejects_openfermion_qubit_operator
    for the same check at that layer)."""
    openfermion = pytest.importorskip("openfermion")

    cp = CuttingPrimitive(
        ket=_ghz_chain_block(),
        operator=openfermion.QubitOperator("Z0"),
        max_subcircuit_qubits=2,
    )
    with pytest.raises(TypeError, match="qarp.operators.QubitOperator"):
        cp.build()


# ── build() pipeline ─────────────────────────────────────────────────────


def test_build_already_separable_circuit_needs_no_cuts():
    """Two independent pairs on a 2-qubit device → 0 cuts, 2 subcircuits."""
    cp = CuttingPrimitive(
        ket=_separable_pairs_block(),
        operator=QubitOperator("Z0 Z1") + QubitOperator("Z2 Z3"),
        max_subcircuit_qubits=2,
        n_shots=200,
    )
    cp.build()

    assert cp._cutter_result.n_cuts == 0
    assert cp._cutter_result.n_subcircuits == 2
    assert isinstance(cp._qpd, QPDDecomposition)
    assert cp._qpd.n_experiments == 6**0 == 1
    assert len(cp.sub_blocks) > 0
    assert all(isinstance(b, qx.Block) for b in cp.sub_blocks)


def test_build_non_separable_circuit_requires_cuts():
    """A single connected 4-qubit chain on a 2-qubit device needs >= 1 cut."""
    config.seed = 42
    cp = CuttingPrimitive(
        ket=_ghz_chain_block(4),
        operator=QubitOperator("Z0 Z1 Z2 Z3"),
        max_subcircuit_qubits=2,
        n_shots=200,
    )
    cp.build()

    assert cp._cutter_result.n_cuts >= 1
    assert cp._cutter_result.n_subcircuits == 2
    assert cp._qpd.n_experiments == 6**cp._cutter_result.n_cuts
    assert len(cp.sub_blocks) > 0


# ── config.max_number_of_cuts guard ──────────────────────────────────────


def test_max_number_of_cuts_guard_raises_and_can_be_forced():
    config.seed = 42
    old_max = config.max_number_of_cuts
    try:
        config.max_number_of_cuts = 1

        cp = CuttingPrimitive(
            ket=_ghz_chain_block(4),
            operator=QubitOperator("Z0 Z1 Z2 Z3"),
            max_subcircuit_qubits=2,
            n_shots=200,
        )
        with pytest.raises(RuntimeError, match="exceeds"):
            cp.build()

        cp_forced = CuttingPrimitive(
            ket=_ghz_chain_block(4),
            operator=QubitOperator("Z0 Z1 Z2 Z3"),
            max_subcircuit_qubits=2,
            n_shots=200,
            force_max_number_cuts=True,
        )
        cp_forced.build()
        assert cp_forced._cutter_result.n_cuts >= 1
    finally:
        config.max_number_of_cuts = old_max


# ── experiment_fraction / rng_seed pass-through ──────────────────────────


def test_experiment_fraction_and_sampling_strategy():
    """A CX-only circuit auto-selects the 'uniform' subsampling strategy."""
    config.seed = 42
    cp = CuttingPrimitive(
        ket=_ghz_chain_block(4),
        operator=QubitOperator("Z0 Z1 Z2 Z3"),
        max_subcircuit_qubits=2,
        n_shots=200,
        experiment_fraction=0.5,
        rng_seed=0,
    )
    cp.build()

    total = 6**cp._cutter_result.n_cuts
    assert cp._qpd.sampling_strategy == "uniform"
    assert cp._qpd.n_experiments == max(1, int(total * 0.5))


def test_rng_seed_reproducible_experiment_selection():
    """Same rng_seed → same subsampled experiment set (uniform strategy)."""
    config.seed = 42

    def _build():
        cp = CuttingPrimitive(
            ket=_ghz_chain_block(4),
            operator=QubitOperator("Z0 Z1 Z2 Z3"),
            max_subcircuit_qubits=2,
            n_shots=200,
            experiment_fraction=0.5,
            rng_seed=7,
        )
        cp.build()
        return cp

    cp1 = _build()
    cp2 = _build()

    assert cp1._qpd.n_experiments == cp2._qpd.n_experiments
    assert len(cp1.sub_blocks) == len(cp2.sub_blocks)


# ── End-to-end engine integration ─────────────────────────────────────────


def test_ghz_chain_zzzz_expectation_is_one():
    """``<Z0 Z1 Z2 Z3>`` on a 4-qubit GHZ chain is exactly +1, even after cutting."""
    config.seed = 42

    cp = CuttingPrimitive(
        ket=_ghz_chain_block(4),
        operator=QubitOperator("Z0 Z1 Z2 Z3"),
        max_subcircuit_qubits=2,
        n_shots=4000,
    )
    eng = QarpEngine(n_shots=4000, seed=42)
    eng.build([cp])
    result = eng.run()[0]

    assert isinstance(result, float)
    assert abs(result - 1.0) < 0.15


def test_cutting_matches_uncut_pauli_averaging():
    """Cutting a circuit should reproduce the uncut expectation value (notebook setup)."""
    config.seed = 42

    n_qubits = 4
    block = CompositeBlock(
        blocks=[
            HnBlock(n_qubits),
            LinearEntanglingBlock(n_qubits, circular=False, use_cz=False),
        ]
    )

    hamiltonian = (
        0.16988452027940318 * QubitOperator("Z0")
        - 0.21886306781219608 * QubitOperator("Z0 Z1 Z2")
        + 0.04544288414432624 * QubitOperator("Y0 Y1 Y2")
        + 0.04544288414432624
    )

    cutting_prim = CuttingPrimitive(
        ket=block,
        operator=hamiltonian,
        max_subcircuit_qubits=2,
        n_shots=8000,
    )
    eng_cut = QarpEngine(n_shots=8000, seed=42)
    eng_cut.build([cutting_prim])
    result_cut = eng_cut.run()[0]

    pauli_avg = PauliAveraging(ket=block, operator=hamiltonian, n_shots=8000)
    eng_exact = QarpEngine(n_shots=8000, seed=42)
    eng_exact.build([pauli_avg])
    result_exact = eng_exact.run()[0]

    assert abs(result_cut - result_exact) < 0.2


def test_cutting_matches_state_vector_exact_value():
    """A cut circuit's sampled expectation value should match the exact one
    obtained from ``StateVector`` (no cutting, no sampling) on the same
    ket/operator."""
    config.seed = 42

    n_qubits = 4

    def _build_block():
        return CompositeBlock(
            blocks=[
                HnBlock(n_qubits),
                LinearEntanglingBlock(n_qubits, circular=False, use_cz=False),
            ]
        )

    hamiltonian = (
        0.16988452027940318 * QubitOperator("Z0")
        - 0.21886306781219608 * QubitOperator("Z0 Z1 Z2")
        + 0.04544288414432624 * QubitOperator("Y0 Y1 Y2")
        + 0.04544288414432624
    )

    cutting_prim = CuttingPrimitive(
        ket=_build_block(),
        operator=hamiltonian,
        max_subcircuit_qubits=2,
        n_shots=8000,
    )
    eng_cut = QarpEngine(n_shots=8000, seed=42)
    eng_cut.build([cutting_prim])
    result_cut = eng_cut.run()[0]

    sv = StateVector(ket=_build_block(), operator=hamiltonian)
    eng_exact = QarpEngine()
    eng_exact.build([sv])
    result_exact = eng_exact.run()[0]

    assert abs(result_cut - result_exact.real) < 0.2
