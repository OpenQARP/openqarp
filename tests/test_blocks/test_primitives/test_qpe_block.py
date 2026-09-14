"""Tests for QPEBlock — structural + numeric phase recovery."""

import numpy as np
import pytest

from qarp.algorithms import Sampler
from qarp.blocks import (
    ComputationalBasisStateBlock,
    HnBlock,
    IdentityBlock,
    QPEBlock,
    SimpleBlock,
)
from qarp.engines import QarpEngine


def _phi_from_sample(distribution, n_ancilla):
    """Sampler emits LSB-first tuples; decode to phi ∈ [0, 1)."""
    best = max(distribution, key=distribution.get)
    return int("".join(str(i) for i in reversed(best)), 2) / 2**n_ancilla


def test_qpe_constructs_and_builds():
    block = QPEBlock(
        eigenstate=HnBlock(2).build(),
        unitary=HnBlock(2).build(),
        n_ancilla=3,
        n_state=2,
    ).build()
    assert block.is_built
    assert block.n_qubits == 5
    assert len(block.flatten()) > 0


@pytest.mark.parametrize(
    "kwarg", ["optimize_ctrl_u_sequences", "max_optimized_repetitions", "fully_optimized"]
)
def test_qpe_dead_optimization_kwargs_are_gone(kwarg):
    with pytest.raises(TypeError):
        QPEBlock(
            eigenstate=HnBlock(2).build(),
            unitary=HnBlock(2).build(),
            n_ancilla=2,
            n_state=2,
            **{kwarg: True},
        )


def test_qpe_measure_false_emits_no_measure_cmds():
    block = QPEBlock(
        eigenstate=HnBlock(1).build(),
        unitary=HnBlock(1).build(),
        n_ancilla=2,
        n_state=1,
        measure=False,
    ).build()
    flat = block.flatten()
    assert not any(str(c).startswith("Measure") for c in flat)


def test_qpe_measure_true_emits_one_measure_per_ancilla():
    n_ancilla = 3
    block = QPEBlock(
        eigenstate=HnBlock(1).build(),
        unitary=HnBlock(1).build(),
        n_ancilla=n_ancilla,
        n_state=1,
        measure=True,
    ).build()
    n_meas = sum(1 for c in block.flatten() if str(c).startswith("Measure"))
    assert n_meas == n_ancilla


def test_qpe_z_on_one_recovers_phi_half():
    """Z gate has eigenvalue e^{iπ} on |1⟩, i.e. φ = 0.5 (representable at n_ancilla≥1)."""
    n_ancilla = 3
    u = SimpleBlock(1, name="Z")
    u.z(0)
    u.build()

    block = QPEBlock(
        eigenstate=ComputationalBasisStateBlock([1]).build(),
        unitary=u,
        n_ancilla=n_ancilla,
        n_state=1,
        measure=True,
    ).build()

    engine = QarpEngine(seed=0, n_shots=200)
    sampler = Sampler(ket=block, measured_qubits=list(range(n_ancilla)))
    sampler.build()
    engine.build([sampler])
    dist = engine.run()[0]

    assert _phi_from_sample(dist, n_ancilla) == pytest.approx(0.5)


@pytest.mark.parametrize("phi", [0.125, 0.375, 0.625, 0.875])
def test_qpe_phase_block_recovers_representable_phi(phi):
    """U = P(2πφ) has eigenvalue e^{2πiφ} on |1⟩.  At n_ancilla=3 every k/8
    is exactly representable → QPE must land on the right bin with prob 1."""
    n_ancilla = 3
    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * phi)
    u.build()

    block = QPEBlock(
        eigenstate=ComputationalBasisStateBlock([1]).build(),
        unitary=u,
        n_ancilla=n_ancilla,
        n_state=1,
        measure=True,
    ).build()

    engine = QarpEngine(seed=0, n_shots=200)
    sampler = Sampler(ket=block, measured_qubits=list(range(n_ancilla)))
    sampler.build()
    engine.build([sampler])
    dist = engine.run()[0]

    assert _phi_from_sample(dist, n_ancilla) == pytest.approx(phi)


@pytest.mark.parametrize(
    "phases,occ",
    [
        ([0.25, 0.75], [1, 1]),  # equal weight on |0⟩, |1⟩ via |+⟩ probe
    ],
)
def test_qpe_two_eigenvalue_spectrum_on_diagonal_u(phases, occ):
    """Diagonal U with two phases (0.25, 0.75) and a superposition probe ⇒
    QPE distribution has support on exactly those two bins."""
    n_ancilla = 3
    n_state = 1
    diag = [np.exp(1j * 2 * np.pi * p) for p in phases]
    u = SimpleBlock(n_state, name="U")
    u.diagonal_unitary(diag)
    u.build()

    block = QPEBlock(
        eigenstate=HnBlock(n_state).build(),  # |+⟩ → uniform mix of |0⟩, |1⟩
        unitary=u,
        n_ancilla=n_ancilla,
        n_state=n_state,
        measure=True,
    ).build()

    engine = QarpEngine(seed=0, n_shots=2000)
    sampler = Sampler(ket=block, measured_qubits=list(range(n_ancilla)))
    sampler.build()
    engine.build([sampler])
    dist = engine.run()[0]

    # Aggregate probability by recovered phi.
    by_phi = {}
    for tup, p in dist.items():
        phi = int("".join(str(i) for i in reversed(tup)), 2) / 2**n_ancilla
        by_phi[phi] = by_phi.get(phi, 0.0) + p

    # Each known phase should carry essentially equal weight (1/len(phases)).
    expected_weight = 1.0 / len(phases)
    for ph, w in zip(phases, occ, strict=True):
        assert by_phi.get(ph, 0.0) == pytest.approx(expected_weight, abs=0.05), (
            f"phase {ph} missing from {by_phi}"
        )


@pytest.mark.parametrize("n_ancilla,n_state", [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2)])
def test_qpe_various_sizes_build(n_ancilla, n_state):
    block = QPEBlock(
        eigenstate=IdentityBlock(n_state).build(),
        unitary=IdentityBlock(n_state).build(),
        n_ancilla=n_ancilla,
        n_state=n_state,
        measure=True,
    ).build()
    assert block.is_built
    assert block.n_qubits == n_ancilla + n_state
