"""Tests for DOSQPEBlock — structural + density-of-states recovery."""

import numpy as np
import pytest

from qarp.algorithms import Sampler
from qarp.blocks import (
    DOSQPEBlock,
    HnBlock,
    IdentityBlock,
    QPEBlock,
    SimpleBlock,
)
from qarp.engines import QarpEngine


def test_dos_qpe_constructs_and_builds():
    block = DOSQPEBlock(
        eigenstate=HnBlock(2).build(),
        unitary=HnBlock(2).build(),
        n_ancilla=3,
        n_state=2,
    ).build()
    assert block.is_built
    # n_ancilla + 2 * n_state.
    assert block.n_qubits == 7
    assert len(block.flatten()) > 0


@pytest.mark.parametrize(
    "kwarg", ["optimize_ctrl_u_sequences", "max_optimized_repetitions", "fully_optimized"]
)
def test_dos_qpe_dead_optimization_kwargs_are_gone(kwarg):
    with pytest.raises(TypeError):
        DOSQPEBlock(
            eigenstate=HnBlock(2).build(),
            unitary=HnBlock(2).build(),
            n_ancilla=2,
            n_state=2,
            **{kwarg: True},
        )


def test_dos_qpe_measure_false_emits_no_measure_cmds():
    block = DOSQPEBlock(
        eigenstate=HnBlock(1).build(),
        unitary=HnBlock(1).build(),
        n_ancilla=2,
        n_state=1,
        measure=False,
    ).build()
    assert not any(str(c).startswith("Measure") for c in block.flatten())


def test_dos_qpe_measure_true_emits_one_measure_per_ancilla():
    n_ancilla = 3
    block = DOSQPEBlock(
        eigenstate=HnBlock(1).build(),
        unitary=HnBlock(1).build(),
        n_ancilla=n_ancilla,
        n_state=1,
        measure=True,
    ).build()
    n_meas = sum(1 for c in block.flatten() if str(c).startswith("Measure"))
    assert n_meas == n_ancilla


@pytest.mark.parametrize("n_state", [1, 2, 3])
def test_dos_qpe_purification_cx_count(n_state):
    """The purification layer must emit exactly n_state CX gates when U=Id
    contributes no CX of its own."""
    block = DOSQPEBlock(
        eigenstate=IdentityBlock(n_state).build(),
        unitary=IdentityBlock(n_state).build(),
        n_ancilla=2,
        n_state=n_state,
        measure=False,
    ).build()
    cx_count = sum(1 for c in block.flatten() if str(c).startswith("CX "))
    assert cx_count == n_state


def test_dos_qpe_n_qubits_strictly_larger_than_qpe():
    """DOSQPE adds a purification register of size n_state to the QPE layout."""
    n_ancilla, n_state = 3, 2
    dos = DOSQPEBlock(
        eigenstate=HnBlock(n_state).build(),
        unitary=HnBlock(n_state).build(),
        n_ancilla=n_ancilla,
        n_state=n_state,
    ).build()
    qpe = QPEBlock(
        eigenstate=HnBlock(n_state).build(),
        unitary=HnBlock(n_state).build(),
        n_ancilla=n_ancilla,
        n_state=n_state,
    ).build()
    assert dos.n_qubits == qpe.n_qubits + n_state


@pytest.mark.parametrize(
    "phases",
    [
        [0.25, 0.75],  # 1-qubit, two phases
        [0.125, 0.625],  # 1-qubit, two phases on different bins
        [0.125, 0.375, 0.625, 0.875],  # 2-qubit, four equal-weight phases
    ],
)
def test_dos_qpe_recovers_diagonal_spectrum(phases):
    """Diagonal U with eigenvalues e^{2πi·φ_k} and a maximally-mixed probe
    (Hn + CNOT purification) ⇒ ancilla histogram has equal-weight peaks at
    exactly the phases φ_k."""
    n_state = int(np.log2(len(phases)))
    assert 2**n_state == len(phases), "phases length must be a power of two"
    n_ancilla = 4

    diag = [np.exp(1j * 2 * np.pi * p) for p in phases]
    u = SimpleBlock(n_state, name="U")
    u.diagonal_unitary(diag)
    u.build()

    block = DOSQPEBlock(
        eigenstate=HnBlock(n_state).build(),
        unitary=u,
        n_ancilla=n_ancilla,
        n_state=n_state,
        measure=True,
    ).build()

    engine = QarpEngine(seed=0, n_shots=4000)
    sampler = Sampler(ket=block, measured_qubits=list(range(n_ancilla)))
    sampler.build()
    engine.build([sampler])
    dist = engine.run()[0]

    by_phi = {}
    for tup, p in dist.items():
        phi = int("".join(str(i) for i in reversed(tup)), 2) / 2**n_ancilla
        by_phi[phi] = by_phi.get(phi, 0.0) + p

    expected_weight = 1.0 / len(phases)
    for ph in phases:
        assert by_phi.get(ph, 0.0) == pytest.approx(expected_weight, abs=0.05), (
            f"phase {ph} not found at expected weight in {sorted(by_phi.items())}"
        )


@pytest.mark.parametrize("n_ancilla,n_state", [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2)])
def test_dos_qpe_various_sizes_build(n_ancilla, n_state):
    block = DOSQPEBlock(
        eigenstate=IdentityBlock(n_state).build(),
        unitary=IdentityBlock(n_state).build(),
        n_ancilla=n_ancilla,
        n_state=n_state,
        measure=True,
    ).build()
    assert block.is_built
    assert block.n_qubits == n_ancilla + 2 * n_state
