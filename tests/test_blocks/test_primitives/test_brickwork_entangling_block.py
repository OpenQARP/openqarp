import pytest

import qarpx as qx
from qarp.blocks import BrickworkEntanglingBlock


def test_brickwork_all_cz():
    """All gates should be CZ for use_cz=True."""
    block = BrickworkEntanglingBlock(n_qubits=4, circular=True, use_cz=True)
    block.build()
    cmds = list(block.commands())
    assert all(qx.gate_name(c.gate) == "CZ" for c in cmds)


def test_brickwork_all_cx():
    """All gates should be CX for use_cz=False."""
    block = BrickworkEntanglingBlock(n_qubits=4, circular=False, use_cz=False)
    block.build()
    cmds = list(block.commands())
    assert all(qx.gate_name(c.gate) == "CX" for c in cmds)


def test_brickwork_circular_adds_one_gate():
    """circular=True should add exactly one extra gate."""
    cz_circular = BrickworkEntanglingBlock(n_qubits=4, circular=True, use_cz=True)
    cz_circular.build()
    cx_linear = BrickworkEntanglingBlock(n_qubits=4, circular=False, use_cz=False)
    cx_linear.build()

    n_circular = len(list(cz_circular.commands()))
    n_linear = len(list(cx_linear.commands()))
    assert n_circular == n_linear + 1


def test_brickwork_gate_pairs_4_qubits():
    """For 4 qubits non-circular: even pairs (0,1),(2,3) then odd pair (1,2)."""
    block = BrickworkEntanglingBlock(n_qubits=4, circular=False, use_cz=False)
    block.build()
    cmds = list(block.commands())

    assert len(cmds) == 3
    assert list(cmds[0].qubits) == [0, 1]
    assert list(cmds[1].qubits) == [2, 3]
    assert list(cmds[2].qubits) == [1, 2]


def test_brickwork_gate_pairs_4_qubits_circular():
    """For 4 qubits circular: even (0,1),(2,3), odd (1,2), then circular (3,0)."""
    block = BrickworkEntanglingBlock(n_qubits=4, circular=True, use_cz=True)
    block.build()
    cmds = list(block.commands())

    assert len(cmds) == 4
    assert list(cmds[0].qubits) == [0, 1]
    assert list(cmds[1].qubits) == [2, 3]
    assert list(cmds[2].qubits) == [1, 2]
    assert list(cmds[3].qubits) == [3, 0]


def test_brickwork_n_qubits():
    block = BrickworkEntanglingBlock(n_qubits=6, circular=False, use_cz=False)
    block.build()
    assert block.n_qubits == 6


@pytest.mark.parametrize("n_qubits", [3, 4, 5, 6, 7])
def test_brickwork_covers_every_qubit(n_qubits):
    """Every qubit takes part in at least one entangler, at both parities.

    Regression: ``n_qubits // 2 - 1`` dropped the final odd-layer pair for odd
    ``n_qubits``, leaving the highest qubit unentangled (n=5 produced only
    (0,1),(2,3),(1,2) and never touched qubit 4).
    """
    block = BrickworkEntanglingBlock(n_qubits=n_qubits, circular=False, use_cz=False)
    block.build()
    touched = {q for cmd in block.commands() for q in cmd.qubits}
    assert touched == set(range(n_qubits))


@pytest.mark.parametrize(
    ("n_qubits", "expected"),
    [
        (3, [(0, 1), (1, 2)]),
        (5, [(0, 1), (2, 3), (1, 2), (3, 4)]),
        (7, [(0, 1), (2, 3), (4, 5), (1, 2), (3, 4), (5, 6)]),
    ],
)
def test_brickwork_odd_n_qubits_pairs(n_qubits, expected):
    """Odd registers emit the full even layer followed by the full odd layer."""
    block = BrickworkEntanglingBlock(n_qubits=n_qubits, circular=False, use_cz=False)
    block.build()
    assert [tuple(cmd.qubits) for cmd in block.commands()] == expected


# ── Circular wrap: only a real ring appends the wrap edge ────────────────


@pytest.mark.parametrize("use_cz", [True, False])
def test_brickwork_two_qubit_ring_is_the_two_qubit_chain(use_cz):
    """Regression: at n=2 the wrap duplicated the chain's only edge, and for
    CZ the pair cancelled to an identity entangler (CZ² = 1)."""
    block = BrickworkEntanglingBlock(n_qubits=2, circular=True, use_cz=use_cz)
    block.build()
    assert [tuple(cmd.qubits) for cmd in block.commands()] == [(0, 1)]


def test_brickwork_two_qubit_circular_cz_is_not_the_identity():
    """The incident's observable, against the hand-written CZ matrix."""
    import numpy as np

    from tests.test_emit.conftest import qarpx_unitary

    block = BrickworkEntanglingBlock(n_qubits=2, circular=True, use_cz=True)
    block.build()
    u = qarpx_unitary(block.flatten(), 2)
    assert not np.allclose(u, np.eye(4))
    np.testing.assert_allclose(u, np.diag([1, 1, 1, -1]), atol=1e-12)


def test_brickwork_one_qubit_ring_has_no_edges():
    """n=1 circular used to append a (0, 0) self-pair."""
    block = BrickworkEntanglingBlock(n_qubits=1, circular=True, use_cz=True)
    block.build()
    assert list(block.commands()) == []


def test_brickwork_three_qubit_ring_keeps_the_wrap():
    """The guard must not trim real rings."""
    block = BrickworkEntanglingBlock(n_qubits=3, circular=True, use_cz=False)
    block.build()
    assert [tuple(cmd.qubits) for cmd in block.commands()] == [(0, 1), (1, 2), (2, 0)]
