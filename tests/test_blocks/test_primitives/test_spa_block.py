"""Contract tests for SPABlock — multi-layer separable-pair ansatz.

Locks in the rename-from-LayeredSPABlock refactor: RSP-vs-A-gate selection,
pair count per topology, brickwork odd-then-even ordering (asymmetric with
HEA's even-then-odd), and controlled-block metadata pass-through.
"""

import pytest

from qarp.blocks import AGateBlock, RSPBlock, SPABlock

# ── construction smoke ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "real,linear,circular",
    [
        (True, True, False),
        (False, True, True),
        (True, False, False),
        (False, False, True),
    ],
)
def test_spa_constructs_and_builds(real, linear, circular):
    block = SPABlock(n_qubits=4, n_layers=2, real=real, linear=linear, circular=circular).build()
    assert block.is_built
    assert block.n_qubits == 4
    cmds = block.flatten()
    assert any(cmd.is_parametric() for cmd in cmds)


# ── child-block type matches `real` ─────────────────────────────────────────


@pytest.mark.parametrize(
    "real,child_cls",
    [(True, RSPBlock), (False, AGateBlock)],
)
def test_spa_real_selects_child_block(real, child_cls):
    """real=True ⇒ RSP entanglers (real-valued); real=False ⇒ A-gate entanglers."""
    block = SPABlock(n_qubits=4, n_layers=2, real=real, linear=True, circular=False)
    assert block.blocks, "expected child blocks to be populated at construction"
    assert all(isinstance(c, child_cls) for c in block.blocks)


# ── pair count per topology ─────────────────────────────────────────────────


@pytest.mark.parametrize(
    "n_qubits,linear,circular,per_layer",
    [
        # linear: n-1 pairs (+1 if circular).
        (4, True, False, 3),
        (4, True, True, 4),
        (6, True, False, 5),
        (6, True, True, 6),
        # brickwork: n//2 even + n//2-1 odd = n-1 pairs (+1 if circular).
        (4, False, False, 3),
        (4, False, True, 4),
        (6, False, False, 5),
        (6, False, True, 6),
    ],
)
def test_spa_pair_count(n_qubits, linear, circular, per_layer):
    n_layers = 2
    block = SPABlock(
        n_qubits=n_qubits, n_layers=n_layers, real=True, linear=linear, circular=circular
    )
    assert len(block.blocks) == n_layers * per_layer


# ── pair indices per topology ───────────────────────────────────────────────


def _pairs(block):
    return [tuple(c.target_qubits) for c in block.blocks]


def test_spa_linear_pairs():
    block = SPABlock(n_qubits=4, n_layers=1, real=True, linear=True, circular=False)
    assert _pairs(block) == [(0, 1), (1, 2), (2, 3)]


def test_spa_linear_circular_pairs():
    block = SPABlock(n_qubits=4, n_layers=1, real=True, linear=True, circular=True)
    assert _pairs(block) == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_spa_brickwork_pairs_odd_then_even():
    """SPA brickwork = odd pairs first, then even pairs.

    NB: this is the *opposite* order from HEABlock brickwork (even-then-odd).
    The asymmetry is the current contract — change this test together with
    the source if you intend to harmonise the two blocks.
    """
    block = SPABlock(n_qubits=4, n_layers=1, real=True, linear=False, circular=False)
    assert _pairs(block) == [(1, 2), (0, 1), (2, 3)]


def test_spa_brickwork_circular_pairs():
    block = SPABlock(n_qubits=4, n_layers=1, real=True, linear=False, circular=True)
    assert _pairs(block) == [(1, 2), (0, 1), (2, 3), (3, 0)]


# ── controlled metadata pass-through ────────────────────────────────────────
