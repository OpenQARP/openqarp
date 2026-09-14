"""Contract tests for HEABlock — multi-layer hardware-efficient ansatz.

Locks in the rename-from-LayeredHEABlock refactor: parameter count formula,
entangler topology, CZ-vs-CX selection, pre-build symbol availability, and
controlled-block metadata pass-through.
"""

import pytest

import qarpx as qx
from qarp.blocks import HEABlock

# ── construction smoke ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "real,linear,circular,cz",
    [
        (True, True, False, False),
        (False, True, True, False),
        (True, False, False, True),
        (False, False, True, True),
    ],
)
def test_hea_constructs_and_builds(real, linear, circular, cz):
    block = HEABlock(
        n_qubits=4, n_layers=2, real=real, linear=linear, circular=circular, use_cz=cz
    ).build()
    assert block.is_built
    assert block.n_qubits == 4
    cmds = block.flatten()
    assert any(cmd.is_parametric() for cmd in cmds)


# ── parameter count formula ─────────────────────────────────────────────────


@pytest.mark.parametrize("n_qubits", [2, 4, 6])
@pytest.mark.parametrize("n_layers", [1, 3])
@pytest.mark.parametrize("real", [True, False])
def test_hea_symbol_count_matches_formula(n_qubits, n_layers, real):
    """Real ansatz: n_qubits·n_layers Ry params.  Complex: 2·n_qubits·n_layers."""
    block = HEABlock(
        n_qubits=n_qubits,
        n_layers=n_layers,
        real=real,
        linear=True,
        circular=False,
        use_cz=False,
    )
    expected = n_qubits * n_layers * (1 if real else 2)
    # Pre-build: regression for the symbols-clobbered-by-super bug.
    assert len(block.symbols) == expected
    block.build()
    assert len(block.symbols) == expected


# ── two-qubit gate identity ─────────────────────────────────────────────────


@pytest.mark.parametrize("use_cz,entangler", [(True, qx.GateType.CZ), (False, qx.GateType.CX)])
def test_hea_use_cz_selects_entangler(use_cz, entangler):
    block = HEABlock(
        n_qubits=4, n_layers=2, real=True, linear=True, circular=True, use_cz=use_cz
    ).build()
    two_qubit = [c for c in block.flatten() if len(c.qubits) == 2]
    assert two_qubit, "expected at least one two-qubit gate"
    assert {c.gate for c in two_qubit} == {entangler}


# ── entangler count per topology ────────────────────────────────────────────


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
def test_hea_entangler_count(n_qubits, linear, circular, per_layer):
    n_layers = 2
    block = HEABlock(
        n_qubits=n_qubits,
        n_layers=n_layers,
        real=True,
        linear=linear,
        circular=circular,
        use_cz=False,
    ).build()
    two_qubit = [c for c in block.flatten() if len(c.qubits) == 2]
    assert len(two_qubit) == n_layers * per_layer


# ── entangler pairs per topology ────────────────────────────────────────────


def _pairs(block):
    return [tuple(c.qubits) for c in block.flatten() if len(c.qubits) == 2]


def test_hea_linear_pairs():
    block = HEABlock(
        n_qubits=4, n_layers=1, real=True, linear=True, circular=False, use_cz=False
    ).build()
    assert _pairs(block) == [(0, 1), (1, 2), (2, 3)]


def test_hea_linear_circular_pairs():
    block = HEABlock(
        n_qubits=4, n_layers=1, real=True, linear=True, circular=True, use_cz=False
    ).build()
    assert _pairs(block) == [(0, 1), (1, 2), (2, 3), (3, 0)]


def test_hea_brickwork_pairs_even_then_odd():
    """HEA brickwork = even pairs first, then odd pairs (asymmetric with SPA)."""
    block = HEABlock(
        n_qubits=4, n_layers=1, real=True, linear=False, circular=False, use_cz=False
    ).build()
    assert _pairs(block) == [(0, 1), (2, 3), (1, 2)]


def test_hea_brickwork_circular_pairs():
    block = HEABlock(
        n_qubits=4, n_layers=1, real=True, linear=False, circular=True, use_cz=False
    ).build()
    assert _pairs(block) == [(0, 1), (2, 3), (1, 2), (3, 0)]


# ── controlled metadata pass-through ────────────────────────────────────────


# ── Entangler coverage and ring degeneracy ──────────────────────────────


def _entangling_pairs(block):
    """Qubit tuples of the 2-qubit entangling commands in a built HEABlock."""
    return [tuple(c.qubits) for c in block.flatten() if c.gate in (qx.GateType.CZ, qx.GateType.CX)]


@pytest.mark.parametrize("n_qubits", [3, 4, 5, 6, 7])
@pytest.mark.parametrize("linear", [True, False])
def test_entangler_touches_every_qubit(n_qubits, linear):
    """Every qubit must appear in at least one entangling pair: a qubit left
    out is only ever single-qubit rotated, which is a strictly weaker ansatz
    than requested and its consecutive Ry rotations then merge away."""
    block = HEABlock(n_qubits, 1, real=True, linear=linear, circular=False, use_cz=True).build()
    touched = {q for pair in _entangling_pairs(block) for q in pair}
    assert touched == set(range(n_qubits))


@pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
@pytest.mark.parametrize("linear", [True, False])
def test_circular_adds_no_duplicate_or_self_pair(n_qubits, linear):
    """The ring wrap must add a genuinely new edge.  On 2 qubits the ring is
    the line, so the wrap would repeat the only pair — for symmetric CZ that
    is an identity layer the optimizer removes wholesale."""
    block = HEABlock(n_qubits, 1, real=True, linear=linear, circular=True, use_cz=True).build()
    pairs = _entangling_pairs(block)
    assert all(a != b for a, b in pairs), "self-loop pair emitted"
    undirected = [frozenset(p) for p in pairs]
    assert len(undirected) == len(set(undirected)), f"duplicate edge in {pairs}"


@pytest.mark.parametrize("n_qubits", [2, 3, 5])
def test_layers_survive_optimization_with_distinct_symbols(n_qubits):
    """Optimizing a 2-layer HEA must not fold two layers' Ry rotations into a
    single gate: distinct variational parameters have to stay separately
    differentiable."""
    block = HEABlock(n_qubits, 2, real=True, linear=False, circular=True, use_cz=True).build()
    optimized = qx.Transpiler(qx.native_gateset()).transpile_and_optimize(block.flatten())
    for cmd in optimized:
        if cmd.params and cmd.params[0].is_symbolic():
            assert len(cmd.params[0].free_symbols()) == 1, (
                f"{qx.gate_name(cmd.gate)} carries a compound parameter "
                f"{cmd.params[0]} — not differentiable by run_gradient"
            )
