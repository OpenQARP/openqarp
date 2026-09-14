"""Verify scalar and variadic Block builder methods produce identical output.

The variadic overloads (`b.h([0,1,2])`, `b.cx([(0,1),(2,3)])`,
`b.rz([(q, θ_q) for q in qubits])`, …) are the bulk-emission API that
collapses N Python→C++ crossings into one nanobind dispatch.  This file
locks in the contract: bulk output ≡ N scalar calls.
"""

import qarpx as qx


def _cmd_signature(b):
    return [(c.gate.name, list(c.qubits)) for c in b.commands()]


# ── 1Q no-param ──────────────────────────────────────────────────────────


def test_one_qubit_noparam_overloads():
    for fn in ["h", "x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "id"]:
        s = qx.SimpleBlock(3, "s")
        for q in range(3):
            getattr(s, fn)(q)
        b = qx.SimpleBlock(3, "b")
        getattr(b, fn)([0, 1, 2])
        assert _cmd_signature(s) == _cmd_signature(b), f"{fn} mismatch"


# ── 1Q parametric ───────────────────────────────────────────────────────


def test_one_qubit_param_overloads():
    angles = [0.1, 0.2, 0.3]
    for fn in ["rx", "ry", "rz", "p"]:
        s = qx.SimpleBlock(3, "s")
        for q, a in enumerate(angles):
            getattr(s, fn)(q, a)
        b = qx.SimpleBlock(3, "b")
        getattr(b, fn)(list(zip(range(3), angles, strict=True)))
        assert _cmd_signature(s) == _cmd_signature(b), f"{fn} mismatch"


# ── 2Q no-param ─────────────────────────────────────────────────────────


def test_two_qubit_noparam_overloads():
    pairs = [(0, 1), (1, 2), (2, 3)]
    for fn in [
        "cx",
        "cy",
        "cz",
        "swap",
        "ecr",
        "iswap",
        "iswapdg",
        "ch",
        "cs",
        "csdg",
        "csx",
        "csxdg",
    ]:
        s = qx.SimpleBlock(4, "s")
        for c, t in pairs:
            getattr(s, fn)(c, t)
        b = qx.SimpleBlock(4, "b")
        getattr(b, fn)(pairs)
        assert _cmd_signature(s) == _cmd_signature(b), f"{fn} mismatch"


# ── 2Q parametric ───────────────────────────────────────────────────────


def test_two_qubit_param_overloads():
    triples = [(0, 1, 0.1), (1, 2, 0.2)]
    for fn in ["crx", "cry", "crz", "cp", "rzz", "rxx", "ryy"]:
        s = qx.SimpleBlock(3, "s")
        for c, t, a in triples:
            getattr(s, fn)(c, t, a)
        b = qx.SimpleBlock(3, "b")
        getattr(b, fn)(triples)
        assert _cmd_signature(s) == _cmd_signature(b), f"{fn} mismatch"


# ── 3Q ──────────────────────────────────────────────────────────────────


def test_three_qubit_overloads():
    triples = [(0, 1, 2), (1, 2, 3)]
    for fn in ["ccx", "cswap"]:
        s = qx.SimpleBlock(4, "s")
        for a, b_, c in triples:
            getattr(s, fn)(a, b_, c)
        b = qx.SimpleBlock(4, "b")
        getattr(b, fn)(triples)
        assert _cmd_signature(s) == _cmd_signature(b), f"{fn} mismatch"


# ── Measure / Reset ─────────────────────────────────────────────────────


def test_measure_overloads():
    pairs = [(0, 0), (1, 1), (2, 2)]
    s = qx.SimpleBlock(3, "s")
    s.n_cbits = 3
    for q, c in pairs:
        s.measure(q, c)
    b = qx.SimpleBlock(3, "b")
    b.n_cbits = 3
    b.measure(pairs)
    assert _cmd_signature(s) == _cmd_signature(b)


def test_reset_overloads():
    s = qx.SimpleBlock(3, "s")
    for q in range(3):
        s.reset(q)
    b = qx.SimpleBlock(3, "b")
    b.reset([0, 1, 2])
    assert _cmd_signature(s) == _cmd_signature(b)


# ── End-to-end: bulk-built block simulates equivalently ──────────────────


def test_bulk_built_block_simulates_to_same_unitary():
    """Build a 3-qubit GHZ-like circuit via scalar and bulk; the simulator
    output should be identical."""
    import numpy as np

    s = qx.SimpleBlock(3, "s")
    s.h(0)
    for c, t in [(0, 1), (1, 2)]:
        s.cx(c, t)
    s.build()

    b = qx.SimpleBlock(3, "b")
    b.h([0])
    b.cx([(0, 1), (1, 2)])
    b.build()

    sim = qx.QarpSimulator()
    Us = np.array(sim.unitary_matrix(s.flatten(), 3))
    Ub = np.array(sim.unitary_matrix(b.flatten(), 3))
    assert np.allclose(Us, Ub)
