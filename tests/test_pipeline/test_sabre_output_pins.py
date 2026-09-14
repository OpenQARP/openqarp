"""SABRE's routed output is pinned bit for bit (pipeline_hardening_plan.md
P2.10): the inner-loop hoists — stamp buffers, sorted-insert ready list,
generation-stamped decay, incremental candidate scoring — must change no
tie-break, no SWAP and no mapping.

The goldens were captured on the pre-P2.10 router (2026-09-13, at
e498fc3) over random CX/Rz circuits on a line, a grid and an all-to-all
device, two seeds each, undirected and directed.  A future *intentional*
change to the heuristic re-records them — and says so in its plan.
"""

import hashlib
import time

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock
from qarp.devices import get_all_to_all_architecture, get_nearest_neighbour_architecture

# (digest of the routed command stream, initial l2p, final l2p)
GOLDEN = {
    "line8/s0/d0": ("a2867643eeb31b00", [3, 2, 1, 6, 7, 5, 0, 4], [5, 1, 7, 4, 0, 6, 2, 3]),
    "line8/s0/d1": ("fa882f1ddb821ec0", [6, 2, 3, 1, 0, 4, 7, 5], [2, 6, 0, 3, 7, 1, 5, 4]),
    "line8/s1/d0": ("e6d7b1e57fd7bd6d", [2, 5, 4, 6, 7, 1, 3, 0], [7, 4, 6, 5, 2, 3, 1, 0]),
    "line8/s1/d1": ("7f95d58f5d69593c", [2, 5, 4, 6, 7, 1, 3, 0], [7, 4, 6, 5, 2, 3, 1, 0]),
    "grid3x3/s0/d0": ("b71e924009872e5d", [1, 5, 0, 8, 6, 4, 7, 2, 3], [4, 2, 6, 3, 5, 0, 1, 7, 8]),
    "grid3x3/s0/d1": ("5b335ab3a4cbb0a5", [1, 5, 0, 8, 6, 4, 7, 2, 3], [4, 2, 6, 3, 5, 0, 1, 7, 8]),
    "grid3x3/s1/d0": ("99d08be105322f84", [0, 8, 6, 5, 2, 7, 4, 3, 1], [4, 2, 8, 6, 1, 3, 5, 7, 0]),
    "grid3x3/s1/d1": ("10f8db6d6e968e6e", [0, 8, 6, 5, 2, 7, 4, 3, 1], [4, 2, 8, 6, 1, 3, 5, 7, 0]),
    "a2a6/s0/d0": ("0790cc58dd048343", [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
    "a2a6/s0/d1": ("052feb31e8b016ea", [4, 2, 3, 0, 5, 1], [4, 2, 3, 0, 5, 1]),
    "a2a6/s1/d0": ("5eb1ee7329e02d8a", [0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]),
    "a2a6/s1/d1": ("38f35ad6e14c9c41", [1, 4, 0, 3, 5, 2], [1, 4, 0, 3, 5, 2]),
}

_DEVICES = {
    "line8": (lambda: get_nearest_neighbour_architecture(8, 1), 8, 120),
    "grid3x3": (lambda: get_nearest_neighbour_architecture(3, 3), 9, 200),
    "a2a6": (lambda: get_all_to_all_architecture(6), 6, 100),
}


def _circuit(n, n_ops, seed):
    rng = np.random.default_rng(seed)
    b = SimpleBlock(n)
    for i in range(n_ops):
        a, c = rng.choice(n, 2, replace=False)
        if i % 3 == 0:
            b.rz(int(a), 0.1 * i)
        b.cx(int(a), int(c))
    return b.build().flatten()


def _digest(cmds):
    h = hashlib.sha1()
    for c in cmds:
        h.update(f"{c.gate}{list(c.qubits)}{[str(p) for p in c.params]};".encode())
    return h.hexdigest()[:16]


@pytest.mark.parametrize("key", sorted(GOLDEN))
def test_sabre_output_is_bit_identical_to_the_golden(key):
    label, seed, directed = key.split("/")
    make_arch, n, n_ops = _DEVICES[label]
    opts = qx.RoutingOptions()
    opts.arch = make_arch()
    opts.router = qx.RouterKind.Sabre
    opts.directedness = directed == "d1"
    result = qx.route(_circuit(n, n_ops, int(seed[1:])), opts)
    digest, initial, final = GOLDEN[key]
    assert _digest(result.commands) == digest
    assert list(result.initial_logical_to_physical) == initial
    assert list(result.final_logical_to_physical) == final


@pytest.mark.bench
def test_sabre_line20_1900_ops_wall_clock():
    """The plan's row: 1900 random CX on a 20-qubit line.  2.55 s before
    P2.10, 1.32 s after (M-series laptop; the 32-trial portfolio is the
    cost — 288 full routings per call).  Bound at 2 s."""
    n, n_ops = 20, 1900
    rng = np.random.default_rng(0)
    b = SimpleBlock(n)
    for _ in range(n_ops):
        a, c = rng.choice(n, 2, replace=False)
        b.cx(int(a), int(c))
    cmds = b.build().flatten()
    opts = qx.RoutingOptions()
    opts.arch = get_nearest_neighbour_architecture(n, 1)
    opts.router = qx.RouterKind.Sabre
    t = time.perf_counter()
    qx.route(cmds, opts)
    elapsed = time.perf_counter() - t
    assert elapsed < 2.0, f"{elapsed:.2f} s"
