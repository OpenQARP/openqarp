"""Simulation fusion (simulation_fusion_plan.md): ``QarpSimulator`` fuses gates
into dense blocks of up to ``fusion_max_qubits`` qubits before dispatch.

Oracles: a gate-by-gate numpy statevector built here from the analytic
matrices of qarp_conventions.md §2–§3 (LSB embedding, ``exp(-iθP/2)``), and the
published H2/STO-3G FCI energy.  Bit-identity across widths is pinned in
addition, never instead.  ``QARP_FUSION_MAX_QUBITS`` is read once per process,
so the env-var rows run in subprocesses, as ``test_threading.py`` does.
"""

import hashlib
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import SimpleBlock

SEED = 4242
WIDTHS = (0, 1, 2, 3, 4)

# ── analytic oracle ──────────────────────────────────────────────────────────

_I = np.eye(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]])
_Z = np.diag([1.0, -1.0]).astype(complex)
_H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)


def _rot(pauli, theta):
    return np.cos(theta / 2) * _I - 1j * np.sin(theta / 2) * pauli


def _embed(gate, qubits, n):
    """2^n embedding of a 2^k gate, local bit b ↔ qubits[b], qubit 0 = LSB."""
    dim, k = 1 << n, len(qubits)
    out = np.zeros((dim, dim), dtype=complex)
    for j in range(dim):
        lj = sum(((j >> q) & 1) << b for b, q in enumerate(qubits))
        rest = j
        for q in qubits:
            rest &= ~(1 << q)
        for li in range(1 << k):
            i = rest
            for b, q in enumerate(qubits):
                if (li >> b) & 1:
                    i |= 1 << q
            out[i, j] = gate[li, lj]
    return out


_CX = np.array(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
)  # q0 control
_CZ = np.diag([1, 1, 1, -1]).astype(complex)
_SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def _oracle_state(ops, n):
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    for op in ops:
        name, args = op[0], op[1:]
        if name == "h":
            g, qs = _H, args
        elif name in ("rx", "ry", "rz"):
            g, qs = _rot({"rx": _X, "ry": _Y, "rz": _Z}[name], args[1]), (args[0],)
        elif name == "cx":
            g, qs = _CX, args
        elif name == "cz":
            g, qs = _CZ, args
        elif name == "swap":
            g, qs = _SWAP, args
        elif name == "rzz":
            zz = np.kron(_Z, _Z)
            g, qs = np.cos(args[2] / 2) * np.eye(4) - 1j * np.sin(args[2] / 2) * zz, args[:2]
        else:
            raise ValueError(name)
        psi = _embed(g, list(qs), n) @ psi
    return psi


def _random_ops(rng, n, length):
    ops = []
    for _ in range(length):
        kind = rng.integers(7)
        if kind == 0:
            ops.append(("h", int(rng.integers(n))))
        elif kind < 4:
            ops.append(
                (["rx", "ry", "rz"][kind - 1], int(rng.integers(n)), float(rng.uniform(-3, 3)))
            )
        else:
            a, b = rng.choice(n, size=2, replace=False)
            if kind == 4:
                ops.append(("cx", int(a), int(b)))
            elif kind == 5:
                ops.append(("cz", int(a), int(b)) if rng.integers(2) else ("swap", int(a), int(b)))
            else:
                ops.append(("rzz", int(a), int(b), float(rng.uniform(-3, 3))))
    return ops


def _brickwork_ops(rng, n, layers=4):
    """Even/odd brick pattern of two-qubit blocks.  Fusion needs the locality:
    a random-pair circuit of the same length shares far fewer qubits between
    neighbouring gates, folds less, and would not pin the speedup below."""
    ops = []
    for layer in range(layers):
        for a in range(layer % 2, n - 1, 2):
            b = a + 1
            for q in (a, b):
                for gate in ("rz", "ry", "rz"):
                    ops.append((gate, q, float(rng.uniform(0, 2 * np.pi))))
            ops.append(("cx", a, b))
            ops.append(("ry", a, float(rng.uniform(0, 2 * np.pi))))
            ops.append(("ry", b, float(rng.uniform(0, 2 * np.pi))))
            ops.append(("cx", a, b))
    return ops


def _block(ops, n, *, measure_all=False):
    blk = SimpleBlock(n)
    for op in ops:
        getattr(blk, op[0])(*op[1:])
    if measure_all:
        for q in range(n):
            blk.measure(q, q)
    blk.build()
    return blk


# ── oracle rows ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize("width", WIDTHS)
def test_fused_statevector_matches_analytic_oracle(width):
    """Every block width reproduces the gate-by-gate analytic state at 1e-12,
    global phase included, on random 7-qubit circuits."""
    rng = np.random.default_rng(SEED + width)
    n = 7
    sim = qx.QarpSimulator()
    sim.fusion_max_qubits = width
    sim.fusion_min_qubits = 0  # reach the dense path on a small register
    assert sim.fusion_max_qubits == width
    for _ in range(12):
        ops = _random_ops(rng, n, 60)
        got = np.asarray(sim.statevector(_block(ops, n).flatten(), n))
        np.testing.assert_allclose(got, _oracle_state(ops, n), atol=1e-12, rtol=0)


@pytest.mark.parametrize("width", (0, 1, 3))
def test_h2_energy_reproduces_fci_at_every_width(width):
    """The VQE workflow's H2/STO-3G energy (published FCI −1.1373060357533993
    Ha) is reached through the engine's simulator at every fusion width."""
    from qarp.algorithms import VQE
    from qarp.blocks import CompositeBlock, MappedONVStateBlock, UCCBlock
    from qarp.engines import QarpEngine
    from qarp.operators import FermionOperator, JordanWigner
    from tests.test_functional.test_vqe_workflow import H2_FCI, H2_INTEGRAL_TERMS

    fham = FermionOperator()
    for term, coeff in H2_INTEGRAL_TERMS:
        fham += FermionOperator(term, coeff)
    qham = JordanWigner().encode_operator(fham)
    onv = [1, 1, 0, 0]
    ansatz = CompositeBlock(
        [
            MappedONVStateBlock(occupation_number_vector=onv, mapping=JordanWigner()),
            UCCBlock(
                occupation_number_vector=onv, mapping=JordanWigner(), singles=True, doubles=True
            ),
        ],
        4,
    )
    ansatz.build()
    engine = QarpEngine()
    engine._sim.fusion_max_qubits = width
    engine._sim.fusion_min_qubits = 0
    vqe = VQE(
        operator=qham,
        ket=ansatz,
        gradient=True,
        initial_parameters=np.zeros(len(ansatz.symbols)),
        engine=engine,
        verbose=False,
    )
    vqe.suppress_success_message = True
    vqe.build()
    energy, _ = vqe.run()
    assert engine._sim.fusion_max_qubits == width
    assert abs(energy - H2_FCI) < 1e-8


# ── bit-identity across widths (in addition to the oracles above) ────────────


def _digest(result) -> str:
    payload = repr((sorted(result.counts.items()), [tuple(h) for h in result.cbit_history]))
    return hashlib.sha1(payload.encode()).hexdigest()


def test_seeded_sampling_is_identical_at_every_width():
    """Terminal-measurement fast path and the feed-forward trajectory path
    give the same seeded counts and cbit history at every width."""
    from tests.test_engines.test_threading import _feedforward

    rng = np.random.default_rng(SEED)
    n = 8
    terminal = _block(_random_ops(rng, n, 80), n, measure_all=True)
    feedforward = _feedforward(n)
    circuits = [terminal.flatten(), feedforward.flatten()]

    digests = {}
    for width in WIDTHS:
        sim = qx.QarpSimulator()
        sim.fusion_max_qubits = width
        sim.fusion_min_qubits = 0
        digests[width] = [_digest(sim.run(cmds, n, 500, SEED)) for cmds in circuits]
    assert all(d == digests[0] for d in digests.values())


# ── knob and env var ─────────────────────────────────────────────────────────


def test_knob_bounds():
    sim = qx.QarpSimulator()
    assert sim.fusion_max_qubits == qx.QarpSimulator.DEFAULT_FUSION_QUBITS == 3
    assert sim.fusion_min_qubits == qx.QarpSimulator.DEFAULT_FUSION_MIN_QUBITS == 12
    sim.fusion_max_qubits = qx.QarpSimulator.MAX_FUSION_QUBITS
    with pytest.raises(ValueError):
        sim.fusion_max_qubits = qx.QarpSimulator.MAX_FUSION_QUBITS + 1
    assert sim.fusion_max_qubits == qx.QarpSimulator.MAX_FUSION_QUBITS
    sim.fusion_min_qubits = 0
    assert sim.fusion_min_qubits == 0


def test_narrow_registers_take_the_single_qubit_pass():
    """Below ``fusion_min_qubits`` the dense pass is skipped: the result is
    still the analytic state (the 1q pass is exact too), pinned here so the
    routing cannot silently disappear — the dense pass costs 2–4× at 8
    qubits (sweep of 2026-09-13)."""
    rng = np.random.default_rng(SEED + 99)
    n = 6
    ops = _random_ops(rng, n, 40)
    cmds = _block(ops, n).flatten()
    sim = qx.QarpSimulator()  # default: k = 3 from 12 qubits
    np.testing.assert_allclose(
        np.asarray(sim.statevector(cmds, n)), _oracle_state(ops, n), atol=1e-12, rtol=0
    )


_ENV_SCRIPT = textwrap.dedent(
    """
    import qarpx as qx
    print(qx.QarpSimulator().fusion_max_qubits)
    """
)


def _default_in_subprocess(value):
    env = {k: v for k, v in os.environ.items() if k != "QARP_FUSION_MAX_QUBITS"}
    env["SKBUILD_EDITABLE_VERBOSE"] = "0"
    if value is not None:
        env["QARP_FUSION_MAX_QUBITS"] = value
    out = subprocess.run(
        [sys.executable, "-c", _ENV_SCRIPT], check=True, capture_output=True, text=True, env=env
    )
    return int(out.stdout.strip())


def test_env_var_sets_the_constructor_default():
    """``QARP_FUSION_MAX_QUBITS`` is the process-wide default; an unparsable
    or out-of-range value is ignored like an invalid ``QARP_NUM_THREADS``."""
    assert _default_in_subprocess(None) == qx.QarpSimulator.DEFAULT_FUSION_QUBITS
    assert _default_in_subprocess("0") == 0
    assert _default_in_subprocess("3") == 3
    for bad in ("many", "-1", str(qx.QarpSimulator.MAX_FUSION_QUBITS + 1), "2.5"):
        assert _default_in_subprocess(bad) == qx.QarpSimulator.DEFAULT_FUSION_QUBITS, bad


def test_transpiler_o1_output_is_unchanged():
    """The transpiler's O1 pass keeps its 1-qubit Custom product: fusion into
    wider blocks is simulator-internal (§16 rebase totality)."""
    rng = np.random.default_rng(SEED)
    n = 5
    blk = _block(_random_ops(rng, n, 40), n)
    for cmd in blk.optimize(qx.native_gateset(), level=1).flatten():
        if cmd.gate == qx.GateType.Custom:
            assert len(cmd.qubits) == 1


# ── bench ────────────────────────────────────────────────────────────────────


@pytest.mark.bench
def test_fusion_speedup_at_twenty_qubits():
    """Plan row: brickwork n = 20 run phase at the default width ≤ 0.6× the
    single-qubit-fusion time (measured 2026-09-13: 0.19×, 39 ms vs 207 ms)."""
    n = 20
    cmds = _block(_brickwork_ops(np.random.default_rng(SEED), n), n).flatten()

    def run_phase(width):
        sim = qx.QarpSimulator()
        sim.fusion_max_qubits = width
        sim.statevector(cmds, n)
        t = time.perf_counter()
        for _ in range(3):
            sim.statevector(cmds, n)
        return (time.perf_counter() - t) / 3

    assert run_phase(qx.QarpSimulator.DEFAULT_FUSION_QUBITS) < 0.6 * run_phase(1)
