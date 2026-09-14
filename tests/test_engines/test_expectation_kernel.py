"""P3.1: the C++ Pauli expectation kernel on ``QarpSimulator`` and its
``StateVector`` dispatch.

Oracles (§18): the dense contraction ``vdot(bra, H.sparse_matrix(n) @ ket)``
(P2.1's flip-mask COO builder, which shares no code with the streaming
kernel), analytic values on product / GHZ states, and the H2 FCI energy.
The numpy sweep ``pauli_expectation`` is the *fallback* under test, not an
oracle: it appears in the fallback-routing test and, alongside the kernel,
as a subject of the property test's first-principles ``kron`` oracle.
"""

import os
import subprocess
import sys

import numpy as np
import pytest
from hypothesis import example, given

import qarpx as qx
from qarp.algorithms import VQE, StateVector, Target
from qarp.blocks import ComputationalBasisStateBlock, HEABlock, HnBlock, SimpleBlock
from qarp.engines import QarpEngine
from qarp.operators import FermionOperator, JordanWigner, QubitOperator
from tests.operator_test_utils import pauli_matrix_lsb
from tests.strategies import pauli_sums

RTOL = 1e-12


def _random_operator(rng, n_qubits, n_terms, *, max_locality=4, complex_coeffs=False):
    op = QubitOperator()
    for _ in range(n_terms):
        k = int(rng.integers(1, min(max_locality, n_qubits) + 1))
        qubits = sorted(int(q) for q in rng.choice(n_qubits, size=k, replace=False))
        term = tuple((q, "XYZ"[int(rng.integers(3))]) for q in qubits)
        coeff = complex(rng.normal(), rng.normal() if complex_coeffs else 0.0)
        op += QubitOperator(term, coeff)
    return op


def _random_state(rng, n_qubits):
    psi = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return psi / np.linalg.norm(psi)


def _dense(bra, ket, op, n_qubits):
    return complex(np.vdot(bra, op.sparse_matrix(n_qubits) @ ket))


def _observable(op):
    return qx.qubit_operator_to_observable(op)


def _chemistry_operator(rng, n_qubits, n_two_body):
    """JW image of random symmetric one-body + paired two-body integrals —
    the x-mask-grouped structure the kernel is designed around, built with
    qarp's own mapping layer."""
    h1 = rng.normal(size=(n_qubits, n_qubits))
    h1 = (h1 + h1.T) / 2
    fop = FermionOperator()
    for p in range(n_qubits):
        for q in range(n_qubits):
            fop += FermionOperator(((p, 1), (q, 0)), float(h1[p, q]))
    seen = set()
    while len(seen) < n_two_body:
        p, q, r, s = (int(v) for v in rng.choice(n_qubits, size=4, replace=False))
        if (p, q, r, s) in seen:
            continue
        seen.add((p, q, r, s))
        c = float(rng.normal())
        fop += FermionOperator(((p, 1), (q, 1), (r, 0), (s, 0)), c)
        fop += FermionOperator(((s, 1), (r, 1), (q, 0), (p, 0)), c)
    return JordanWigner().encode_operator(fop)


# ── transition: dense oracle ────────────────────────────────────────────────


@pytest.mark.parametrize("n_qubits", [1, 3, 5, 6, 7, 10])
def test_transition_matches_dense_contraction(n_qubits):
    """Random ≤4-local operators with complex coefficients and bra ≠ ket
    (non-Hermitian): both the sub-block (n < 6) and blocked kernels."""
    rng = np.random.default_rng(n_qubits)
    op = _random_operator(rng, n_qubits, 60, complex_coeffs=True)
    bra, ket = _random_state(rng, n_qubits), _random_state(rng, n_qubits)
    got = qx.QarpSimulator().transition(bra, ket, n_qubits, _observable(op))
    want = _dense(bra, ket, op, n_qubits)
    assert got == pytest.approx(want, rel=RTOL, abs=RTOL)


def test_transition_idle_qubits_wider_than_operator():
    rng = np.random.default_rng(7)
    n_qubits = 8
    op = _random_operator(rng, 3, 20)  # acts on qubits 0..2 only
    bra, ket = _random_state(rng, n_qubits), _random_state(rng, n_qubits)
    got = qx.QarpSimulator().transition(bra, ket, n_qubits, _observable(op))
    assert got == pytest.approx(_dense(bra, ket, op, n_qubits), rel=RTOL, abs=RTOL)


def test_transition_group_with_mixed_y_parity():
    """Two terms sharing the flip mask {0, 1} but with one and two Y's: the
    ``i^{n_Y}`` factor is per term, not per group (a hoisted factor would
    pass every single-mask test and fail here)."""
    rng = np.random.default_rng(3)
    n_qubits = 7
    op = QubitOperator(((0, "Y"), (1, "X")), 0.7) + QubitOperator(((0, "Y"), (1, "Y")), -1.3)
    bra, ket = _random_state(rng, n_qubits), _random_state(rng, n_qubits)
    got = qx.QarpSimulator().transition(bra, ket, n_qubits, _observable(op))
    assert got == pytest.approx(_dense(bra, ket, op, n_qubits), rel=RTOL, abs=RTOL)


def test_transition_identity_and_empty_operator():
    rng = np.random.default_rng(5)
    n_qubits = 6
    bra, ket = _random_state(rng, n_qubits), _random_state(rng, n_qubits)
    sim = qx.QarpSimulator()
    ident = QubitOperator((), 2.5 - 0.5j)
    assert sim.transition(bra, ket, n_qubits, _observable(ident)) == pytest.approx(
        (2.5 - 0.5j) * np.vdot(bra, ket), rel=RTOL
    )
    assert sim.transition(bra, ket, n_qubits, []) == 0


def test_transition_unnormalised_inputs_are_bilinear():
    rng = np.random.default_rng(11)
    n_qubits = 6
    op = _random_operator(rng, n_qubits, 30)
    bra, ket = _random_state(rng, n_qubits), _random_state(rng, n_qubits)
    sim = qx.QarpSimulator()
    base = sim.transition(bra, ket, n_qubits, _observable(op))
    scaled = sim.transition(2.0 * bra, (0.5 + 0.5j) * ket, n_qubits, _observable(op))
    assert scaled == pytest.approx(2.0 * (0.5 + 0.5j) * base, rel=RTOL)


def test_transition_rejects_bad_inputs():
    sim = qx.QarpSimulator()
    psi = np.zeros(8, dtype=complex)
    with pytest.raises(ValueError, match="out of range"):
        sim.transition(psi, psi, 3, [([(3, "Z")], 1.0)])
    with pytest.raises(ValueError, match="repeated"):
        sim.transition(psi, psi, 3, [([(0, "X"), (0, "Z")], 1.0)])
    with pytest.raises(ValueError, match="invalid Pauli"):
        sim.transition(psi, psi, 3, [([(0, "Q")], 1.0)])
    with pytest.raises(ValueError, match="do not match"):
        sim.transition(psi, psi[:4], 3, [([(0, "Z")], 1.0)])


@pytest.mark.parametrize("n_qubits", [-1, 63, 64, 70])
def test_transition_rejects_out_of_range_width(n_qubits):
    """The width is validated before ``1 << n_qubits`` is formed for the
    length check: that shift is undefined outside [0, 62], and on arm64
    ``n = 64`` wrapped to a length of 1 and let a one-element vector through
    (review A.3, 2026-09-13)."""
    sim = qx.QarpSimulator()
    one = np.ones(1, dtype=complex)
    with pytest.raises(ValueError, match=r"n_qubits must be in \[0, 62\]"):
        sim.transition(one, one, n_qubits, [([(0, "Z")], 1.0)])


# ── expectation / batch_expectation: analytic oracles ───────────────────────


def _hadamard_all(n_qubits):
    return HnBlock(n_qubits).build()


def _ghz(n_qubits):
    blk = SimpleBlock(n_qubits, name="ghz")
    blk.h(0)
    for q in range(1, n_qubits):
        blk.cx(0, q)
    return blk.build()


def test_expectation_on_product_states():
    """|+⟩^n: ⟨X⟩ = 1, ⟨Z⟩ = 0, ⟨Y⟩ = 0; |0110⟩: ⟨Z_q⟩ = ±1 by occupation."""
    sim = qx.QarpSimulator()
    n_qubits = 6
    plus = _hadamard_all(n_qubits).flatten()
    for q in range(n_qubits):
        assert sim.expectation(plus, n_qubits, [([(q, "X")], 1.0)]) == pytest.approx(1.0, abs=RTOL)
        assert sim.expectation(plus, n_qubits, [([(q, "Z")], 1.0)]) == pytest.approx(0.0, abs=RTOL)
        assert sim.expectation(plus, n_qubits, [([(q, "Y")], 1.0)]) == pytest.approx(0.0, abs=RTOL)
    bits = [0, 1, 1, 0, 1, 0]
    basis = ComputationalBasisStateBlock(bits).build().flatten()
    for q, b in enumerate(bits):
        assert sim.expectation(basis, n_qubits, [([(q, "Z")], 1.0)]) == pytest.approx(
            1.0 - 2.0 * b, abs=RTOL
        )


def test_expectation_on_ghz():
    """GHZ_n: ⟨X⊗…⊗X⟩ = 1, ⟨Z_i Z_j⟩ = 1, ⟨Z_i⟩ = 0, ⟨Y_0 Y_1 X_2…X_{n-1}⟩ = −1."""
    sim = qx.QarpSimulator()
    n_qubits = 7
    ghz = _ghz(n_qubits).flatten()
    x_all = [([(q, "X") for q in range(n_qubits)], 1.0)]
    assert sim.expectation(ghz, n_qubits, x_all) == pytest.approx(1.0, abs=RTOL)
    assert sim.expectation(ghz, n_qubits, [([(2, "Z"), (5, "Z")], 1.0)]) == pytest.approx(
        1.0, abs=RTOL
    )
    assert sim.expectation(ghz, n_qubits, [([(4, "Z")], 1.0)]) == pytest.approx(0.0, abs=RTOL)
    yyx = [([(0, "Y"), (1, "Y")] + [(q, "X") for q in range(2, n_qubits)], 1.0)]
    assert sim.expectation(ghz, n_qubits, yyx) == pytest.approx(-1.0, abs=RTOL)


def test_expectation_y_on_plus_i_state():
    """|+i⟩ = S H |0⟩ has ⟨Y⟩ = +1 — the sign of the Y phase convention."""
    blk = SimpleBlock(1, name="plus_i")
    blk.h(0)
    blk.s(0)
    cmds = blk.build().flatten()
    assert qx.QarpSimulator().expectation(cmds, 1, [([(0, "Y")], 1.0)]) == pytest.approx(
        1.0, abs=RTOL
    )


def test_expectation_with_initial_state_matches_transition():
    rng = np.random.default_rng(21)
    n_qubits = 6
    op = _random_operator(rng, n_qubits, 40)
    psi0 = _random_state(rng, n_qubits)
    hea = HEABlock(n_qubits, 1, real=True, linear=True, circular=False, use_cz=False).build()
    hea = hea.set_symbols({s: 0.2 for s in hea.symbols})
    cmds = hea.flatten()
    sim = qx.QarpSimulator()
    psi = np.asarray(sim.statevector(cmds, n_qubits, initial_state=psi0))
    got = sim.expectation(cmds, n_qubits, _observable(op), initial_state=psi0)
    assert got == pytest.approx(_dense(psi, psi, op, n_qubits).real, rel=RTOL)


def test_batch_expectation_is_one_value_per_parameter_set():
    """Same contract as ``CudaqSimulator.batch_expectation``: a float per
    set, each equal to ``expectation`` on the substituted circuit."""
    rng = np.random.default_rng(4)
    n_qubits = 5
    op = _random_operator(rng, n_qubits, 25)
    hea = HEABlock(n_qubits, 2, real=True, linear=True, circular=False, use_cz=False).build()
    names = [str(s) for s in hea.symbols]
    sets = [{name: float(rng.uniform(-3, 3)) for name in names} for _ in range(6)]
    sim = qx.QarpSimulator()
    batch = sim.batch_expectation(hea.flatten(), n_qubits, _observable(op), sets)
    assert len(batch) == 6
    for values, got in zip(sets, batch, strict=True):
        concrete = hea.set_symbols(dict(zip(hea.symbols, [values[n] for n in names], strict=True)))
        psi = np.asarray(sim.statevector(concrete.flatten(), n_qubits))
        assert got == pytest.approx(_dense(psi, psi, op, n_qubits).real, rel=RTOL)


def test_thread_count_changes_only_rounding():
    """The OpenMP reduction order follows the team size: results agree to
    1e-12, and bit-identity across thread counts is deliberately *not*
    claimed (Deviations, P3.1)."""
    rng = np.random.default_rng(9)
    n_qubits, n_terms = 12, 300
    op = _random_operator(rng, n_qubits, n_terms)
    psi = _random_state(rng, n_qubits)
    here = qx.QarpSimulator().transition(psi, psi, n_qubits, _observable(op))
    script = (
        "import sys, numpy as np, qarpx as qx\n"
        "from tests.test_engines.test_expectation_kernel import _random_operator, _random_state\n"
        f"rng = np.random.default_rng(9); op = _random_operator(rng, {n_qubits}, {n_terms})\n"
        f"psi = _random_state(rng, {n_qubits})\n"
        f"v = qx.QarpSimulator().transition(psi, psi, {n_qubits}, "
        "qx.qubit_operator_to_observable(op))\n"
        "print(repr(v.real))\n"
    )
    env = {**os.environ, "QARP_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}
    out = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True
    )
    single = float(out.stdout.strip().splitlines()[-1])
    assert single == pytest.approx(here.real, rel=RTOL, abs=RTOL)


# ── property: every path against a first-principles kron oracle ─────────────


@pytest.mark.property
@example(problem=(1, [], 0))  # empty sum, one qubit
@example(
    problem=(6, [({0: "Y", 1: "X"}, 1.0), ({0: "Y", 1: "Y"}, -1.0)], 1)
)  # mixed n_Y group, blocked
@example(problem=(3, [({}, 1.5 - 0.5j)], 2))  # identity term below one block
@example(problem=(5, [({4: "Z"}, 1.0), ({4: "Z"}, 1.0)], 3))  # repeated term must add
@given(problem=pauli_sums())
def test_kernel_matches_kron_oracle_on_every_path(problem):
    """The C++ ``transition``, ``expectation`` and the numpy fallback all
    equal ``⟨bra| Σ c_t P_t |ket⟩`` built from explicit ``kron`` products
    (``pauli_matrix_lsb``) — an oracle that shares neither the mask grouping
    nor the ``sparse_matrix`` COO path.  Draws random states from a seed, not
    entries, so shrinking stays inside valid inputs."""
    from qarp.algorithms._primitives.state_vector import pauli_expectation
    from tests.strategies import render_pauli_term

    n_qubits, terms, seed = problem
    rng = np.random.default_rng(seed)
    bra, ket = _random_state(rng, n_qubits), _random_state(rng, n_qubits)
    dim = 2**n_qubits

    h = np.zeros((dim, dim), dtype=complex)
    op = QubitOperator()
    for term_map, coeff in terms:
        h += coeff * pauli_matrix_lsb(term_map, n_qubits)
        op += QubitOperator(render_pauli_term(term_map), coeff)
    want = complex(np.vdot(bra, h @ ket))
    obs = _observable(op)

    sim = qx.QarpSimulator()
    assert sim.transition(bra, ket, n_qubits, obs) == pytest.approx(want, abs=1e-12)
    assert pauli_expectation(bra, ket, op, n_qubits) == pytest.approx(want, abs=1e-12)
    # expectation: the real part of ⟨ψ|H|ψ⟩ from the circuit's own state —
    # seed the register with |ket⟩ through initial_state and no gates.
    want_ev = complex(np.vdot(ket, h @ ket)).real
    assert sim.expectation([], n_qubits, obs, initial_state=ket) == pytest.approx(
        want_ev, abs=1e-12
    )


# ── StateVector dispatch ────────────────────────────────────────────────────


def test_state_vector_uses_kernel_and_matches_dense():
    rng = np.random.default_rng(13)
    n_qubits = 6
    op = _random_operator(rng, n_qubits, 50)
    hea = HEABlock(n_qubits, 2, real=True, linear=True, circular=False, use_cz=False).build()
    hea = hea.set_symbols(
        {
            s: float(v)
            for s, v in zip(hea.symbols, rng.uniform(-1, 1, len(hea.symbols)), strict=True)
        }
    )
    sv = StateVector(operator=op, ket=hea)
    sv.build()
    assert sv._observable is not None  # kernel path selected at build
    engine = QarpEngine()
    engine.build([sv])
    energy = engine.run()[0]
    psi = np.asarray(qx.QarpSimulator().statevector(hea.flatten(), n_qubits))
    assert energy == pytest.approx(_dense(psi, psi, op, n_qubits), rel=RTOL, abs=RTOL)


def test_state_vector_transition_amplitude_uses_kernel():
    rng = np.random.default_rng(14)
    n_qubits = 6
    op = _random_operator(rng, n_qubits, 50, complex_coeffs=True)
    ket = HEABlock(n_qubits, 1, real=True, linear=True, circular=False, use_cz=False).build()
    ket = ket.set_symbols({s: 0.4 for s in ket.symbols})
    bra = _ghz(n_qubits)
    sv = StateVector(bra=bra, operator=op, ket=ket)
    sv.build()
    assert sv.target is Target.TRANSITION_AMPLITUDE
    assert sv._observable is not None
    engine = QarpEngine()
    engine.build([sv])
    amp = engine.run()[0]
    sim = qx.QarpSimulator()
    bra_sv = np.asarray(sim.statevector(bra.flatten(), n_qubits))
    ket_sv = np.asarray(sim.statevector(ket.flatten(), n_qubits))
    assert amp == pytest.approx(_dense(bra_sv, ket_sv, op, n_qubits), rel=RTOL, abs=RTOL)


def test_state_vector_falls_back_without_transition(monkeypatch):
    """A simulator exposing only ``statevector`` (CudaqEngine's host path,
    any third-party backend) still evaluates — through the numpy sweep."""
    from qarp.algorithms._primitives import state_vector as sv_mod

    rng = np.random.default_rng(15)
    n_qubits = 5
    op = _random_operator(rng, n_qubits, 30)
    hea = HEABlock(n_qubits, 1, real=True, linear=True, circular=False, use_cz=False).build()
    hea = hea.set_symbols({s: 0.3 for s in hea.symbols})

    class StatevectorOnly:
        def __init__(self):
            self._sim = qx.QarpSimulator()

        def statevector(self, commands, n_qubits, initial_state=None):
            return self._sim.statevector(commands, n_qubits, initial_state=initial_state)

    calls = []
    real_sweep = sv_mod.pauli_expectation
    monkeypatch.setattr(
        sv_mod, "pauli_expectation", lambda *a, **k: calls.append(1) or real_sweep(*a, **k)
    )
    sv = StateVector(operator=op, ket=hea)
    sv.build()
    engine = QarpEngine()
    engine.build([sv])
    engine._sim = StatevectorOnly()  # the engine hands its simulator to the primitive
    energy = engine.run()[0]
    assert calls, "fallback sweep was not used"
    psi = np.asarray(qx.QarpSimulator().statevector(hea.flatten(), n_qubits))
    assert energy == pytest.approx(_dense(psi, psi, op, n_qubits), rel=RTOL, abs=RTOL)


def test_state_vector_diagonal_operator_keeps_diag_path():
    """Z-only observables keep the build-time diagonal; no observable list."""
    op = QubitOperator(((0, "Z"), (1, "Z")), 1.0) + QubitOperator(((2, "Z"),), 0.5)
    sv = StateVector(operator=op, ket=_hadamard_all(3))
    sv.build()
    assert sv._diag is not None
    assert sv._observable is None


def test_h2_vqe_reaches_fci_through_the_kernel(h2_ev):
    """FCI energy of the H2/STO-3G fixture, −1.1373060357534004 Ha, from a
    VQE whose every energy evaluation goes through ``transition``."""
    ansatz, qop = h2_ev
    vqe = VQE(operator=qop, ket=ansatz, gradient=False)
    vqe.build()
    assert vqe.primitive._observable is not None
    energy, _ = vqe.run()
    assert energy == pytest.approx(-1.1373060357534004, abs=1e-8)


# ── bench ───────────────────────────────────────────────────────────────────


@pytest.mark.bench
@pytest.mark.parametrize("n_qubits", [12, 16, 20])
def test_kernel_speedup_over_numpy_sweep(n_qubits):
    """P3.1 bench row: ≥ 5× over the numpy sweep at every width on the
    chemistry-shaped JW operator (~2000 terms).  Measured 2026-09-13 on an
    M-series laptop: 20q 67× at 12 threads, 8.6× at 1 thread."""
    import time

    from qarp.algorithms._primitives.state_vector import pauli_expectation

    rng = np.random.default_rng(0)
    op = _chemistry_operator(rng, n_qubits, 250)
    obs = _observable(op)
    psi = _random_state(rng, n_qubits)
    sim = qx.QarpSimulator()

    def best_of(f, reps=3):
        f()
        return min(_timed(f) for _ in range(reps))

    def _timed(f):
        t = time.perf_counter()
        f()
        return time.perf_counter() - t

    t_kernel = best_of(lambda: sim.transition(psi, psi, n_qubits, obs))
    t_numpy = best_of(lambda: pauli_expectation(psi, psi, op, n_qubits), reps=1)
    ratio = t_numpy / t_kernel
    print(
        f"n={n_qubits} terms={len(op.terms)} numpy {t_numpy * 1e3:.1f} ms kernel {t_kernel * 1e3:.2f} ms ({ratio:.0f}x)"
    )
    assert ratio >= 5.0, f"kernel only {ratio:.1f}× over the numpy sweep at {n_qubits} qubits"
