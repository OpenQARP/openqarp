"""ADAPT-VQE / ADAPT-VQD pyscf-free units on a Fermi-Hubbard chain.

The pool-scan equivalence tests pin the generic (shot-primitive) gradient
path against the statevector fast path at exact shots — two independent
implementations of the same ⟨[H, e]⟩ pool gradients.
"""

import numpy as np
import pytest

import qarp
from qarp.algorithms import AdaptVQD, AdaptVQE, StateVector, TermwiseHadamardTest
from qarp.blocks import MappedONVStateBlock
from qarp.operators import JordanWigner
from qarp.operators.models import fermi_hubbard
from qarp.operators.ucc import ucc_singles_and_doubles
from qarp.optimizers import ScipyOptimizer


def _fh_problem(n=2):
    qham = JordanWigner().encode_operator(fermi_hubbard((n,), 1.4, 2.31))
    onv = [1] * n + [0] * n
    qucc = JordanWigner().encode_operator(
        ucc_singles_and_doubles(onv, spin_conserving=True, generalised=False)[0]
    )
    return qham, onv, qucc


def _adapt_vqe(primitive, **kwargs):
    qham, onv, qucc = _fh_problem()
    defaults = dict(
        reference_block=MappedONVStateBlock(onv),
        system_hamiltonian=qham,
        excitation_pool=qucc,
        primitive=primitive,
        optimizer=ScipyOptimizer("COBYLA"),
        diminishing=False,
        verbose=False,
        gradient=False,
    )
    defaults.update(kwargs)
    return AdaptVQE(**defaults)


def _adapt_vqd(primitive, **kwargs):
    qham, onv, qucc = _fh_problem()
    defaults = dict(
        reference_block=MappedONVStateBlock(onv).build(),
        hamiltonian=qham,
        excitation_pool=qucc,
        orthogonal_states=[MappedONVStateBlock(onv).build()],
        betas=[2.0],
        optimizer=ScipyOptimizer("COBYLA"),
        verbose=False,
        primitive=primitive,
    )
    defaults.update(kwargs)
    return AdaptVQD(**defaults)


def test_adapt_vqd_rejects_unknown_term_gradient():
    with pytest.raises(ValueError, match="ADAPT-VQE or ADAPT-VQD"):
        _adapt_vqd(StateVector(), term_gradient="bogus")


def test_adapt_vqd_default_optimizer_is_cg():
    adapt = _adapt_vqd(StateVector(), optimizer=None)
    assert adapt.optimizer.method == "CG"


def test_adapt_vqe_pool_scan_hadamard_matches_statevector():
    sv = _adapt_vqe(StateVector())
    sv.build()
    grads_sv = np.asarray(sv.pool_scan(), dtype=float)

    ht = _adapt_vqe(TermwiseHadamardTest(n_shots=qarp.EXACT))
    ht.build()
    grads_ht = np.asarray(ht.pool_scan(), dtype=float)

    assert grads_sv.shape == grads_ht.shape
    assert np.allclose(grads_sv, grads_ht, atol=1e-9)


def test_adapt_vqd_pool_scan_hadamard_matches_statevector():
    # Deflation terms included: the overlap primitives carry the identity
    # operator (bare ⟨orth|ψ⟩), so the scan runs on shot primitives too.
    sv = _adapt_vqd(StateVector())
    sv.build()
    grads_sv = np.asarray(sv.pool_scan(), dtype=float)

    ht = _adapt_vqd(TermwiseHadamardTest(n_shots=qarp.EXACT))
    ht.build()
    grads_ht = np.asarray(ht.pool_scan(), dtype=float)

    assert grads_sv.shape == grads_ht.shape
    assert np.allclose(grads_sv, grads_ht, atol=1e-9)


def test_adapt_vqd_pool_scan_vqe_gradients_hadamard_matches_statevector():
    # term_gradient="ADAPT-VQE": deflation terms excluded from the scan.
    sv = _adapt_vqd(StateVector(), term_gradient="ADAPT-VQE")
    sv.build()
    grads_sv = np.asarray(sv.pool_scan(), dtype=float)

    ht = _adapt_vqd(TermwiseHadamardTest(n_shots=qarp.EXACT), term_gradient="ADAPT-VQE")
    ht.build()
    grads_ht = np.asarray(ht.pool_scan(), dtype=float)

    assert np.allclose(grads_sv, grads_ht, atol=1e-9)


def test_adapt_vqe_empty_pool_verbose_exit(capsys):
    adapt = _adapt_vqe(StateVector(), verbose=True)
    adapt.build()
    adapt.pool = []
    adapt.iterate()
    assert "Excitation pool fully diminished" in capsys.readouterr().out


def test_adapt_vqd_empty_pool_verbose_exit(capsys):
    adapt = _adapt_vqd(StateVector(), verbose=True)
    adapt.build()
    adapt.pool = []
    adapt.iterate()
    assert "Excitation pool fully diminished" in capsys.readouterr().out


def test_adapt_vqe_run_with_empty_pool_reports_failure(capsys):
    adapt = _adapt_vqe(StateVector())
    adapt.build()
    adapt.pool = []
    energy, params = adapt.run(max_iter=1)
    assert energy == 0.0
    assert params is None
    assert "ADAPT-VQE failed" in capsys.readouterr().out


def test_adapt_vqe_loose_convergence_threshold_terminates():
    adapt = _adapt_vqe(StateVector(), convergence_thresh=100.0)
    adapt.build()
    adapt.run(max_iter=5)
    # A 100-Ha threshold is met after the second energy evaluation.
    assert len(adapt.iter_energies) <= 3


def test_adapt_vqd_loose_gradient_threshold_terminates(capsys):
    adapt = _adapt_vqd(StateVector(), verbose=True)
    adapt.gradient_thresh = 1e6
    adapt.build()
    adapt.iterate()
    assert "Terminating" in capsys.readouterr().out


def test_adapt_vqd_rejects_a_betas_length_mismatch():
    """One penalty weight per orthogonal state; a short list used to truncate
    the penalty sum silently in both pool-scan paths."""
    with pytest.raises(ValueError, match="one penalty weight per orthogonal state"):
        _adapt_vqd(StateVector(), betas=[])
    with pytest.raises(ValueError, match="one penalty weight per orthogonal state"):
        _adapt_vqd(StateVector(), betas=[2.0, 3.0])


def _rephased_reference(phi):
    """The ADAPT reference ONV carrying a global phase ``e^{-iφ/2}``.

    ``Rz`` on an occupied wire is a pure global phase on a basis state, so this
    is the *same physical state* — the deflation penalty must not move.
    """
    qham, onv, qucc = _fh_problem()
    block = MappedONVStateBlock(onv)
    block.rz(0, phi)
    return block.build()


class _CapturingOptimizer:
    """Captures ``iterate()``'s objective/gradient closures instead of
    optimizing, so the two can be compared at a chosen point."""

    def __init__(self):
        self.objective = None
        self.gradient = None

    def minimize(self, objective_function, initial_parameters, gradient=None):
        from scipy.optimize import OptimizeResult

        self.objective = objective_function
        self.gradient = gradient
        x = np.asarray(initial_parameters, dtype=float)
        objective_function(x)  # iterate() reads vqe.en afterwards
        return OptimizeResult(x=x, fun=0.0, success=True)


def _captured_adapt_vqd(orthogonal_state, primitive=None, **kwargs):
    """``iterate()``-d AdaptVQD whose objective/gradient closures are captured."""
    opt = _CapturingOptimizer()
    adapt = _adapt_vqd(
        primitive if primitive is not None else StateVector(),
        optimizer=opt,
        orthogonal_states=[orthogonal_state],
        **kwargs,
    )
    adapt.build()
    adapt.iterate()
    return adapt, opt


def test_adapt_vqd_penalty_is_invariant_under_a_phase_on_the_orthogonal_state():
    """|⟨ψ_i|ψ⟩|² cannot depend on the arbitrary global phase of |ψ_i⟩.

    That is what made the old ``Re(o²)`` penalty wrong rather than merely
    different: ``Re(o²) = |o|²·cos(2·arg o)`` moves when a deflation state is
    rephased — a physically empty change — and vanishes outright at arg o = π/4.
    """
    x = None
    energies, objectives, overlaps = [], [], []
    for phi in (0.0, 0.7, np.pi / 2, 2.0):
        adapt, opt = _captured_adapt_vqd(_rephased_reference(phi))
        if x is None:
            x = np.full(len(adapt.ansatz_parameters), 0.35)
        values = adapt.engine.run(dict(zip(adapt.ansatz_symbols, x, strict=True)))
        energies.append(complex(values[0]))
        overlaps.append(complex(values[1]))
        objectives.append(opt.objective(x))

    # The rephasing is real: the overlaps differ in phase but not in modulus.
    assert len({round(o.imag, 9) for o in overlaps}) > 1
    assert all(abs(o) == pytest.approx(abs(overlaps[0])) for o in overlaps)

    # Oracle: energy + β·|o|², hand-assembled from the engine's own values.
    beta = 2.0
    for energy, overlap, objective in zip(energies, overlaps, objectives, strict=True):
        assert objective == pytest.approx(np.real(energy) + beta * abs(overlap) ** 2)
    assert objectives == pytest.approx([objectives[0]] * len(objectives))

    # The old form is not invariant, so this is a discriminating check.
    old = [
        np.real(energy + beta * overlap**2)
        for energy, overlap in zip(energies, overlaps, strict=True)
    ]
    assert not np.allclose(old, old[0])


# Primitives constructed inside the test, not in the parametrize list -- a
# qarpx-backed object there stays alive to interpreter shutdown.
@pytest.mark.parametrize("primitive_name", ["StateVector", "TermwiseHadamardTest"])
@pytest.mark.parametrize("phi", [0.0, 0.7])
def test_adapt_vqd_analytic_gradient_matches_finite_differences_of_its_objective(
    phi, primitive_name
):
    """Gradient and objective must describe the same function.

    Central FD of ``iterate()``'s own objective is the independent check on the
    penalty assembly, which routes through ``squared_overlap_gradient``.  The
    φ = 0.7 row is the one that matters: it gives the overlap an imaginary part,
    which is exactly where ``Re(o²)`` and ``|o|²`` part company.  StateVector
    routes through the engine's |o|² gradient; TermwiseHadamardTest returns the
    amplitude, so it exercises the ``2·Re(o*·∂o)`` chain-rule branch of
    ``deflation_gradient`` that StateVector cannot.
    """
    primitive = {
        "StateVector": lambda: StateVector(),
        "TermwiseHadamardTest": lambda: TermwiseHadamardTest(
            n_shots=qarp.EXACT, real=True, imaginary=True
        ),
    }[primitive_name]()
    adapt, opt = _captured_adapt_vqd(_rephased_reference(phi), primitive=primitive, gradient=True)

    assert opt.gradient is not None, "gradient=True must reach the optimizer"
    x = np.full(len(adapt.ansatz_parameters), 0.35)
    analytic = np.asarray(opt.gradient(x), dtype=float)

    h = 1e-5
    fd = np.empty_like(analytic)
    for i in range(len(x)):
        xp, xm = x.copy(), x.copy()
        xp[i] += h
        xm[i] -= h
        fd[i] = (opt.objective(xp) - opt.objective(xm)) / (2 * h)

    assert np.allclose(analytic, fd, atol=1e-5), f"analytic {analytic} vs FD {fd}"


def test_adapt_vqd_pool_scan_builds_its_deflation_overlaps_as_overlaps():
    """The ⟨orth|ψ⟩ term of the shot-primitive pool scan is an OVERLAP target with
    no operator, not a transition amplitude of a substituted identity.  The
    engine keeps the last pool element's three terms: commutator, transition
    amplitude, overlap."""
    from qarp.algorithms._primitives.target import Target

    ht = _adapt_vqd(TermwiseHadamardTest(n_shots=qarp.EXACT))
    ht.build()
    ht.pool_scan()
    comm, tamp, ov = ht.engine._primitives
    assert (comm.target, tamp.target) == (Target.EXPECTATION_VALUE, Target.TRANSITION_AMPLITUDE)
    assert ov.target is Target.OVERLAP
    assert ov.operator is None


# ── P1.1: run(max_iter) stops on convergence; an empty pool terminates ─────


def _count_iterates(adapt, max_iter=25):
    calls = {"n": 0}
    real = adapt.iterate

    def counting():
        calls["n"] += 1
        real()

    adapt.iterate = counting
    adapt.run(max_iter=max_iter)
    return calls["n"]


def test_adapt_vqe_run_breaks_on_convergence():
    # 1st iterate adds an operator; the 2nd sees |dE| < 1e9 and converges.
    adapt = _adapt_vqe(StateVector(), convergence_thresh=1e9).build()
    assert _count_iterates(adapt) == 2


def test_adapt_vqe_empty_pool_terminates():
    adapt = _adapt_vqe(StateVector(), excitation_pool=[]).build()
    assert _count_iterates(adapt) == 1


def test_adapt_vqd_run_breaks_on_convergence():
    # AdaptVQD converges on the gradient threshold (no energy-difference
    # test): the first pool scan is below 1e9, so exactly one iterate runs.
    adapt = _adapt_vqd(StateVector(), gradient_thresh=1e9).build()
    assert _count_iterates(adapt) == 1
