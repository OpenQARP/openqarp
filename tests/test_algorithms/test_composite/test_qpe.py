import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from qarp.algorithms import QPE, Sampler, dirichlet_kernel_squared
from qarp.blocks import ComputationalBasisStateBlock, SimpleBlock, TrotterBlock
from qarp.engines import QarpEngine


def test_qpe_h2(scaled_h2_hamiltonian_jw):
    ham, eigs = scaled_h2_hamiltonian_jw
    target = eigs[0]
    trotter = TrotterBlock(operator=-ham, n_qubits=4, steps=5, time=2 * np.pi)
    # Hartree-Fock |1100⟩ in qarpx LSB convention.
    hf_state_block = ComputationalBasisStateBlock([1, 1, 0, 0])
    qpe = QPE(hf_state_block, trotter, n_ancilla=6).build()
    result = qpe.run()
    assert abs(result - target) < 1e-2


@pytest.mark.parametrize("phi", [0.125, 0.375, 0.625, 0.875])
def test_qpe_recovers_exact_phase_synthetic(phi):
    """Synthetic-phase QPE: ``U = P(2π · φ)`` has eigenvalue ``e^{2πi φ}`` on |1⟩.

    With ``n_ancilla = 3`` every ``φ ∈ {k/8 : k = 0..7}`` is exactly representable,
    so QPE must land on the correct bin with probability 1 and recover ``φ`` to
    machine precision.

    Each parametrised phase is deliberately **not** a fixed point of ``φ ↔ 1-φ`` —
    that's the trap that hid the original QFT-sign bug behind ``test_qpe_h2``
    (whose only numeric target was ``eigs[0] = 0``, the unique fixed point).  If
    QFTBlock ever flips back to the IQFT convention, every assertion here fails
    immediately with ``result == 1 - φ`` instead of ``φ``.
    """
    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * phi)
    u.build()
    eigenstate = ComputationalBasisStateBlock([1])
    qpe = QPE(
        state=eigenstate,
        unitary=u,
        n_ancilla=3,
        engine=QarpEngine(seed=0, n_shots=500),
    ).build()
    result = qpe.run()
    assert result == pytest.approx(phi), f"expected exactly {phi}, got {result}"


def test_qpe_plot(scaled_h2_hamiltonian_jw):
    ham, _ = scaled_h2_hamiltonian_jw
    trotter = TrotterBlock(operator=-ham, n_qubits=4, steps=1, time=2 * np.pi).build()
    hf_state_block = ComputationalBasisStateBlock([1, 1, 0, 0])
    qpe = QPE(hf_state_block, trotter, n_ancilla=3).build()
    qpe.run()
    fig, ax = qpe.plot(return_fig=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def generate_dirichlet_dist(phase, n_ancilla):
    N = 2**n_ancilla
    x = [i / N for i in range(N)]
    y = dirichlet_kernel_squared(np.array(x), phase, N)
    y = y / np.sum(y)
    # qarpx SamplingResult / structured QPE both emit LSB-first keys
    # (qubit q's measurement is at bit q of the integer outcome).
    distribution = {tuple((i >> q) & 1 for q in range(n_ancilla)): y[i] for i in range(N)}
    return distribution


@pytest.mark.parametrize("true_phase", [0.125, 0.410, 0.857])
def test_estimate_phase_matches_known_dirichlet_peak(true_phase):
    n_ancilla = 4
    qpe = QPE(state=None, unitary=None, n_ancilla=n_ancilla)
    qpe.distribution = generate_dirichlet_dist(true_phase, n_ancilla)
    estimated = qpe.estimate_phase()

    assert np.isclose(estimated, true_phase, atol=1e-2), f"Expected ~{true_phase}, got {estimated}"


def test_qpe_engine_wide_exact_bypasses_structured_path():
    """QarpEngine(n_shots=qarp.EXACT): the structured C++ sampler has no
    analytic branch, so QPE must take the generic path (which supports
    EXACT) instead of feeding the enum into nanobind."""
    import qarp

    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * 0.375)
    u.build()
    eigenstate = ComputationalBasisStateBlock([1])
    qpe = QPE(
        state=eigenstate,
        unitary=u,
        n_ancilla=3,
        engine=QarpEngine(n_shots=qarp.EXACT),
    ).build()
    assert qpe._plan is None
    result = qpe.run()
    assert result == pytest.approx(0.375, abs=1e-12)
    assert qpe.result_probability == pytest.approx(1.0, abs=1e-10)


# ── Structured fast-path eligibility (Engine.prepare_structured_qpe) ──────


def _phase_u(phi=0.375):
    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * phi)
    u.build()
    return u


def test_qpe_primitive_exact_bypasses_structured_path():
    """A per-primitive EXACT override must refuse the fast path too (it wins
    over the engine default in shot resolution)."""
    import qarp
    from qarp.algorithms import Sampler

    qpe = QPE(
        state=ComputationalBasisStateBlock([1]),
        unitary=_phase_u(),
        n_ancilla=3,
        primitive=Sampler(n_shots=qarp.EXACT),
        engine=QarpEngine(n_shots=500),
    ).build()
    assert qpe._plan is None
    assert qpe.run() == pytest.approx(0.375, abs=1e-12)


def test_qpe_noisy_engine_bypasses_structured_path():
    """Enabled noise → generic (trajectory) path.

    Regression: the structured C++ sampler never applies the noise model, so
    taking the fast path under noise silently returned noiseless samples.
    """
    from qarp.devices import NoiseModel

    qpe = QPE(
        state=ComputationalBasisStateBlock([1]),
        unitary=_phase_u(),
        n_ancilla=3,
        engine=QarpEngine(n_qubits=4, noise_model=NoiseModel.bit_flip(0.02), n_shots=200, seed=0),
    ).build()
    assert qpe._plan is None
    qpe.run()
    assert sum(qpe.distribution.values()) == pytest.approx(1.0)


def test_qpe_parametric_unitary_gets_no_plan():
    from sympy import Symbol

    from qarp.algorithms import Sampler

    u = SimpleBlock(1, name="U")
    u.p(0, Symbol("theta"))
    u.build()
    state = ComputationalBasisStateBlock([1]).build()
    plan = QarpEngine(n_shots=100).prepare_structured_qpe("qpe", u, state, 2, Sampler())
    assert plan is None


def test_qpe_routed_engine_gets_no_plan():
    from qarp.algorithms import Sampler
    from qarp.devices import Architecture

    state = ComputationalBasisStateBlock([1]).build()
    eng = QarpEngine(
        n_qubits=4,
        architecture=Architecture(4, [(0, 1), (1, 2), (2, 3)], "directed_edge"),
        n_shots=100,
    )
    plan = eng.prepare_structured_qpe("qpe", _phase_u(), state, 3, Sampler())
    assert plan is None


def test_qpe_seeded_primitive_bypasses_structured_path():
    """A Sampler carrying initial_state must refuse the plan — ``sample()``
    cannot thread the seed.  The generic path must read phase 3/4 from the
    seeded |1⟩ eigenstate; a fast path ignoring the seed would read 0 (|0⟩ is
    also an eigenstate of P)."""
    phi = 0.75
    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * phi)
    u.build()
    prep = SimpleBlock(1)  # identity prep: the eigenstate comes from the seed
    n_ancilla = 2
    psi = np.zeros(2 ** (n_ancilla + 1), dtype=complex)
    psi[1 << n_ancilla] = 1.0  # system qubit (index n_ancilla) in |1⟩

    qpe = QPE(
        state=prep,
        unitary=u,
        n_ancilla=n_ancilla,
        primitive=Sampler(n_shots=500, initial_state=psi),
        engine=QarpEngine(seed=0),
    ).build()
    assert qpe._plan is None
    result = qpe.run()
    assert result == pytest.approx(phi)


def test_base_engine_default_has_no_structured_path():
    """Engines without an override (a base-default stub here) fall back by
    contract, not by the old accidental hasattr(engine._sim, ...) miss."""
    from qarp.algorithms import Sampler
    from tests.conftest import _StubEngine

    state = ComputationalBasisStateBlock([1]).build()
    assert _StubEngine().prepare_structured_qpe("qpe", _phase_u(), state, 3, Sampler()) is None


def test_structured_plan_refuses_late_enabled_noise():
    """A plan prepared noise-free must not silently run once noise is enabled."""
    from qarp.devices import NoiseModel
    from qarp.errors import CapabilityError

    eng = QarpEngine(n_qubits=4, noise_model=NoiseModel.bit_flip(0.02), n_shots=200, seed=0)
    eng.noise_model.enabled = False
    qpe = QPE(
        state=ComputationalBasisStateBlock([1]),
        unitary=_phase_u(),
        n_ancilla=3,
        engine=eng,
    ).build()
    assert qpe._plan is not None
    eng.noise_model.enabled = True
    with pytest.raises(CapabilityError):
        qpe.run()


def test_qpe_run_before_build_raises():
    qpe = QPE(state=None, unitary=None, n_ancilla=3)
    with pytest.raises(ValueError, match="not built"):
        qpe.run()


def test_qpe_estimate_phase_before_run_raises():
    qpe = QPE(state=None, unitary=None, n_ancilla=3)
    with pytest.raises(ValueError, match="No distribution"):
        qpe.estimate_phase()


def test_qpe_plot_before_run_raises():
    qpe = QPE(state=None, unitary=None, n_ancilla=3)
    with pytest.raises(ValueError, match="No distribution"):
        qpe.plot()


def test_estimate_phase_fit_range_and_verbose(capsys):
    # Restricting the fit window around the true phase must still recover it
    # from the exact analytic (Dirichlet-kernel) QPE distribution.
    true_phase = 0.41
    qpe = QPE(state=None, unitary=None, n_ancilla=4)
    qpe.distribution = generate_dirichlet_dist(true_phase, 4)

    estimated = qpe.estimate_phase(fit_range=(0.2, 0.6), verbose=True)
    assert np.isclose(estimated, true_phase, atol=1e-2)

    out = capsys.readouterr().out
    assert "Fit range: (0.2, 0.6)" in out
    assert "estimated phase using the Dirichlet kernel" in out


def test_estimate_phase_verbose_full_range(capsys):
    qpe = QPE(state=None, unitary=None, n_ancilla=4)
    qpe.distribution = generate_dirichlet_dist(0.125, 4)
    qpe.estimate_phase(verbose=True)
    assert "Fit range: (0, 1)" in capsys.readouterr().out


def test_qpe_plot_bars_match_distribution_and_show_path(monkeypatch):
    # Bar heights are the distribution re-keyed MSB-first onto the phase grid.
    n_ancilla = 3
    qpe = QPE(state=None, unitary=None, n_ancilla=n_ancilla)
    qpe.distribution = generate_dirichlet_dist(0.375, n_ancilla)

    fig, ax = qpe.plot(figsize=(4, 3), return_fig=True)
    heights = [patch.get_height() for patch in ax.patches]
    # φ = 3/8 is exactly representable: all weight on one bar.
    assert np.isclose(max(heights), 1.0)
    assert np.argmax(heights) == 3
    assert np.isclose(sum(heights), 1.0)
    plt.close(fig)

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    assert qpe.plot() is None
    assert shown == [True]
    plt.close("all")


def test_qpe_presets_sampler_measured_qubits_respected():
    # A Sampler with measured_qubits already set keeps it through build.
    qpe = QPE(
        state=ComputationalBasisStateBlock([1]),
        unitary=_phase_u(),
        n_ancilla=3,
        primitive=Sampler(n_shots=200),
        engine=QarpEngine(seed=0),
    )
    qpe.primitive.measured_qubits = [0, 1, 2]
    qpe.build()
    if qpe._plan is None:
        assert qpe.primitive.measured_qubits == [0, 1, 2]
    assert qpe.run() == pytest.approx(0.375)


def test_qpe_with_synthesized_unitary_recovers_eigenphase():
    """End-to-end pin of the global-phase calibration: QPE over a
    SynthesizedUnitaryBlock must peak at the grid point nearest the TRUE
    eigenphase — an uncalibrated synthesis phase shifted every readout."""
    from qarp.blocks import SynthesizedStateBlock, SynthesizedUnitaryBlock

    rng = np.random.default_rng(5)
    m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    q, r = np.linalg.qr(m)
    v = q * (np.diag(r) / np.abs(np.diag(r)))
    phis = np.array([0.3, 0.62, 0.11, 0.85])
    u = v @ np.diag(np.exp(2j * np.pi * phis)) @ v.conj().T

    n_anc = 4
    state = SynthesizedStateBlock(2, list(v[:, 0]))  # exact eigenvector, φ = 0.3
    unitary = SynthesizedUnitaryBlock(u)
    qpe = QPE(
        state=state,
        unitary=unitary,
        n_ancilla=n_anc,
        engine=QarpEngine(seed=1, n_shots=4096),
    ).build()
    result = qpe.run()
    assert result == pytest.approx(round(0.3 * 2**n_anc) / 2**n_anc)  # 5/16


@pytest.mark.parametrize(
    "kwarg", ["optimize_ctrl_u_sequences", "max_optimized_repetitions", "fully_optimized"]
)
def test_qpe_dead_optimization_kwargs_are_gone(kwarg):
    with pytest.raises(TypeError):
        QPE(
            state=ComputationalBasisStateBlock([1]),
            unitary=_phase_u(),
            n_ancilla=3,
            **{kwarg: True},
        )
