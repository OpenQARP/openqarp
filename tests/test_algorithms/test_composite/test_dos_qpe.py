import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")

from qarp.algorithms import DOSQPE
from qarp.blocks import TrotterBlock
from qarp.operators import QubitOperator

# Scope note:
#
# `unitary_synthesis()` is phase-exact, so DOSQPE spectrum recovery over a
# synthesized U is meaningful — see `test_dosqpe_recovers_synthesized_spectrum`
# below.  It was not always: the ZYZ base case emitted a wrong global phase on
# its diagonal / anti-diagonal branches, which DOSQPE reads directly as an
# eigenphase shift.  Phase exactness is pinned at source by
# `cpp/libqarpx/tests/cpp/test_synthesis_unitary.cpp` and
# `tests/test_blocks/test_primitives/test_synthesized_blocks.py`.
#
# Transpiler-level optimisation is covered separately by the C++ suite in
# `cpp/libqarpx/tests/cpp/test_transpiler*.cpp`.


def test_dosqpe_plot(scaled_h2_hamiltonian_jw):
    ham, _ = scaled_h2_hamiltonian_jw
    trotter = TrotterBlock(operator=ham, n_qubits=4, steps=5, time=2 * np.pi).build()
    dosqpe = DOSQPE(trotter, 3)
    dosqpe.build().run()
    fig, ax = dosqpe.plot(return_fig=True)
    assert isinstance(fig, plt.Figure)
    assert isinstance(ax, plt.Axes)
    plt.close(fig)


def test_dosqpe_structured_matches_generic_exact():
    """Fast path (sampled, seeded) vs generic path (EXACT analytic) on a
    1-qubit U = P(2π·3/8) over the maximally mixed probe: the DOS is half
    weight at phase 0, half at 3/8, both exactly representable at
    n_ancilla=3.  Pins the two paths to the same distribution."""
    import qarp
    from qarp.blocks import SimpleBlock
    from qarp.engines import QarpEngine

    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * 0.375)
    u.build()

    structured = DOSQPE(unitary=u, n_ancilla=3, engine=QarpEngine(seed=0, n_shots=4000)).build()
    assert structured._plan is not None
    d_fast = structured.run()

    generic = DOSQPE(unitary=u, n_ancilla=3, engine=QarpEngine(n_shots=qarp.EXACT)).build()
    assert generic._plan is None
    d_exact = generic.run()

    keys = set(d_fast) | set(d_exact)
    tv = 0.5 * sum(abs(d_fast.get(k, 0.0) - d_exact.get(k, 0.0)) for k in keys)
    assert tv < 0.05
    # Both paths: half the weight at phase 0 (bits (0,0,0)), half at 3/8 (bits LSB-first of 3).
    assert d_exact[(0, 0, 0)] == pytest.approx(0.5, abs=1e-10)
    assert d_exact[(1, 1, 0)] == pytest.approx(0.5, abs=1e-10)


def test_dosqpe_recovers_synthesized_spectrum():
    """Full-spectrum recovery over a synthesized (not hand-built) 2-qubit U.

    Eigenphases j/8 for j = 0,1,3,5 are exactly representable at n_ancilla=3,
    and the maximally mixed probe puts 1/4 of the weight on each.  U is
    conjugated out of diagonal form so QSD takes its generic recursion.

    DOSQPE reads the synthesized block's global phase as an eigenphase, so this
    fails outright if `unitary_synthesis` is correct only up to e^{iγ}.
    """
    import qarp
    from qarp.blocks import SynthesizedUnitaryBlock
    from qarp.engines import QarpEngine

    js = [0, 1, 3, 5]
    D = np.diag([np.exp(2j * np.pi * j / 8) for j in js])
    h = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    W = np.kron(h, h)
    U = W @ D @ W.conj().T

    u = SynthesizedUnitaryBlock(unitary_matrix=U).build()
    dist = DOSQPE(unitary=u, n_ancilla=3, engine=QarpEngine(n_shots=qarp.EXACT)).build().run()

    for j in js:
        key = tuple((j >> b) & 1 for b in range(3))  # ancilla bits are LSB-first
        assert dist.get(key, 0.0) == pytest.approx(0.25, abs=1e-9), f"phase {j}/8"
    assert sum(dist.values()) == pytest.approx(1.0)


def test_dosqpe_freqs_grid_and_unset_raise():
    dosqpe = DOSQPE(unitary=None, n_ancilla=3)
    assert np.allclose(dosqpe.freqs, np.arange(8) / 8)
    dosqpe.n_ancilla = None
    with pytest.raises(ValueError, match="n_ancilla not set"):
        dosqpe.freqs


def test_dosqpe_run_before_build_raises():
    dosqpe = DOSQPE(unitary=None, n_ancilla=3)
    with pytest.raises(ValueError, match="not built"):
        dosqpe.run()


def _two_qubit_phase_unitary():
    """Diagonal U with §2 P-gate eigenphases: |q1 q0⟩ = |01⟩ → 1/4, |10⟩ → 1/2."""
    from qarp.blocks import SimpleBlock

    u = SimpleBlock(2, name="U")
    u.p(0, 2 * np.pi * 0.25)
    u.p(1, 2 * np.pi * 0.5)
    return u.build()


def test_dosqpe_dicke_probe_exact_sector_dos():
    """Dicke |2,1⟩ probe = maximally mixed over the hamming-weight-1 sector:
    the exact DOS is half weight at each sector eigenphase (1/4 and 1/2),
    both exactly representable at n_ancilla=4."""
    import qarp
    from qarp.engines import QarpEngine

    dosqpe = DOSQPE(
        unitary=_two_qubit_phase_unitary(),
        n_ancilla=4,
        hamming_weight=1,
        engine=QarpEngine(n_shots=qarp.EXACT),
    ).build()
    dist = dosqpe.run()

    bits_quarter = tuple((4 >> b) & 1 for b in range(4))  # φ = 4/16
    bits_half = tuple((8 >> b) & 1 for b in range(4))  # φ = 8/16
    assert dist.get(bits_quarter, 0.0) == pytest.approx(0.5, abs=1e-9)
    assert dist.get(bits_half, 0.0) == pytest.approx(0.5, abs=1e-9)
    assert sum(dist.values()) == pytest.approx(1.0)
    return dosqpe


def test_dosqpe_plot_before_run_raises():
    dosqpe = DOSQPE(unitary=None, n_ancilla=3)
    with pytest.raises(ValueError, match="No distribution"):
        dosqpe.plot()
    with pytest.raises(ValueError, match="No distribution"):
        dosqpe.plot_against_spectrum([], [], [])


def test_dosqpe_dicke_plot_structural(monkeypatch):
    dosqpe = test_dosqpe_dicke_probe_exact_sector_dos()

    fig, ax = dosqpe.plot(return_fig=True)
    assert ax.get_title() == "probe: Dicke state |2, 1>"
    heights = [p.get_height() for p in ax.patches]
    assert np.isclose(sum(heights), 1.0)
    # Two half-weight bars at bins 4 and 8 of 16.
    assert np.isclose(heights[4], 0.5) and np.isclose(heights[8], 0.5)
    plt.close(fig)

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    assert dosqpe.plot(figsize=(4, 3)) is None
    assert shown == [True]
    plt.close("all")


def test_dosqpe_plot_against_spectrum_structural(monkeypatch):
    dosqpe = test_dosqpe_dicke_probe_exact_sector_dos()

    fig, ax = dosqpe.plot_against_spectrum(
        unique_eigs=[0.25, 0.5],
        normalized_degeneracy=[0.5, 0.5],
        unique_occ_numbers=[1, 1],
        return_fig=True,
    )
    # Legend carries one entry per occupation number present.
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert labels == ["Occ. #: 1"]
    plt.close(fig)

    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    assert (
        dosqpe.plot_against_spectrum(
            unique_eigs=[0.25, 0.5],
            normalized_degeneracy=[0.5, 0.5],
            unique_occ_numbers=[1, 1],
        )
        is None
    )
    assert shown == [True]
    plt.close("all")


def test_dosqpe_noisy_engine_bypasses_structured_path():
    from qarp.blocks import SimpleBlock
    from qarp.devices import NoiseModel
    from qarp.engines import QarpEngine

    u = SimpleBlock(1, name="U")
    u.p(0, 2 * np.pi * 0.375)
    u.build()
    dosqpe = DOSQPE(
        unitary=u,
        n_ancilla=2,
        engine=QarpEngine(n_qubits=4, noise_model=NoiseModel.bit_flip(0.02), n_shots=200, seed=0),
    ).build()
    assert dosqpe._plan is None
    dist = dosqpe.run()
    assert sum(dist.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwarg", ["optimize_ctrl_u_sequences", "max_optimized_repetitions", "fully_optimized"]
)
def test_dos_qpe_dead_optimization_kwargs_are_gone(kwarg):
    u = TrotterBlock(operator=QubitOperator("Z0"), n_qubits=1, steps=1, time=1.0)
    with pytest.raises(TypeError):
        DOSQPE(unitary=u, n_ancilla=2, **{kwarg: True})
