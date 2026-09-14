"""Pollmann, Khemani, Cirac, Sondhi, PRB 94, 041116(R) (2016), arXiv:1506.07179.

Eq. (5): ``H = J sum_n S_n . S_{n+1} - sum_n h_n S^z_n``, spin-1/2, ``J = 1``,
``h_n`` uniform in ``[-W, W]``.  Fig. 3 plots the disorder-averaged cost of
Eq. (4) divided by ``2^L`` ("mean variance") against ``L`` at ``W = 8``: it
grows linearly in ``L`` with a slope that falls as ``N_layer`` grows.
"""

import numpy as np
import pytest

qtn = pytest.importorskip("quimb.tensor")

from qarp.operators import VUMPO, qubit_operator_to_mpo
from tests.test_operators.test_vumpo_network import heisenberg_chain

W = 8.0


def _mean_variance(L, n_layers, seed, n_sweeps=6):
    rng = np.random.default_rng(seed)
    H = qubit_operator_to_mpo(heisenberg_chain(L, rng.uniform(-W, W, size=L)), L)
    v = VUMPO(H, n_layers=n_layers, n_sweeps=n_sweeps, tol=1e-6)
    if n_layers == 0:
        return v.energy_variance(np.zeros(0)) / 2**L
    params = v.optimize_local_sweep(
        [np.zeros(v.params_per_gate) for _ in range(v.n_gates)]
    )  # identity start
    return v.energy_variance(params) / 2**L


@pytest.mark.parametrize("L", [6, 8, 10])
def test_zero_layers_mean_variance_is_closed_form(L):
    """With no gates the eigenstates are product states.  Only the flip-flop
    term ``J/2 (S+S- + h.c.)`` is off-diagonal, each antiparallel bond adds
    ``(J/2)^2`` to a state's variance, and half the bonds are antiparallel on
    average, so the mean variance is ``(L - 1) / 8`` for any disorder."""
    for seed in (1, 2):
        assert _mean_variance(L, 0, seed) == pytest.approx((L - 1) / 8, rel=1e-10)


@pytest.mark.slow
def test_fig3_mean_variance_grows_linearly_with_decreasing_slope():
    """The paper's reading of Fig. 3: linear in L, slope decreasing in
    N_layer, every optimised point below the N_layer = 0 line."""
    Ls = [8, 12, 16]
    seeds = range(10)
    means = {
        n: np.array([np.mean([_mean_variance(L, n, s) for s in seeds]) for L in Ls]) for n in (1, 2)
    }
    slopes = {n: np.polyfit(Ls, means[n], 1)[0] for n in (1, 2)}
    zero_line = (np.array(Ls) - 1) / 8
    assert np.all(means[1] < zero_line)
    assert np.all(means[2] < means[1])
    assert slopes[2] < slopes[1] < 1 / 8
    assert slopes[2] > 0  # still linear growth, not flat


@pytest.mark.slow
def test_two_layers_cut_the_variance_by_more_than_an_order_of_magnitude():
    """The paper's discussion of Fig. 2/3 at W = 8, L = 8: adding layers
    "strongly improves" the eigenstates; two layers beat product states by
    more than 10x in the summed variance."""
    ratios = [_mean_variance(8, 2, s) / _mean_variance(8, 0, s) for s in range(5)]
    assert np.mean(ratios) < 0.1


@pytest.mark.bench
def test_sweep_versus_global_benchmark_protocol():
    """The comparison the plan fixed before implementation (Decisions §1):
    Eq. (5) at W = 8, L in {8, 16, 24}, two layers, five seeds, both optimisers
    from the identity start to the same relative tolerance.  Reports wall
    clock and the mean variance reached; asserts only that both beat the
    product-state line, since the default is the paper's sweep by decision."""
    import time

    rows = []
    for L in (8, 16, 24):
        for seed in range(5):
            rng = np.random.default_rng(seed)
            H = qubit_operator_to_mpo(heisenberg_chain(L, rng.uniform(-W, W, size=L)), L)
            out = {}
            for opt in ("local", "global"):
                v = VUMPO(H, n_layers=2, n_sweeps=50, tol=1e-6, maxiter_global=500, opt=opt)
                start = [np.zeros(v.params_per_gate) for _ in range(v.n_gates)]
                t = time.perf_counter()
                p = v.optimize_local_sweep(start) if opt == "local" else v.optimize_global(start)
                out[opt] = (time.perf_counter() - t, v.energy_variance(p) / 2**L)
            rows.append((L, seed, out))
    print()
    for L in (8, 16, 24):
        sub = [r[2] for r in rows if r[0] == L]
        t_l = np.mean([o["local"][0] for o in sub])
        t_g = np.mean([o["global"][0] for o in sub])
        f_l = np.mean([o["local"][1] for o in sub])
        f_g = np.mean([o["global"][1] for o in sub])
        print(
            f"L={L:2d}  sweep {t_l:6.2f} s -> f/2^L={f_l:.4f} | global {t_g:6.2f} s -> f/2^L={f_g:.4f} | product-state line {(L - 1) / 8:.3f}"
        )
        assert f_l < (L - 1) / 8 and f_g < (L - 1) / 8
