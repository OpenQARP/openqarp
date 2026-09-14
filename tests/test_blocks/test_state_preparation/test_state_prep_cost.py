"""Cost tests for the state-preparation blocks (§18 resources, §19).

Two assertions per block, both on the CNOT count after ``mcx`` decomposes on
``qx.clifford_t_rz_gateset()`` at ``O0`` (deterministic; no optimiser):

* **Ceiling** — pins the *current* construction's measured cost, +5 % for
  transpiler-rule churn.  A rise is a regression; a fall means the constant
  should be lowered.
* **Cited scaling** — the bar for the declared follow-up PR,
  ``xfail(strict=True)`` so it XPASSes, and the marker must be removed, the
  moment the rewrite lands.

Every block-level number rests on the ancilla-free ``k``-controlled ``mcx``
decomposition, pinned separately at the bottom.
"""

import math

import numpy as np
import pytest

import qarpx as qx
from qarp.blocks import (
    LowRankStateBlock,
    MPSStateBlock,
    PiecewiseLinearStateBlock,
    QROMBlock,
    SimpleBlock,
    SlaterDeterminantBlock,
    SparseStateBlock,
    SynthesizedStateBlock,
    UniformSuperpositionBlock,
)
from qarp.resources import estimate


def _cnots(block) -> int:
    block.build()
    report = estimate(block, gateset=qx.clifford_t_rz_gateset(), opt_level=qx.OptLevel.O0)
    return report.final.n_2q


def _ceiling(measured: int) -> int:
    return math.ceil(1.05 * measured)


# --- case builders (plain functions; qarpx objects are built inside the test) ---


def _sparse(n_qubits: int, s: int, seed: int) -> SparseStateBlock:
    rng = np.random.default_rng(seed)
    amplitudes: dict[tuple[int, ...], complex] = {}
    while len(amplitudes) < s:
        address = tuple(int(b) for b in rng.integers(0, 2, size=n_qubits))
        amplitudes[address] = complex(rng.normal(), rng.normal())
    return SparseStateBlock(n_qubits=n_qubits, amplitudes=amplitudes)


def _uniform(n_bits: int) -> UniformSuperpositionBlock:
    return UniformSuperpositionBlock(M=2**n_bits - 1)


def _low_rank(n_qubits: int, seed: int, max_schmidt_rank) -> LowRankStateBlock:
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=2**n_qubits) + 1j * rng.normal(size=2**n_qubits)
    return LowRankStateBlock(
        n_qubits=n_qubits,
        amplitudes=list(vector / np.linalg.norm(vector)),
        cut=4,
        max_schmidt_rank=max_schmidt_rank,
    )


def _mps(n_sites: int, seed: int) -> MPSStateBlock:
    rng = np.random.default_rng(seed)
    shapes = [(1, 2, 2), *([(2, 2, 2)] * (n_sites - 2)), (2, 2, 1)]
    return MPSStateBlock(tensors=[rng.normal(size=s) + 1j * rng.normal(size=s) for s in shapes])


def _piecewise_linear() -> PiecewiseLinearStateBlock:
    return PiecewiseLinearStateBlock(
        n_domain_qubits=8, breakpoints=[50, 120, 200], slopes=[0.1] * 4, intercepts=[0.2] * 4
    )


def _qrom(index_qubits: int, width: int, seed: int) -> QROMBlock:
    rng = np.random.default_rng(seed)
    data = {
        i: tuple(int(b) for b in rng.integers(0, 2, size=width)) for i in range(2**index_qubits)
    }
    return QROMBlock(index_qubits=index_qubits, data=data)


def _slater(n_orbitals: int, n_electrons: int, seed: int) -> SlaterDeterminantBlock:
    rng = np.random.default_rng(seed)
    q = np.linalg.qr(rng.normal(size=(n_orbitals, n_electrons)))[0]
    return SlaterDeterminantBlock(orbital_coefficients=q)


# name -> (builder, measured CNOTs on this branch, 2026-09-11)
_CASES = {
    "sparse_n10_s2": (lambda: _sparse(10, 2, seed=0), 11_894),
    "sparse_n14_s4": (lambda: _sparse(14, 4, seed=1), 100_056),
    "uniform_L8": (lambda: _uniform(8), 4_816),
    "uniform_L12": (lambda: _uniform(12), 82_920),
    "low_rank_n8_rank1": (lambda: _low_rank(8, seed=2, max_schmidt_rank=1), 336),
    "low_rank_n8_full": (lambda: _low_rank(8, seed=3, max_schmidt_rank=None), 568),
    "mps_N8_chi2": (lambda: _mps(8, seed=4), 48),
    "piecewise_linear_n8_P4": (_piecewise_linear, 20_080),
    "qrom_32x5": (lambda: _qrom(5, 5, seed=5), 15_600),
    "slater_N8_M4": (lambda: _slater(8, 4, seed=6), 88),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_cnot_ceiling(case):
    """Regression guard: the current construction's CNOT count, +5 % for
    transpiler-rule churn.  A rise is a regression; a fall means the
    constant in ``_CASES`` should be lowered."""
    build, measured = _CASES[case]
    assert _cnots(build()) <= _ceiling(measured)


_XFAIL = "declared follow-up: "


@pytest.mark.parametrize(
    "case,bound",
    [
        # Gleinig–Hoefler O(s·n): 20 * s * n.
        pytest.param(
            "sparse_n10_s2",
            20 * 2 * 10,
            marks=pytest.mark.xfail(
                strict=True, reason=_XFAIL + "SparseStateBlock -> Gleinig–Hoefler pairwise merge"
            ),
        ),
        pytest.param(
            "sparse_n14_s4",
            20 * 4 * 14,
            marks=pytest.mark.xfail(
                strict=True, reason=_XFAIL + "SparseStateBlock -> Gleinig–Hoefler pairwise merge"
            ),
        ),
        # Shukla–Vedula O(L): 8 * L.
        pytest.param(
            "uniform_L8",
            8 * 8,
            marks=pytest.mark.xfail(
                strict=True,
                reason=_XFAIL + "UniformSuperpositionBlock -> Shukla & Vedula Algorithm 1",
            ),
        ),
        pytest.param(
            "uniform_L12",
            8 * 12,
            marks=pytest.mark.xfail(
                strict=True,
                reason=_XFAIL + "UniformSuperpositionBlock -> Shukla & Vedula Algorithm 1",
            ),
        ),
        # Rank 1 across cut=4 is a product of two 4-qubit state preparations.
        pytest.param(
            "low_rank_n8_rank1",
            64,
            marks=pytest.mark.xfail(
                strict=True, reason=_XFAIL + "LowRankStateBlock -> rank-1 product-state path"
            ),
        ),
        # Woerner–Egger delta form, O(n·P): 40 * n * P.
        pytest.param(
            "piecewise_linear_n8_P4",
            40 * 8 * 4,
            marks=pytest.mark.xfail(
                strict=True,
                reason=_XFAIL + "PiecewiseLinearStateBlock -> per-piece delta rotations",
            ),
        ),
        # Unary iteration, ~4N Toffolis: 30 * N.
        pytest.param(
            "qrom_32x5",
            30 * 32,
            marks=pytest.mark.xfail(strict=True, reason=_XFAIL + "QROMBlock -> unary iteration"),
        ),
    ],
)
def test_cited_scaling(case, bound):
    """The cited construction's scaling times a generous constant.  Strict
    xfail: XPASSes when the named rewrite lands, and the marker comes off."""
    build, _ = _CASES[case]
    assert _cnots(build()) <= bound


def test_mps_beats_dense_synthesis():
    """The one construction that already wins: linear in ``N`` for fixed
    ``chi``, against the O(2^N) dense ``SynthesizedStateBlock`` baseline
    computed here from the same target column."""
    block = _mps(8, seed=4)
    mps_cnots = _cnots(block)
    dense = SynthesizedStateBlock(8, list(block.target_statevector()))
    assert mps_cnots < _cnots(dense)


@pytest.mark.parametrize("k,measured", [(3, 24), (6, 372), (8, 1004), (10, 1972)])
def test_mcx_decomposition_ceiling(k, measured):
    """The ancilla-free ``k``-controlled ``mcx`` decomposition (quadratic in
    ``k``) is the constant underneath every block ceiling above; a change
    here moves all of them."""
    block = SimpleBlock(k + 1)
    block.mcx(*range(k), k)
    assert _cnots(block) <= _ceiling(measured)
