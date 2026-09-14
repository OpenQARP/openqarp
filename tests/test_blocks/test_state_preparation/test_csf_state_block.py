"""Independent-oracle tests for direct CSF (spin-adapted) state preparation."""

import numpy as np
import pytest

from qarp.blocks import CSFStateBlock
from qarp.operators import FermionOperator, JordanWigner


def _s_squared_operator(n_spatial: int) -> FermionOperator:
    """S^2 = Sz^2 + (S+ S- + S- S+) / 2, built from scratch in the
    second-quantized alpha/beta (abab, §1) convention — independent of
    CSFStateBlock's own genealogical-coupling construction entirely."""
    sz = FermionOperator()
    s_plus = FermionOperator()
    s_minus = FermionOperator()
    for p in range(n_spatial):
        alpha, beta = 2 * p, 2 * p + 1
        sz += FermionOperator(((alpha, 1), (alpha, 0)), 0.5)
        sz += FermionOperator(((beta, 1), (beta, 0)), -0.5)
        s_plus += FermionOperator(((alpha, 1), (beta, 0)), 1.0)
        s_minus += FermionOperator(((beta, 1), (alpha, 0)), 1.0)
    return sz * sz + 0.5 * (s_plus * s_minus + s_minus * s_plus)


def _sz_operator(n_spatial: int) -> FermionOperator:
    sz = FermionOperator()
    for p in range(n_spatial):
        alpha, beta = 2 * p, 2 * p + 1
        sz += FermionOperator(((alpha, 1), (alpha, 0)), 0.5)
        sz += FermionOperator(((beta, 1), (beta, 0)), -0.5)
    return sz


def _expectation(psi: np.ndarray, fermion_op: FermionOperator) -> complex:
    matrix = JordanWigner().encode_operator(fermion_op).sparse_matrix()
    return complex(np.vdot(psi, matrix.dot(psi)))


@pytest.mark.parametrize(
    "kwargs,n_spatial,expected_S,expected_Ms",
    [
        (
            dict(
                n_spatial_orbitals=3,
                core_orbitals=[0],
                open_shell_orbitals=[1, 2],
                coupling_path=[0.5, 0],
                Ms=0,
            ),
            3,
            0,
            0,
        ),
        (
            dict(
                n_spatial_orbitals=3,
                core_orbitals=[0],
                open_shell_orbitals=[1, 2],
                coupling_path=[0.5, 1],
                Ms=0,
            ),
            3,
            1,
            0,
        ),
        (
            dict(
                n_spatial_orbitals=3,
                core_orbitals=[0],
                open_shell_orbitals=[1, 2],
                coupling_path=[0.5, 1],
                Ms=1,
            ),
            3,
            1,
            1,
        ),
        (
            dict(
                n_spatial_orbitals=4,
                core_orbitals=[0],
                open_shell_orbitals=[1, 2, 3],
                coupling_path=[0.5, 0, 0.5],
                Ms=0.5,
            ),
            4,
            0.5,
            0.5,
        ),
        (
            dict(
                n_spatial_orbitals=4,
                core_orbitals=[0],
                open_shell_orbitals=[1, 2, 3],
                coupling_path=[0.5, 1, 0.5],
                Ms=0.5,
            ),
            4,
            0.5,
            0.5,
        ),
        (
            dict(
                n_spatial_orbitals=4,
                core_orbitals=[0],
                open_shell_orbitals=[1, 2, 3],
                coupling_path=[0.5, 1, 1.5],
                Ms=1.5,
            ),
            4,
            1.5,
            1.5,
        ),
        (
            dict(
                n_spatial_orbitals=2,
                core_orbitals=[0, 1],
                open_shell_orbitals=[],
                coupling_path=[],
                Ms=0,
            ),
            2,
            0,
            0,
        ),
    ],
    ids=[
        "singlet_2open",
        "triplet_2open_ms0",
        "triplet_2open_ms1",
        "doublet_3open_pathA",
        "doublet_3open_pathB",
        "quartet_3open",
        "closed_shell",
    ],
)
def test_is_genuine_spin_eigenstate(kwargs, n_spatial, expected_S, expected_Ms):
    """The prepared state is an exact eigenstate of S^2 (eigenvalue S(S+1))
    and Sz (eigenvalue Ms) — built from qarp's own FermionOperator/JW
    machinery, entirely independent of CSFStateBlock's construction code."""
    block = CSFStateBlock(**kwargs)
    block.build()
    psi = block.statevector()

    s_squared = _expectation(psi, _s_squared_operator(n_spatial))
    np.testing.assert_allclose(s_squared, expected_S * (expected_S + 1), atol=1e-8)

    sz = _expectation(psi, _sz_operator(n_spatial))
    np.testing.assert_allclose(sz, expected_Ms, atol=1e-8)


def test_hand_verified_singlet_amplitudes():
    """The textbook 2-electron singlet: (|up down> - |down up>) / sqrt(2)."""
    block = CSFStateBlock(
        n_spatial_orbitals=2,
        core_orbitals=[],
        open_shell_orbitals=[0, 1],
        coupling_path=[0.5, 0],
        Ms=0,
    )
    block.build()
    psi = block.statevector()
    # orbital 0 alpha-occ, orbital 1 beta-occ -> spin orbitals (0,3): idx=0b1001=9
    # orbital 0 beta-occ, orbital 1 alpha-occ -> spin orbitals (1,2): idx=0b0110=6
    assert np.isclose(psi[9].real, 1 / np.sqrt(2), atol=1e-10)
    assert np.isclose(psi[6].real, -1 / np.sqrt(2), atol=1e-10)
    nonzero = np.flatnonzero(np.abs(psi) > 1e-10)
    assert sorted(nonzero.tolist()) == [6, 9]


def test_two_doublet_paths_are_orthogonal():
    """The 3-open-shell-electron doublet pair (allyl-radical case) must be
    linearly independent, distinct CSFs, not the same state twice."""
    a = CSFStateBlock(
        n_spatial_orbitals=4,
        core_orbitals=[0],
        open_shell_orbitals=[1, 2, 3],
        coupling_path=[0.5, 0, 0.5],
        Ms=0.5,
    )
    b = CSFStateBlock(
        n_spatial_orbitals=4,
        core_orbitals=[0],
        open_shell_orbitals=[1, 2, 3],
        coupling_path=[0.5, 1, 0.5],
        Ms=0.5,
    )
    a.build()
    b.build()
    overlap = np.vdot(a.statevector(), b.statevector())
    assert abs(overlap) < 1e-8


@pytest.mark.parametrize(
    "open_shell,S,Ms,explicit_path",
    [
        ([1, 2], 0, 0, [0.5, 0]),
        ([1, 2, 3], 0.5, 0.5, [0.5, 1, 0.5]),
        ([1, 2, 3], 1.5, 1.5, [0.5, 1, 1.5]),
        ([1, 2, 3], 1.5, -0.5, [0.5, 1, 1.5]),
        ([1, 2, 3, 4], 0, 0, [0.5, 1, 0.5, 0]),
        ([], 0, 0, []),
    ],
    ids=[
        "singlet_2open",
        "doublet_3open",
        "quartet_3open",
        "quartet_ms-1/2",
        "singlet_4open",
        "closed_shell",
    ],
)
def test_S_selects_canonical_coupling_path(open_shell, S, Ms, explicit_path):
    """``S=…`` with no ``coupling_path`` prepares the same state as the
    hand-written up-then-down path."""
    n = 5
    by_S = CSFStateBlock(n, [0], open_shell, Ms=Ms, S=S)
    by_path = CSFStateBlock(n, [0], open_shell, explicit_path, Ms)
    assert by_S.coupling_path == by_path.coupling_path
    by_S.build()
    by_path.build()
    np.testing.assert_allclose(by_S.statevector(), by_path.statevector(), atol=1e-12)


def test_canonical_path_is_a_spin_eigenstate():
    """The canonical doublet for 3 open shells is a genuine S=1/2 eigenstate
    (independent S^2 oracle, not only equality with the explicit path)."""
    block = CSFStateBlock(4, [0], [1, 2, 3], Ms=0.5, S=0.5)
    block.build()
    psi = block.statevector()
    np.testing.assert_allclose(_expectation(psi, _s_squared_operator(4)), 0.75, atol=1e-8)


def test_rejects_both_coupling_path_and_S():
    with pytest.raises(ValueError, match="exactly one"):
        CSFStateBlock(3, [0], [1, 2], coupling_path=[0.5, 0], Ms=0, S=0)


def test_rejects_neither_coupling_path_nor_S():
    with pytest.raises(ValueError, match="exactly one"):
        CSFStateBlock(3, [0], [1, 2], Ms=0)


def test_rejects_missing_Ms():
    with pytest.raises(ValueError, match="Ms"):
        CSFStateBlock(3, [0], [1, 2], S=0)


@pytest.mark.parametrize(
    "open_shell,S",
    [([1, 2], 0.5), ([1, 2], 2), ([1, 2, 3], 0), ([], 0.5), ([1, 2], -1)],
    ids=["parity_2open", "too_high_2open", "parity_3open", "closed_shell_doublet", "negative"],
)
def test_rejects_unreachable_S(open_shell, S):
    with pytest.raises(ValueError, match="not reachable"):
        CSFStateBlock(4, [0], open_shell, Ms=0, S=S)


def test_rejects_mismatched_coupling_path_length():
    with pytest.raises(ValueError):
        CSFStateBlock(
            n_spatial_orbitals=3,
            core_orbitals=[0],
            open_shell_orbitals=[1, 2],
            coupling_path=[0.5],
            Ms=0,
        )


def test_rejects_invalid_coupling_step():
    with pytest.raises(ValueError):
        CSFStateBlock(
            n_spatial_orbitals=3,
            core_orbitals=[0],
            open_shell_orbitals=[1, 2],
            coupling_path=[0.5, 1.5],  # jump of 1, not 1/2
            Ms=0,
        )


def test_rejects_unreachable_Ms():
    with pytest.raises(ValueError):
        CSFStateBlock(
            n_spatial_orbitals=3,
            core_orbitals=[0],
            open_shell_orbitals=[1, 2],
            coupling_path=[0.5, 0],
            Ms=1,  # singlet can only have Ms=0
        )


def test_rejects_overlapping_core_and_open_shell():
    with pytest.raises(ValueError):
        CSFStateBlock(
            n_spatial_orbitals=3,
            core_orbitals=[0, 1],
            open_shell_orbitals=[1, 2],
            coupling_path=[0.5, 0],
            Ms=0,
        )
