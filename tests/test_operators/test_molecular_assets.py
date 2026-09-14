"""The frozen molecular assets stay reproducible from live pyscf.

Regenerates each stored system with the recipe named in
``tests/assets/molecules/generate.py`` and compares gauge-invariant
quantities: the SCF energy, the electron count, and the low-lying spectrum
of the second-quantized Hamiltonian.  Raw integral entries are deliberately
not compared — they are defined only up to MO sign flips, which vary across
pyscf/BLAS builds without changing the physics.
"""

import numpy as np
import pytest

pytest.importorskip("pyscf")

import scipy.sparse.linalg

from qarp.operators.pyscf import fermion_operator_from_mf
from tests.molecular_assets import (
    _ASSETS,
    GEOMETRIES,
    fermion_operator,
    hf_energy,
    load_integrals,
)
from tests.pyscf_recipes import rhf


def test_every_asset_has_a_geometry():
    assert {path.stem for path in _ASSETS.glob("*.npz")} == set(GEOMETRIES)


def _low_spectrum(op, k=4):
    return np.sort(
        scipy.sparse.linalg.eigsh(op.sparse_matrix(), k=k, which="SA", return_eigenvectors=False)
    )


# `slow`: ~30 s of SCF + Lanczos re-checking frozen bytes that cannot drift
# on their own.  Run it after regenerating assets or bumping pyscf.
@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(GEOMETRIES))
def test_stored_assets_match_live_pyscf(name):
    mf = rhf(GEOMETRIES[name], "sto3g")
    assert np.isclose(mf.e_tot, hf_energy(name), atol=1e-9)
    *_, nelectron = load_integrals(name)
    assert nelectron == mf.mol.nelectron
    assert np.allclose(
        _low_spectrum(fermion_operator(name)),
        _low_spectrum(fermion_operator_from_mf(mf)),
        atol=1e-8,
    )
