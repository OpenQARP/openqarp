"""Frozen molecular-integral snapshots — pyscf-free test inputs.

Each ``tests/assets/molecules/<name>.npz`` holds restricted RHF/MO-basis
integrals in chemists' notation (``constant``, ``one_electron``,
``two_electron``) plus ``nelectron`` and the SCF energy ``e_hf``.

Provenance: ``assets/molecules/generate.py`` regenerates every file from
live pyscf via ``qarp.operators.pyscf`` (fixtures in ``tests/pyscf_recipes.py``), and
``tests/test_operators/test_molecular_assets.py`` re-checks the stored data
against a live run on gauge-invariant quantities.  The snapshots decouple
the incidental algorithm/block tests from pyscf *and* pin the MO gauge —
the t2 tables in ``tests/pccd_reference_energies.py`` are expressed in the
stored basis, so regenerating is a deliberate act (see ``generate.py``).
The physics oracle tests still derive their references from live pyscf.

Systems and geometries live in ``GEOMETRIES``: three H2 bond lengths
(0.735 Å equilibrium, 0.635 compressed, 3.0 stretched), a linear H4 chain,
and LiH in both atom orders.
"""

from pathlib import Path

import numpy as np

from qarp.operators.integrals import restricted_integrals_to_fermion_operator
from qarp.operators.onv import Onv, onv_from_spatial_occupations

_ASSETS = Path(__file__).parent / "assets" / "molecules"

# pyscf `atom` strings; keys are the stored file stems.
GEOMETRIES = {
    "h2_0.635_sto3g": "H 0 0 0; H 0 0 0.635",
    "h2_0.735_sto3g": "H 0 0 0; H 0 0 0.735",
    "h2_3.000_sto3g": "H 0 0 0; H 0 0 3.0",
    "h4_1.000_sto3g": "H 0 0 0; H 0 0 1; H 0 0 2; H 0 0 3",
    "lih_1.30_sto3g": "Li 0 0 0; H 0 0 1.3",
    "lih_1.59_sto3g": "H 0 0 0; Li 0 0 1.59",
}


def load_integrals(name: str) -> tuple[float, np.ndarray, np.ndarray, int]:
    """(constant, one_electron, two_electron, nelectron) for a stored system."""
    with np.load(_ASSETS / f"{name}.npz") as data:
        return (
            float(data["constant"]),
            data["one_electron"],
            data["two_electron"],
            int(data["nelectron"]),
        )


def hf_energy(name: str) -> float:
    """The stored converged RHF energy of the system."""
    with np.load(_ASSETS / f"{name}.npz") as data:
        return float(data["e_hf"])


def fermion_operator(name: str, threshold: float = 1e-12):
    """Molecular FermionOperator of a stored system."""
    constant, one_electron, two_electron, _ = load_integrals(name)
    return restricted_integrals_to_fermion_operator(constant, one_electron, two_electron, threshold)


def reference_onv(name: str) -> Onv:
    """Closed-shell Aufbau reference ONV (abab) of a stored system."""
    constant, one_electron, _, nelectron = load_integrals(name)
    n_orbitals = one_electron.shape[0]
    occupations = [2] * (nelectron // 2) + [0] * (n_orbitals - nelectron // 2)
    return onv_from_spatial_occupations(occupations)
