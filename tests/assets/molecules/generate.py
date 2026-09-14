"""Regenerate the frozen molecular-integral snapshots in this directory.

Run from anywhere: ``python tests/assets/molecules/generate.py``.  Needs
pyscf (test-only, ``[full-dev]``); systems come from
``tests.molecular_assets.GEOMETRIES``, the mean-field from
``tests.pyscf_recipes.rhf`` at its defaults (RHF, ``conv_tol=1e-10``,
``ao2mo.full(..., aosym="s1")``, no point-group symmetry), basis sto-3g.

Regenerating is a deliberate act, not routine maintenance: the stored
arrays are gauge-sensitive (a different pyscf/BLAS can flip MO signs
without changing the physics), and the pinned t2 amplitude tables in
``tests/pccd_reference_energies.py`` are expressed in the stored basis.
After regenerating, run
``pytest tests/test_operators/test_molecular_assets.py -m ''`` (the
gauge-invariant consistency check is marked ``slow``) and the UPCCD oracle
tests (gauge-sensitive).
"""

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent


def main() -> None:
    from qarp.operators.pyscf import integrals_from_mf
    from tests.molecular_assets import GEOMETRIES
    from tests.pyscf_recipes import rhf

    for name, atom in sorted(GEOMETRIES.items()):
        mf = rhf(atom, "sto3g")
        constant, one_electron, two_electron = integrals_from_mf(mf)
        np.savez(
            _HERE / f"{name}.npz",
            constant=constant,
            one_electron=one_electron,
            two_electron=two_electron,
            nelectron=mf.mol.nelectron,
            e_hf=mf.e_tot,
        )
        print(f"{name}: e_hf = {mf.e_tot:.12f}")


if __name__ == "__main__":
    sys.path.insert(0, str(_HERE.parents[2]))
    main()
