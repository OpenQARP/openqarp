"""Molecule fixtures for the pyscf-oracled tests.

pyscf is a test-only dependency here (``[full-dev]``); importing tests guard
with ``pytest.importorskip("pyscf")``.  The mean-field → integrals/ONV
adapters live in ``qarp.operators.pyscf``; what stays here is test policy —
tight convergence and a hard ``converged`` assert — that is not API.
"""


def rhf(atom: str, basis: str, spin: int = 0, symmetry: bool = False):
    """Converged RHF/ROHF mean-field for a molecule."""
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, spin=spin, symmetry=symmetry)
    mol.verbose = 0
    mol.build()
    mf = scf.RHF(mol) if spin == 0 else scf.ROHF(mol)
    mf.conv_tol = 1e-10
    mf.run()
    assert mf.converged
    return mf


def uhf(atom: str, basis: str, spin: int = 0, symmetry: bool = False):
    """Converged UHF mean-field for a molecule."""
    from pyscf import gto, scf

    mol = gto.M(atom=atom, basis=basis, spin=spin, symmetry=symmetry)
    mol.verbose = 0
    mol.build()
    mf = scf.UHF(mol)
    mf.conv_tol = 1e-10
    mf.run()
    assert mf.converged
    return mf
