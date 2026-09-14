"""Generate the committed molecular Hamiltonians for the vqe_molecular family.

Run once, offline, by a developer — never at bench time:

    python -m benchmarks.statevector.data.generate_molecules

pyscf RHF integrals -> openfermion InteractionOperator -> Jordan-Wigner ->
JSON term lists (LSB qubit indices, real/imag coefficient pairs).  The script
validates itself before writing: the dense Hamiltonian's ground state must
reproduce the published FCI energy for H2/STO-3G and the pyscf FCI value
computed here for the others — a generator bug cannot produce a plausible
but wrong committed fixture.
"""

import json
import pathlib

import numpy as np

MOLECULES = {
    # name: (geometry, n_qubits, comment)
    "h2": ([("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.735))], 4, "H2 / STO-3G, 0.735 A"),
    "lih": ([("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.595))], 12, "LiH / STO-3G, 1.595 A"),
    "h2o": (
        [
            ("O", (0.0, 0.0, 0.1173)),
            ("H", (0.0, 0.7572, -0.4692)),
            ("H", (0.0, -0.7572, -0.4692)),
        ],
        14,
        "H2O / STO-3G, experimental geometry",
    ),
}
BASIS = "sto-3g"
OUT_DIR = pathlib.Path(__file__).resolve().parent


def qubit_hamiltonian(geometry):
    from openfermion import InteractionOperator, jordan_wigner
    from openfermion.chem.molecular_data import spinorb_from_spatial
    from pyscf import ao2mo, gto, scf

    mol = gto.M(atom=geometry, basis=BASIS, unit="angstrom")
    mf = scf.RHF(mol).run()
    n_orb = mf.mo_coeff.shape[1]

    one_body = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    # ao2mo gives chemist ordering (pq|rs); openfermion wants physicist <pq|rs>.
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), n_orb)
    two_body = np.asarray(eri.transpose(0, 2, 3, 1), order="C")

    one_spin, two_spin = spinorb_from_spatial(one_body, two_body)
    operator = InteractionOperator(float(mol.energy_nuc()), one_spin, 0.5 * two_spin)
    # Spatial MO integrals ride along in the JSON (chemist ordering for the
    # two-body tensor, as ao2mo emits it) so fermionic simulators (ffsim) can
    # consume the same molecule without pyscf at bench time.
    spatial = {
        "constant": float(mol.energy_nuc()),
        "one_body": one_body.tolist(),
        "two_body_chemist": ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), n_orb).tolist(),
        "n_electrons": int(mol.nelectron),
    }
    return jordan_wigner(operator), mf, mol, spatial


def fci_energy(mf, mol) -> float:
    from pyscf import fci

    return float(fci.FCI(mf).kernel()[0])


def dense_ground_state(qubit_op, n_qubits: int) -> float:
    from openfermion import get_sparse_operator

    sparse = get_sparse_operator(qubit_op, n_qubits=n_qubits)
    if n_qubits <= 12:
        matrix = sparse.toarray()
        assert np.abs(matrix - matrix.conj().T).max() < 1e-9, "Hamiltonian not Hermitian"
        return float(np.linalg.eigvalsh(matrix)[0])
    # 2^14 x 2^14 dense is ~4 GB; Lanczos on the sparse operator instead.
    from scipy.sparse.linalg import eigsh

    assert abs(sparse - sparse.getH()).max() < 1e-9, "Hamiltonian not Hermitian"
    return float(eigsh(sparse, k=1, which="SA", maxiter=5000)[0][0])


def main() -> None:
    for name, (geometry, n_qubits, comment) in MOLECULES.items():
        qubit_op, mf, mol, spatial = qubit_hamiltonian(geometry)
        reference = fci_energy(mf, mol)
        ground = dense_ground_state(qubit_op, n_qubits)
        delta = abs(ground - reference)
        print(f"{name}: JW ground {ground:.6f} Ha, pyscf FCI {reference:.6f} Ha, |d|={delta:.2e}")
        assert delta < 1e-6, f"{name}: JW Hamiltonian does not reproduce FCI"

        terms = []
        for factors, coeff in sorted(qubit_op.terms.items(), key=lambda kv: (len(kv[0]), kv[0])):
            c = complex(coeff)
            terms.append([[[int(q), p] for q, p in factors], c.real, c.imag])
        payload = {
            "comment": comment,
            "basis": BASIS,
            "n_qubits": n_qubits,
            "fci_energy": reference,
            "hf_energy": float(mf.e_tot),
            "n_terms": len(terms),
            "spatial": spatial,
            "terms": terms,
        }
        out = OUT_DIR / f"{name}.json"
        out.write_text(json.dumps(payload))
        print(f"  wrote {out.name}: {len(terms)} terms")


if __name__ == "__main__":
    main()
