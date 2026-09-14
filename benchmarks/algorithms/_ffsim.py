"""ffsim adapter — the different-ansatz column, chemistry rows only.

ffsim simulates number-conserving fermionic circuits; forcing it through the
qubit HEA would erase exactly the structure that makes it fast.  It optimizes
its native **LUCJ ansatz** (UCJOpSpinBalanced, one repetition) from the
Hartree-Fock state, built from the same committed MO integrals as everyone
else's Hamiltonian.  Its check is therefore a variational band
(FCI <= E <= HF), not trajectory equality — the generated doc says so.

`prepare` self-validates: at zero parameters the LUCJ circuit is the identity,
so the energy must equal the stored HF energy — a wrong tensor convention
lands far away and fails loudly rather than benchmarking a wrong molecule.
"""

import numpy as np

from benchmarks.algorithms import _shell

LABEL = "ffsim"


def prepare(problem: dict):
    import ffsim

    from benchmarks.statevector.inputs import MOLECULES, molecular_terms

    n = problem["n_qubits"]
    if n not in MOLECULES:
        raise RuntimeError("ffsim column exists only for molecular problems")
    _, payload = molecular_terms(n)
    spatial = payload["spatial"]

    norb = len(spatial["one_body"])
    n_electrons = spatial["n_electrons"]
    nelec = (n_electrons // 2 + n_electrons % 2, n_electrons // 2)

    hamiltonian = ffsim.MolecularHamiltonian(
        one_body_tensor=np.array(spatial["one_body"]),
        two_body_tensor=np.array(spatial["two_body_chemist"]),
        constant=float(spatial["constant"]),
    )
    linop = ffsim.linear_operator(hamiltonian, norb=norb, nelec=nelec)
    reference = ffsim.hartree_fock_state(norb, nelec)

    n_reps = 1
    n_params = ffsim.UCJOpSpinBalanced.n_params(norb, n_reps)

    def energy_fn(params) -> float:
        operator = ffsim.UCJOpSpinBalanced.from_parameters(
            np.asarray(params, dtype=float), norb=norb, n_reps=n_reps
        )
        state = ffsim.apply_unitary(reference, operator, norb=norb, nelec=nelec)
        return float(np.real(np.vdot(state, linop @ state)))

    hf_check = energy_fn(np.zeros(n_params))
    if abs(hf_check - payload["hf_energy"]) > 1e-6:
        raise RuntimeError(
            f"ffsim HF self-check failed: {hf_check:.6f} vs stored {payload['hf_energy']:.6f} "
            "— integral convention mismatch"
        )

    return {
        "energy_fn": energy_fn,
        "x0": np.zeros(n_params),
        "budget": __import__("benchmarks.algorithms.spec", fromlist=["x"]).eval_budget(n_params),
    }


def run(state, problem: dict):
    return _shell.minimize(state["energy_fn"], state["x0"], state["budget"], 0.05)
