"""The shared classical optimizer shell.

One COBYLA configuration for every stack — same start, same trust region,
same evaluation budget — so the only thing that differs between columns is
the quantum-side energy evaluation.  scipy's COBYLA is deterministic: given
identical energies it takes identical steps.  The trust radius is part of
the *problem* (VQE probes 0.05 rad to stay in the Hartree-Fock basin; QAOA
probes 0.4 on its pi-scale angles), never per stack.
"""

DEFAULT_RHOBEG = 0.4


def minimize(energy_fn, x0, budget: int, rhobeg: float = DEFAULT_RHOBEG):
    """Run COBYLA to the fixed budget; returns (final_energy, n_evals)."""
    import numpy as np
    from scipy.optimize import minimize as scipy_minimize

    result = scipy_minimize(
        energy_fn,
        np.asarray(x0, dtype=float),
        method="COBYLA",
        options={"maxiter": budget, "rhobeg": rhobeg},
    )
    return float(result.fun), int(result.nfev)
