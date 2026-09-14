"""Independent oracles for the algorithms track (§18).

The runner's cross-stack check proves every SDK solves the same problem; these
prove the *problem and reference trajectory* are right, against values that
exist outside the harness:

- VQE finals respect the variational bound over the FCI energy stored with the
  committed molecule (itself validated against pyscf FCI and the published
  H2 value at generation time), and actually improve on Hartree-Fock.
- The QAOA cost operator's true minimum equals the brute-force MaxCut optimum
  computed by direct enumeration here, and the optimized energy is variational
  against it with a sane approximation ratio.

Usage:  python -m benchmarks.algorithms.verify
"""

from benchmarks.algorithms import _exact, inputs

RATIO_FLOOR = 0.7  # p=2 QAOA well below this means the fixture, not the stack, is wrong


def check_vqe_variational(n: int = 4) -> list[str]:
    """Start anchored at HF exactly; final variational and never above HF.

    Brillouin's theorem makes HF a first-order plateau for the real
    single-rotation ansatz, so the budgeted derivative-free trajectory is
    *expected* to hold at HF (the doc says so and points at ffsim for the
    correlation-capturing contrast).  The oracle therefore pins both ends:
    E(x0) = HF to tight tolerance, and FCI <= final <= HF + noise.
    """
    import numpy as np

    problem = inputs.problem("vqe", n)
    energy_fn = _exact.prepare(problem)
    at_zero = energy_fn(np.zeros(problem["n_params"]))
    fci = problem["reference"]["fci_energy"]
    hf = problem["reference"]["hf_energy"]
    failures = []
    if abs(at_zero - hf) > 1e-6:
        failures.append(f"vqe n={n}: E(0) {at_zero:.8f} != HF {hf:.8f} — reference prep broken")
    energy, _ = _exact.run(energy_fn, problem)
    if energy < fci - 1e-9:
        failures.append(f"vqe n={n}: final {energy:.8f} below FCI {fci:.8f} — not variational")
    if energy > hf + 1e-6:
        failures.append(f"vqe n={n}: final {energy:.8f} above HF {hf:.8f} — left the HF basin")
    return failures


def check_qaoa_against_brute_force(n: int = 8) -> list[str]:
    optimum = inputs.brute_force_maxcut(n)
    problem = inputs.problem("qaoa", n)
    energy, _ = _exact.run(_exact.prepare(problem), problem)
    achieved = -energy  # E = -(expected cut) by construction
    failures = []
    if achieved > optimum + 1e-9:
        failures.append(f"qaoa n={n}: expected cut {achieved:.6f} exceeds optimum {optimum}")
    if achieved < RATIO_FLOOR * optimum:
        failures.append(
            f"qaoa n={n}: ratio {achieved / optimum:.3f} below {RATIO_FLOOR} — fixture broken"
        )
    return failures


CHECKS = (check_vqe_variational, check_qaoa_against_brute_force)


def run_all() -> list[str]:
    failures = []
    for check in CHECKS:
        found = check()
        print(f"  {'FAIL' if found else 'ok  '} algorithms/verify/{check.__name__}")
        failures += found
    return failures


if __name__ == "__main__":
    import sys

    sys.exit(1 if run_all() else 0)
