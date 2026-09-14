"""One end-to-end kernel per algorithm family, shared by every stack.

Build phase: constructing the parameterized energy machinery (circuits,
observables, engines).  Run phase: the full optimization to the shared
budget.  The check is the final energy — trajectory-equal across
statevector-exact stacks, a variational band for different-ansatz stacks.
"""

from benchmarks.algorithms import checks, inputs, spec
from benchmarks.common import peak_mem_mib, timed, with_kernel_probe

_DESCRIPTIONS = {
    "vqe": (
        "full VQE to a fixed COBYLA budget: hardware-efficient ansatz "
        f"({inputs.HEA_LAYERS} layers), committed molecular Hamiltonian "
        "(4 q = H2, 12 q = LiH); ffsim optimizes its native LUCJ ansatz instead"
    ),
    "qaoa": (
        "full QAOA (p = 2) for MaxCut on a deterministic 3-regular graph, "
        "same COBYLA budget on every stack; anchored to the brute-force optimum"
    ),
}


def _family(family: str):
    def factory(stack):
        def warmup() -> None:
            # Tiny end-to-end pass drives every lazy import and backend init.
            tiny = inputs.problem(family, 4 if family == "vqe" else 8)
            tiny = dict(tiny, budget=2)
            stack.run(stack.prepare(tiny), tiny)

        def bench(size: int) -> dict:
            problem = inputs.problem(family, size)
            state, t_build = timed(lambda: stack.prepare(problem))
            # Tight, trajectory-independent anchor: the energy at the shared
            # starting point (adapters expose the energy callable directly;
            # different-ansatz stacks return a dict and skip it).
            energy_x0 = float(state(problem["x0"])) if callable(state) else None
            (energy, n_evals), t_run = timed(lambda: stack.run(state, problem))
            meta = {
                "mem_peak_mib": peak_mem_mib(),
                "n_params": problem["n_params"],
                "budget": problem["budget"],
                "n_evals": n_evals,
                **problem["reference"],
            }
            band_ok = None
            if stack.LABEL in spec.DIFFERENT_ANSATZ:
                fci = problem["reference"]["fci_energy"]
                hf = problem["reference"]["hf_energy"]
                band_ok = (fci - 1e-6) <= energy <= (hf + 1e-6)
                meta["ansatz"] = "LUCJ (native fermionic — not trajectory-comparable)"
            return {
                "t_build": t_build,
                "t_run": t_run,
                "check": checks.result_check(energy, energy_x0, band_ok),
                "meta": meta,
            }

        return _DESCRIPTIONS[family], bench, warmup

    return factory


FACTORIES = {family: with_kernel_probe(_family(family)) for family in _DESCRIPTIONS}
