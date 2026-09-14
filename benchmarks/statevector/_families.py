"""One timing kernel per workload family, shared by every stack.

Factored here rather than copied into the 30 workload shims so the measurement
protocol is identical across stacks by construction: same split between build
and run, same point at which peak memory is pinned, same reduction to a check.

Peak RSS is captured immediately after the timed run and before the check pass
— fingerprinting allocates, and this track publishes memory.

Each factory also returns a `warmup` that drives the family's own code path at
a trivial size.  The child runs it *before* sampling the memory baseline: the
adapters import their SDKs lazily inside functions (§15 style), so without it
the whole import and engine-initialization footprint would be charged to the
simulation.  It also moves one-time backend initialization out of `t_run`.
"""

from benchmarks.common import peak_mem_mib, timed, with_kernel_probe
from benchmarks.statevector import checks, inputs

_DESCRIPTIONS = {
    "brickwork": (
        f"{inputs.BRICKWORK_LAYERS} layers of random two-qubit blocks "
        "(2 CX + 6 rotations each) in an even/odd brick pattern; gate count O(n)"
    ),
    "qft": "textbook QFT including final bit-reversal swaps; gate count O(n^2)",
    "trotter_step": (
        f"{inputs.TROTTER_STEPS} first-order Trotter steps of the 1D "
        "transverse-field Ising chain (J=1.0, h=0.6, dt=0.1)"
    ),
    "vqe_energy": (
        f"one VQE energy evaluation: {inputs.VQE_LAYERS}-layer hardware-efficient "
        "ansatz, expectation of the Jordan-Wigner Fermi-Hubbard Hamiltonian "
        f"(t={inputs.HUBBARD_T}, U={inputs.HUBBARD_U})"
    ),
    "qpe_phase": (
        f"QPE-shaped circuit: {inputs.QPE_ANCILLAS} ancillas, controlled "
        "Trotter evolution of a TFIM system register, inverse QFT on the ancillas"
    ),
}


def _state_family(family: str):
    def factory(stack):
        def warmup() -> None:
            prepared = stack.build_circuit(1, [("h", 0)])
            stack.run_state(1, prepared)

        def bench(size: int) -> dict:
            n_qubits, ops = inputs.CIRCUITS[family](size)
            prepared, t_build = timed(lambda: stack.build_circuit(n_qubits, ops))
            psi, t_run = timed(lambda: stack.run_state(n_qubits, prepared))
            meta = {"mem_peak_mib": peak_mem_mib(), "n_gates": len(ops)}
            if stack.MSB:
                psi = checks.to_lsb(psi, n_qubits)
            return {
                "t_build": t_build,
                "t_run": t_run,
                "check": checks.state_fingerprint(psi, n_qubits),
                "meta": meta,
            }

        return _DESCRIPTIONS[family], bench, warmup

    return factory


def _energy_family(family: str):
    def factory(stack):
        def warmup() -> None:
            # Hamiltonian assembly imports openfermion; that is input
            # generation, so it belongs in the baseline, not the measurement.
            inputs.hubbard_terms(4)
            prepared = stack.build_energy(1, [("h", 0)], [(((0, "Z"),), 1.0 + 0j)])
            stack.run_energy(1, prepared)

        def bench(size: int) -> dict:
            n_qubits, ops = inputs.CIRCUITS[family](size)
            terms = inputs.hubbard_terms(size)
            prepared, t_build = timed(lambda: stack.build_energy(n_qubits, ops, terms))
            energy, t_run = timed(lambda: stack.run_energy(n_qubits, prepared))
            meta = {
                "mem_peak_mib": peak_mem_mib(),
                "n_gates": len(ops),
                "n_terms": len(terms),
            }
            return {
                "t_build": t_build,
                "t_run": t_run,
                "check": checks.energy_fingerprint(energy),
                "meta": meta,
            }

        return _DESCRIPTIONS[family], bench, warmup

    return factory


def _term_axis_family():
    """Energy kernel at fixed width; `size` is the % of Hubbard terms kept.

    Separates the two costs "Hamiltonian size" conflates: the statevector
    (2^n, held fixed here) and the expectation loop (linear in terms, swept).
    """
    from benchmarks.statevector import spec

    def factory(stack):
        def warmup() -> None:
            inputs.hubbard_terms(4)
            prepared = stack.build_energy(1, [("h", 0)], [(((0, "Z"),), 1.0 + 0j)])
            stack.run_energy(1, prepared)

        def bench(size: int) -> dict:
            n_qubits = spec.TERM_AXIS_QUBITS
            _, ops = inputs.vqe_ansatz(n_qubits)
            terms = inputs.hubbard_terms(n_qubits, fraction=size / 100.0)
            prepared, t_build = timed(lambda: stack.build_energy(n_qubits, ops, terms))
            energy, t_run = timed(lambda: stack.run_energy(n_qubits, prepared))
            meta = {"mem_peak_mib": peak_mem_mib(), "n_gates": len(ops), "n_terms": len(terms)}
            return {
                "t_build": t_build,
                "t_run": t_run,
                "check": checks.energy_fingerprint(energy),
                "meta": meta,
            }

        return (
            (
                f"one energy evaluation at fixed n = {spec.TERM_AXIS_QUBITS}; the row's "
                "size is the percentage of Fermi-Hubbard terms kept — the term-count "
                "axis of the cost, with the statevector held fixed"
            ),
            bench,
            warmup,
        )

    return factory


def _molecular_family():
    """Energy kernel over the committed molecular Hamiltonians (data/)."""

    def factory(stack):
        def warmup() -> None:
            prepared = stack.build_energy(1, [("h", 0)], [(((0, "Z"),), 1.0 + 0j)])
            stack.run_energy(1, prepared)

        def bench(size: int) -> dict:
            terms, payload = inputs.molecular_terms(size)
            _, ops = inputs.vqe_ansatz(size)
            prepared, t_build = timed(lambda: stack.build_energy(size, ops, terms))
            energy, t_run = timed(lambda: stack.run_energy(size, prepared))
            meta = {
                "mem_peak_mib": peak_mem_mib(),
                "n_gates": len(ops),
                "n_terms": len(terms),
                "molecule": payload["comment"],
            }
            return {
                "t_build": t_build,
                "t_run": t_run,
                "check": checks.energy_fingerprint(energy),
                "meta": meta,
            }

        return (
            (
                "one energy evaluation of a pyscf-generated molecular Hamiltonian "
                "(4 q = H2, 12 q = LiH, 14 q = H2O, STO-3G; committed JSON, FCI-validated "
                "at generation time), same hardware-efficient ansatz as vqe_energy"
            ),
            bench,
            warmup,
        )

    return factory


def _thread_axis_family():
    """Brickwork at fixed n; `size` is the thread count set by the runner.

    The kernel is identical to `brickwork` at THREAD_AXIS_QUBITS — the swept
    variable lives entirely in the child's environment (OMP_NUM_THREADS and
    friends), so this factory only fixes the width.
    """
    from benchmarks.statevector import spec

    def factory(stack):
        def warmup() -> None:
            prepared = stack.build_circuit(1, [("h", 0)])
            stack.run_state(1, prepared)

        def bench(size: int) -> dict:
            del size  # the thread count; applied by the runner via the environment
            n_qubits, ops = inputs.brickwork(spec.THREAD_AXIS_QUBITS)
            prepared, t_build = timed(lambda: stack.build_circuit(n_qubits, ops))
            psi, t_run = timed(lambda: stack.run_state(n_qubits, prepared))
            meta = {"mem_peak_mib": peak_mem_mib(), "n_gates": len(ops)}
            if stack.MSB:
                psi = checks.to_lsb(psi, n_qubits)
            return {
                "t_build": t_build,
                "t_run": t_run,
                "check": checks.state_fingerprint(psi, n_qubits),
                "meta": meta,
            }

        return (
            (
                f"the brickwork circuit at fixed n = {spec.THREAD_AXIS_QUBITS}; the "
                "row's size is the thread count (OMP and friends set in the child "
                "environment) — the core-scaling axis"
            ),
            bench,
            warmup,
        )

    return factory


FACTORIES = {
    name: with_kernel_probe(factory)
    for name, factory in {
        "brickwork": _state_family("brickwork"),
        "qft": _state_family("qft"),
        "trotter_step": _state_family("trotter_step"),
        "qpe_phase": _state_family("qpe_phase"),
        "vqe_energy": _energy_family("vqe_energy"),
        "vqe_terms": _term_axis_family(),
        "vqe_molecular": _molecular_family(),
        "brickwork_threads": _thread_axis_family(),
    }.items()
}
