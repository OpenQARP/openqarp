"""One timing kernel per sampler family, shared by every stack.

The timed region is exactly "produce the shots": circuit and device
construction are the build phase, and reducing the raw output to a
Hamming-weight histogram happens after the clock stops, since each SDK returns
a different container and that conversion is not the thing being measured.

Each factory also returns a `warmup` that samples a trivial circuit; the child
runs it before sampling the memory baseline, so lazy SDK imports and backend
initialization land in the baseline rather than in the measurement.
"""

from benchmarks.common import peak_mem_mib, timed, with_kernel_probe
from benchmarks.sampler import checks, inputs, spec

_DESCRIPTIONS = {
    "brickwork_sample": f"sample {spec.SHOTS} shots from the brickwork circuit",
    "trotter_sample": f"sample {spec.SHOTS} shots after the TFIM Trotter steps",
    "qpe_sample": f"sample {spec.SHOTS} shots from the QPE-shaped circuit",
    "shots_axis": (
        f"shot-count sweep at a fixed {spec.SHOTS_AXIS_QUBITS}-qubit brickwork "
        "circuit; the ladder value is the shot count"
    ),
}


def _sampler_family(family: str):
    def factory(stack):
        def warmup() -> None:
            prepared = stack.build_sampler(1, [("h", 0)], 16)
            stack.to_weights(1, stack.run_sample(1, prepared))

        def bench(size: int) -> dict:
            n_qubits, ops, shots = inputs.workload(family, size)
            prepared, t_build = timed(lambda: stack.build_sampler(n_qubits, ops, shots))
            raw, t_run = timed(lambda: stack.run_sample(n_qubits, prepared))
            meta = {
                "mem_peak_mib": peak_mem_mib(),
                "n_gates": len(ops),
                "n_qubits": n_qubits,
                "shots": shots,
            }
            weight_probs, reported_shots = stack.to_weights(n_qubits, raw)
            return {
                "t_build": t_build,
                "t_run": t_run,
                "check": checks.weight_moments(weight_probs, reported_shots),
                "meta": meta,
            }

        return _DESCRIPTIONS[family], bench, warmup

    return factory


FACTORIES = {family: with_kernel_probe(_sampler_family(family)) for family in _DESCRIPTIONS}
