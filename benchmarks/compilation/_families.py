"""One timing/quality kernel per (circuit shape, topology), shared by every stack.

The timed region is the compile call itself; building the SDK's own circuit
object from the canonical fixture is the build phase, so no stack is charged
for its constructor idiom.  Quality metrics are computed after the clock stops,
from the canonical op list, by shared code — "depth" therefore means the same
thing in every column.
"""

from benchmarks.common import peak_mem_mib, timed
from benchmarks.compilation import checks, inputs, spec

_SHAPE_TEXT = {
    "trotter": "all-pairs ZZ + X layer (every qubit pair interacts, so routing is forced)",
    "hea": "hardware-efficient ansatz, circular entangler, 3 layers",
    "qv": "quantum-volume-like random disjoint pairings, 4 layers",
    "mqt": f"MQT Bench `{inputs.MQT_ALGORITHM}`, flattened to the common basis",
}


def _family(name: str):
    shape = spec.FAMILIES[name]["shape"]
    topology = spec.FAMILIES[name]["topology"]
    description = f"{_SHAPE_TEXT[shape]}; routed onto a {topology} coupling map"

    def factory(stack):
        def warmup() -> None:
            ops = inputs.FIXTURES["hea"](4)
            stack.compile_circuit(4, stack.prepare(4, ops, topology))

        def bench(size: int) -> dict:
            ops = inputs.FIXTURES[shape](size)
            prepared, t_build = timed(lambda: stack.prepare(size, ops, topology))
            compiled, t_run = timed(lambda: stack.compile_circuit(size, prepared))
            meta = {"mem_peak_mib": peak_mem_mib()}

            compiled_ops = stack.to_ops(size, compiled)
            meta.update(checks.metrics(compiled_ops, size, topology))
            meta["routed"] = getattr(stack, "ROUTED", True)
            meta["input_gates"] = len(ops)
            return {
                "t_build": t_build,
                "t_run": t_run,
                "check": checks.equivalence_fingerprint(
                    compiled_ops, size, stack.layout_of(compiled)
                ),
                "meta": meta,
            }

        return description, bench, warmup

    return factory


FACTORIES = {name: _family(name) for name in spec.FAMILIES}
