"""qarp adapter: symbolic block + StateVector primitive, engine built once.

The idiomatic repeated-evaluation path: parameterized angles become sympy
symbols (zero-padded names, so qarp's str-sorted `.symbols` order equals the
parameter index order — §17), the engine is built once, and each COBYLA step
is a single `engine.run({name: value})` rebind.
"""

from benchmarks.algorithms import _shell
from benchmarks.statevector import _qx as sv

LABEL = "qarpx"
kernel_probe = sv.kernel_probe


def _symbol_name(index: int) -> str:
    return f"p{index:04d}"


def prepare(problem: dict):
    import sympy

    from qarp.algorithms import StateVector
    from qarp.blocks import SimpleBlock
    from qarp.engines import QarpEngine
    from qarp.operators import QubitOperator

    n = problem["n_qubits"]
    block = SimpleBlock(n)
    for op in problem["template"]:
        if op[0] == "h":
            block.h(op[1])
        elif op[0] == "cx":
            block.cx(op[1], op[2])
        elif isinstance(op[2], tuple):
            _, index, coefficient = op[2]
            getattr(block, op[0])(op[1], coefficient * sympy.Symbol(_symbol_name(index)))
        else:
            getattr(block, op[0])(op[1], op[2])
    block.build()

    operator = QubitOperator()
    for factors, coeff in problem["terms"]:
        operator += QubitOperator(tuple(factors), coeff)

    primitive = StateVector(operator=operator, ket=block)
    primitive.build()
    engine = QarpEngine()
    engine.build([primitive])

    names = [_symbol_name(k) for k in range(problem["n_params"])]

    def energy_fn(params) -> float:
        binding = {name: float(value) for name, value in zip(names, params, strict=True)}
        return float(complex(engine.run(binding)[0]).real)

    return energy_fn


def run(state, problem: dict):
    return _shell.minimize(state, problem["x0"], problem["budget"], problem["rhobeg"])
