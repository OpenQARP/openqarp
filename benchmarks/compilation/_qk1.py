"""qiskit `transpile` at optimization level 1."""

from benchmarks.compilation import _qk

LABEL = "qiskit1"
LEVEL = 1
prepare = _qk.prepare
to_ops = _qk.to_ops
layout_of = _qk.layout_of


def compile_circuit(n: int, prepared):
    return _qk.compile_at(n, prepared, LEVEL)


initial_layout_of = _qk.initial_layout_of
