from benchmarks.statevector import _families, _np

DESCRIPTION, bench, warmup = _families.FACTORIES["qft"](_np)
