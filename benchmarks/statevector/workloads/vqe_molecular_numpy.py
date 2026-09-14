from benchmarks.statevector import _families, _np

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe_molecular"](_np)
