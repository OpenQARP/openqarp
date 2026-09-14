from benchmarks.statevector import _families, _np

DESCRIPTION, bench, warmup = _families.FACTORIES["brickwork_threads"](_np)
