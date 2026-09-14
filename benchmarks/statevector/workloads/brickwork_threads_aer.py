from benchmarks.statevector import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["brickwork_threads"](_aer)
