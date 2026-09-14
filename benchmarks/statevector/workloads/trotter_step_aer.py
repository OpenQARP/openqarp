from benchmarks.statevector import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_step"](_aer)
