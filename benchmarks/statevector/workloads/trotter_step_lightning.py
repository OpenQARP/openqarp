from benchmarks.statevector import _families, _lightning

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_step"](_lightning)
