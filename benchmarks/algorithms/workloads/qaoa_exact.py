from benchmarks.algorithms import _exact, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["qaoa"](_exact)
