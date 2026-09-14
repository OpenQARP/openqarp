from benchmarks.algorithms import _exact, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe"](_exact)
