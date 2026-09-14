from benchmarks.algorithms import _families, _lightning

DESCRIPTION, bench, warmup = _families.FACTORIES["qaoa"](_lightning)
