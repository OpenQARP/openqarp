from benchmarks.algorithms import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe"](_aer)
