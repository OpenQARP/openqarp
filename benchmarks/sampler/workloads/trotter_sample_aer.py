from benchmarks.sampler import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_sample"](_aer)
