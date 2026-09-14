from benchmarks.sampler import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["shots_axis"](_aer)
