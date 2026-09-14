from benchmarks.sampler import _exact, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["shots_axis"](_exact)
