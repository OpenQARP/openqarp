from benchmarks.sampler import _exact, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_sample"](_exact)
