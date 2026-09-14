from benchmarks.sampler import _exact, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_sample"](_exact)
