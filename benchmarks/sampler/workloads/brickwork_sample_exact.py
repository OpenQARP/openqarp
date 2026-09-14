from benchmarks.sampler import _exact, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["brickwork_sample"](_exact)
