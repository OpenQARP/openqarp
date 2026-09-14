from benchmarks.sampler import _families, _lightning

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_sample"](_lightning)
