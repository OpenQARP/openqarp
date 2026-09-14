from benchmarks.sampler import _families, _lightning

DESCRIPTION, bench, warmup = _families.FACTORIES["shots_axis"](_lightning)
