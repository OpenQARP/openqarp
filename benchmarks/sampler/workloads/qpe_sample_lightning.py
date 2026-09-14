from benchmarks.sampler import _families, _lightning

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_sample"](_lightning)
