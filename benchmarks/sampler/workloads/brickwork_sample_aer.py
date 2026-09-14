from benchmarks.sampler import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["brickwork_sample"](_aer)
