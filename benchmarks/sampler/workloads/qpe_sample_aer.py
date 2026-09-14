from benchmarks.sampler import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_sample"](_aer)
