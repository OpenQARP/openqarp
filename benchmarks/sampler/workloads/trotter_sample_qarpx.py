from benchmarks.sampler import _families, _qx

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_sample"](_qx)
