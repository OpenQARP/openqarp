from benchmarks.sampler import _families, _qsim

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_sample"](_qsim)
