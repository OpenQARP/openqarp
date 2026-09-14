from benchmarks.sampler import _families, _qsim

DESCRIPTION, bench, warmup = _families.FACTORIES["shots_axis"](_qsim)
