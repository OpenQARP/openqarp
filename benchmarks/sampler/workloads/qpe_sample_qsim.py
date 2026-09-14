from benchmarks.sampler import _families, _qsim

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_sample"](_qsim)
