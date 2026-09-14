from benchmarks.algorithms import _families, _qsim

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe"](_qsim)
