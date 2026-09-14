from benchmarks.algorithms import _families, _qsim

DESCRIPTION, bench, warmup = _families.FACTORIES["qaoa"](_qsim)
