from benchmarks.statevector import _families, _qsim

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe_energy"](_qsim)
