from benchmarks.statevector import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe_energy"](_aer)
