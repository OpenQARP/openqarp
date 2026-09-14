from benchmarks.statevector import _families, _qulacs

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe_energy"](_qulacs)
