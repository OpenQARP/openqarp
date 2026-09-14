from benchmarks.statevector import _families, _qx

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe_molecular"](_qx)
