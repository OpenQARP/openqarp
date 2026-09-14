from benchmarks.statevector import _families, _qx

DESCRIPTION, bench, warmup = _families.FACTORIES["qft"](_qx)
