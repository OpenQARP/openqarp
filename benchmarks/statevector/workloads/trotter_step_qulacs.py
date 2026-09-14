from benchmarks.statevector import _families, _qulacs

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_step"](_qulacs)
