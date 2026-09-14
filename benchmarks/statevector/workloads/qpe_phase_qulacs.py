from benchmarks.statevector import _families, _qulacs

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_phase"](_qulacs)
