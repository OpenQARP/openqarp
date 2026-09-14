from benchmarks.algorithms import _families, _qulacs

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe"](_qulacs)
