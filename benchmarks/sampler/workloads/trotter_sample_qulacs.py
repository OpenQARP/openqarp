from benchmarks.sampler import _families, _qulacs

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_sample"](_qulacs)
