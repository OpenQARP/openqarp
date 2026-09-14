from benchmarks.sampler import _families, _qulacs

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_sample"](_qulacs)
