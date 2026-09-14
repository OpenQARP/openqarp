from benchmarks.sampler import _families, _qulacs

DESCRIPTION, bench, warmup = _families.FACTORIES["shots_axis"](_qulacs)
