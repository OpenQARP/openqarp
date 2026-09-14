from benchmarks.compilation import _families, _qk1

DESCRIPTION, bench, warmup = _families.FACTORIES["qv_grid"](_qk1)
