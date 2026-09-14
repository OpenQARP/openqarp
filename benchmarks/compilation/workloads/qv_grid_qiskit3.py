from benchmarks.compilation import _families, _qk3

DESCRIPTION, bench, warmup = _families.FACTORIES["qv_grid"](_qk3)
