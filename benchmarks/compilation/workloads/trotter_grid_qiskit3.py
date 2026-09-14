from benchmarks.compilation import _families, _qk3

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_grid"](_qk3)
