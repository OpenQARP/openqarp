from benchmarks.compilation import _families, _qk1

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_grid"](_qk1)
