from benchmarks.compilation import _families, _qk2

DESCRIPTION, bench, warmup = _families.FACTORIES["qv_line"](_qk2)
