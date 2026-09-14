from benchmarks.compilation import _families, _tket

DESCRIPTION, bench, warmup = _families.FACTORIES["qv_grid"](_tket)
