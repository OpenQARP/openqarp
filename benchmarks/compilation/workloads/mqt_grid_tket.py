from benchmarks.compilation import _families, _tket

DESCRIPTION, bench, warmup = _families.FACTORIES["mqt_grid"](_tket)
