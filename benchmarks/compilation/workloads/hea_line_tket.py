from benchmarks.compilation import _families, _tket

DESCRIPTION, bench, warmup = _families.FACTORIES["hea_line"](_tket)
