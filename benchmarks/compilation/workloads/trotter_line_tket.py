from benchmarks.compilation import _families, _tket

DESCRIPTION, bench, warmup = _families.FACTORIES["trotter_line"](_tket)
