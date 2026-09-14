from benchmarks.statevector import _families, _np

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_phase"](_np)
