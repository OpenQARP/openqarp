from benchmarks.statevector import _families, _lightning

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_phase"](_lightning)
