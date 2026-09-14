from benchmarks.statevector import _aer, _families

DESCRIPTION, bench, warmup = _families.FACTORIES["qpe_phase"](_aer)
