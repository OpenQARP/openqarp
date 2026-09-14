from benchmarks.statevector import _families, _lightning

DESCRIPTION, bench, warmup = _families.FACTORIES["vqe_terms"](_lightning)
