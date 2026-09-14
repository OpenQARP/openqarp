"""sparse_construct — pennylane stack shim; the kernel lives in
benchmarks/operators/_families.py, the stack idiom in benchmarks/operators/_pl.py."""

from benchmarks.operators import _families, _pl

DESCRIPTION, bench = _families.FACTORIES["sparse_construct"](_pl)
