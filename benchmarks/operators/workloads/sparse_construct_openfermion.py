"""sparse_construct — openfermion stack shim; the kernel lives in
benchmarks/operators/_families.py, the stack idiom in benchmarks/operators/_of.py."""

from benchmarks.operators import _families, _of

DESCRIPTION, bench = _families.FACTORIES["sparse_construct"](_of)
