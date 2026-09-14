"""sparse_construct — qarpx stack shim; the kernel lives in
benchmarks/operators/_families.py, the stack idiom in benchmarks/operators/_qx.py."""

from benchmarks.operators import _families, _qx

DESCRIPTION, bench = _families.FACTORIES["sparse_construct"](_qx)
