"""op_product — qiskit stack shim; the kernel lives in
benchmarks/operators/_families.py, the stack idiom in benchmarks/operators/_qk.py."""

from benchmarks.operators import _families, _qk

DESCRIPTION, bench = _families.FACTORIES["op_product"](_qk)
