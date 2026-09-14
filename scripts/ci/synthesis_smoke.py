"""Wheel smoke test — exercises the CSD/QSD synthesis path.

Run by cibuildwheel as `test-command` against the freshly built+repaired wheel,
in an isolated environment with cwd set to a temp dir (so `import qarp` resolves
to the installed wheel, not the source tree). Exits non-zero on failure, so a
broken CSD — or, on a QARP_USE_LAPACK=ON build, a mis-linked BLAS — fails the
build.

Mirrors tests/test_blocks/test_primitives/test_synthetized_unitary_block.py.
"""

import sys

import numpy as np

import qarp
import qarpx as qx
from qarp.blocks import SynthesizedUnitaryBlock

print(f"  qarp  {qarp.__version__} from {qarp.__file__}")
print(f"  qarpx from {qx.__file__}")


def _recovered_unitary(block):
    return np.array(qx.QarpSimulator().unitary_matrix(block.flatten(), block.n_qubits))


def main() -> int:
    ok = True
    for n in (1, 2, 3):
        np.random.seed(42 + n)
        dim = 2**n
        a = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
        q, _ = np.linalg.qr(a)  # Haar-ish unitary

        block = SynthesizedUnitaryBlock(q).build()  # → C++ QSD → CSD
        u = _recovered_unitary(block)

        # Recovery up to a global phase: Q† U should be a constant phase × I.
        m = q.conj().T @ u
        phase = m[0, 0] / abs(m[0, 0]) if abs(m[0, 0]) > 1e-10 else 1.0
        err = float(np.linalg.norm(m - phase * np.eye(dim)))
        status = "OK" if err < 1e-9 else "FAIL"
        ok = ok and err < 1e-9
        print(f"  n={n} dim={dim}  synth-recovery-err={err:.2e}  {status}")

    print("SYNTH_SMOKE_OK" if ok else "SYNTH_SMOKE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
