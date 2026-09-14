"""Workload registry for the end-to-end algorithms track.

The measured quantity is **wall time to a converged result**: the same
classical optimizer (COBYLA, fixed seed point and evaluation budget) drives
each SDK's native energy-evaluation pathway.  Unlike tracks C/D the stacks do
not execute one fixed circuit — they execute the same *optimization*, and the
check is that they land on the same final energy (a deterministic optimizer
over numerically identical energies follows the same trajectory).

The oracle stack is `exact`: the numpy statevector reference driven by the
identical optimizer shell — the reference trajectory every SDK must reproduce.

`ffsim` is the exception, present on the chemistry family only: it simulates a
*different, fermionic* ansatz (LUCJ), because that is the point of ffsim — so
its check is a variational-bound sanity band, not trajectory equality, and the
generated doc labels the column accordingly.
"""

ORACLE_STACK = "exact"
CHECK_RTOL = 1e-5

# qsim is float32-only: energy noise ~1e-7 can steer COBYLA onto a slightly
# different trajectory, so its final energy gets a convergence-scale band.
SINGLE_PRECISION = {"qsim"}
SINGLE_PRECISION_RTOL = 5e-3

# ffsim optimizes a different (fermionic LUCJ) ansatz on purpose — trajectory
# equality is undefined; the family kernel gives it a variational-band check.
DIFFERENT_ANSATZ = {"ffsim"}


def rtol_for(stack: str) -> float:
    return SINGLE_PRECISION_RTOL if stack in SINGLE_PRECISION else CHECK_RTOL


_COMMON = ["exact", "qarpx", "qulacs", "aer", "lightning", "qsim"]

FAMILIES: dict[str, dict] = {
    # size = qubit count selecting the molecule (4 = H2, 12 = LiH).
    "vqe": {
        "sizes": [4, 12],
        "smoke": 4,
        "stacks": _COMMON + ["ffsim"],
    },
    # size = MaxCut graph vertices (3-regular: ring + cross chords).
    "qaoa": {
        "sizes": [8, 12],
        "headroom": [16],
        "smoke": 8,
        "stacks": _COMMON,
    },
}


# COBYLA needs > n_params evaluations just to build its simplex; the budget
# scales with the parameter count so every problem actually converges, and is
# identical across stacks by construction.
def eval_budget(n_params: int) -> int:
    return 40 + 8 * n_params
