"""Workload registry for the statevector track.

Sizes are qubit counts.  The published ladder stops at n = 20 (1 M amplitudes)
so a full track stays inside the iteration-speed budget; `--headroom` appends
the larger sizes for a deliberate run.

The oracle stack is `numpy` — a direct gate-by-gate statevector reference in
this repo, independent of every SDK under test (§18).  It doubles as the
baseline column: what the optimized kernels actually buy over a plain
`einsum` loop.
"""

ORACLE_STACK = "numpy"
CHECK_RTOL = 1e-9

# qsim's published wheels are float32-only, so its fingerprints drift from the
# double-precision oracle by accumulated round-off.  That is a genuine
# speed/accuracy trade the table discloses, not a failure.
#
# Nearly all of that drift lands in the norm rather than the shape: on Apple
# silicon qsim returns n = 20 states with |psi| ~ 1.00015 (vs ~1+1e-7 on x86),
# and the probability fields then inherit twice that error while the overlap
# inherits it once — five fields failing on one scalar.  `state_fingerprint`
# divides normalization out and `agree` gives `norm` its own budget, which
# leaves the shape residual at ~1e-6 on both hosts, so this stays tight.
# Structural errors (wrong convention, bit order, or decomposition) are O(1)
# and cannot hide under any of these numbers.
SINGLE_PRECISION = {"qsim"}
SINGLE_PRECISION_RTOL = 1e-5

_ALL = ["numpy", "qarpx", "qulacs", "aer", "lightning", "qsim"]


def rtol_for(stack: str) -> float:
    return SINGLE_PRECISION_RTOL if stack in SINGLE_PRECISION else CHECK_RTOL


FAMILIES: dict[str, dict] = {
    "brickwork": {
        "sizes": [8, 12, 16, 20],
        "headroom": [22, 24, 26],
        "smoke": 4,
        "stacks": _ALL,
    },
    "qft": {
        "sizes": [8, 12, 16, 20],
        "headroom": [22, 24],
        "smoke": 4,
        "stacks": _ALL,
    },
    "trotter_step": {
        "sizes": [8, 12, 16, 20],
        "headroom": [22, 24],
        "smoke": 4,
        "stacks": _ALL,
    },
    # Fermi-Hubbard ladder: 4 q = 1x2 sites, 8 q = 2x2, 12 q = 2x3, 16 q = 2x4.
    "vqe_energy": {
        "sizes": [4, 8, 12],
        "headroom": [16],
        "smoke": 4,
        "stacks": _ALL,
    },
    # n = n_ancilla (4, fixed) + n_system; the ladder grows the system register.
    "qpe_phase": {
        "sizes": [6, 10, 14],
        "headroom": [16, 18],
        "smoke": 6,
        "stacks": _ALL,
    },
    # Axis families: one swept parameter each, so core-scaling and term-scaling
    # are answered once instead of multiplying every row.
    #
    # "size" is the PERCENTAGE of Hubbard terms kept, at fixed n = TERM_AXIS_QUBITS.
    "vqe_terms": {
        "sizes": [25, 50, 100],
        "smoke": 25,
        "stacks": _ALL,
    },
    # "size" is the THREAD COUNT, at fixed n = THREAD_AXIS_QUBITS; the runner
    # sets the pool variables in the child environment per size.
    "brickwork_threads": {
        "sizes": [1, 2, 4, 8],
        "headroom": [16],
        "smoke": 1,
        "stacks": _ALL,
        "axis": "threads",
    },
    # "size" is the qubit count selecting the molecule: 4 = H2, 12 = LiH,
    # 14 = H2O (STO-3G, committed JSON under data/ — no pyscf at bench time).
    "vqe_molecular": {
        "sizes": [4, 12, 14],
        "smoke": 4,
        "stacks": _ALL,
    },
}

TERM_AXIS_QUBITS = 12
THREAD_AXIS_QUBITS = 20

# Gate fusion, per stack — every stack runs the fusion its engine offers, at
# the setting `python -m benchmarks.statevector.tune` measured fastest on the
# publishing host: single thread, geometric mean of the run phase over the
# three state families at n >= 16 (the sizes where a pass over the state,
# not dispatch, is the cost).  Fusion is part of the *run* phase for every
# stack, because its product bakes in the gate parameters and so must be
# redone per parameter set: qarp and Aer fuse inside the call anyway, qulacs's
# optimizer runs inside `run_state` on a fresh circuit, qsim fuses inside
# `simulate`.  Measured 2026-09-13, macOS arm64 (M-series), one thread:
#
#   aer        fusion_max_qubit 5 / fusion_threshold 14 (its defaults) — every
#              other width within noise or slower (k=4: 119/236/214 ms vs
#              119/213/212 at brickwork/qft/trotter 20q); fusion off is 375/680/312.
#   qulacs     QuantumCircuitOptimizer.optimize_light (single-qubit merging):
#              geomean 48 ms vs 57 off, 52 block_size=2, 79 block_size=1; the
#              optimizer's own cost outweighs wider blocks on qft/trotter
#              (block_size=2: brickwork 270 -> 89 ms but qft 361 -> 481).  The
#              same cost shows on the small energy rows the rule does not
#              weigh (qpe_phase 14q: 2.6 -> 5.8 ms); one setting per stack,
#              chosen where a pass over the state is the cost, is the rule.
#   qsim       max_fused_gate_size 3 (default 2): 44/217/103 ms vs 56/285/135.
#   lightning  none available in lightning.qubit's kernels; PennyLane's tape
#              transform single_qubit_fusion costs more in Python than it
#              saves (brickwork 20q 238 -> 303 ms) — off.
#   qarpx      its own default (fusion_max_qubits = 3 from 12 qubits,
#              simulation_fusion_plan.md), not overridden here.
FUSION: dict[str, dict] = {
    "aer": {"fusion_enable": True, "fusion_max_qubit": 5, "fusion_threshold": 14},
    "qulacs": {"optimizer": "light"},  # "off" | "light" | int block_size
    "qsim": {"max_fused_gate_size": 3},
    "lightning": {"transform": "off"},  # "off" | "1q_fusion"
}

# Expectation-value path, per stack — the same rule as FUSION: every stack
# evaluates ⟨ψ|H|ψ⟩ through the fastest path its engine offers, chosen by
# `python -m benchmarks.statevector.tune --energy` on the energy families at
# n >= 12 (vqe_molecular 12 / 14, vqe_energy 12 / 16 — the widths where the
# contraction, not dispatch, is the cost; geometric mean over all four).  A
# path qualifies only if the contraction runs inside the SDK's simulator —
# lightning's C++ sparse expval, qulacs's Observable / GeneralQuantumOperator,
# Aer's save_expectation_value (or its EstimatorV2, same core), qsim's
# simulate_expectation_values.  Pulling the statevector out and contracting
# it with scipy is not the SDK's number and is not a candidate.  Building the
# observable — lightning's CSR matrix included — is the build phase for every
# stack.  Measured 2026-09-14, macOS arm64 (M4 Pro), one thread, run phase
# in ms at LiH-12 / H2O-14 / Hubbard-12 / Hubbard-16:
#
#   lightning  hamiltonian 20.0 / 70.4 / 3.1 / 15.4  ->  sparse 2.0 / 4.1 /
#              1.7 / 5.6 (SparseHamiltonian; the CSR matrix is built once at
#              build time: 49 / 220 / 5 / 134 ms, 1.6 / 11.6 / 0.6 / 12.9 MiB).
#   qulacs     general 3.4 / 19.6 / 0.7 / 13.5 vs hermitian 3.1 / 19.9 / 0.7 /
#              13.1 — within noise of each other; the sweep's pick stands.
#   aer        save_expectation_value 7.0 / 43.7 / 1.5 / 19.5 vs EstimatorV2
#              69.4 / 187.2 / 5.6 / 24.2 (exact, precision 0).
#   qsim       pauli_sum 31.3 / 136.9 / 2.8 / 23.1 (simulate_expectation_values)
#              — the only in-engine path; cirq's expectation_from_state_vector
#              contracts in numpy and is excluded by the rule.
EXPECTATION: dict[str, dict] = {
    "lightning": {"observable": "sparse"},  # "hamiltonian" | "sparse"
    "qulacs": {"observable": "hermitian"},  # "general" | "hermitian"
    "aer": {"observable": "save_expectation_value"},  # | "estimator"
    "qsim": {"observable": "pauli_sum"},
}
