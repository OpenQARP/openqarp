"""Hypothesis strategies and shared generators for the property-based tests
(`-m property`).

Generators live here rather than beside individual tests so that a single
definition of "a random unitary" (or "a registry block", or "a pipeline
configuration") is exercised everywhere, and so their own distribution can be
reviewed in one place.  The parameterized-block registry (``FACTORIES`` /
``EXCLUDED``) also lives here, shared by the symbols contract
(`tests/test_blocks/test_symbols_contract.py`, which keeps the completeness
guard) and the pipeline property suite (`tests/test_pipeline/`).
"""

import random
from typing import TYPE_CHECKING, NamedTuple, Optional

import networkx as nx
import numpy as np
from hypothesis import strategies as st
from sympy import Symbol

import qarp
import qarpx as qx
from qarp.blocks import (
    BrickworkPCEBlock,
    GivensBlock,
    HEABlock,
    LayerBlock,
    QAOABlock,
    SimpleBlock,
    SPABlock,
    TrotterAnsatzBlock,
    TrotterBlock,
    UCCBlock,
    UPCCDBlock,
)
from qarp.devices import NoiseModel, get_nearest_neighbour_architecture
from qarp.operators import QubitOperator

if TYPE_CHECKING:
    from qarp.engines import Engine, Runnable

# 2**n statevector cost — n > 4 makes the exploratory profile impractical.
qubit_counts = st.integers(min_value=1, max_value=4)

# Bounds span two full turns so 2pi/4pi periodicity is reachable; Hypothesis
# preferentially probes 0.0, -0.0 and the bounds, which is where phase bugs sit.
angles = st.floats(
    min_value=-4 * np.pi,
    max_value=4 * np.pi,
    allow_nan=False,
    allow_infinity=False,
)

# Seeds must round-trip through np.random.default_rng.
unitary_seeds = st.integers(min_value=0, max_value=2**32 - 1)


def random_unitary(n: int, seed: int) -> np.ndarray:
    """Haar-ish unitary from QR of a complex Gaussian, keyed by ``seed``."""
    rng = np.random.default_rng(seed)
    dim = 2**n
    m = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    q, r = np.linalg.qr(m)
    return q * (np.diag(r) / np.abs(np.diag(r)))


@st.composite
def symbol_name_pools(draw, min_size: int = 4) -> list:
    """Distinct sympy symbol names, deliberately UNPADDED so lexicographic
    order can diverge from numeric intuition (``x10`` < ``x2`` < ``x9``).  The
    hand-written adversarial rename uses zero-padded ``r000``-style names,
    where the two orders coincide, so that case is unreachable there."""
    # prefix+number rather than free text: 5*31 names makes unique draws cheap
    # (free text over a short alphabet collided on ~40% of draws), and it
    # produces x9/x10 pairs directly.
    return draw(
        st.lists(
            st.builds(
                lambda prefix, i: f"{prefix}{i}",
                st.sampled_from("abxyz"),
                st.integers(min_value=0, max_value=30),
            ),
            min_size=min_size,
            max_size=8,  # no registry factory exceeds 4 symbols; the rest is waste
            unique=True,
        )
    )


@st.composite
def pauli_term_maps(draw, n_qubits: int) -> dict:
    """``{qubit: 'X'|'Y'|'Z'}``; identity qubits are simply absent, and an
    all-identity draw yields ``{}`` — a boundary case hand-written Pauli tests
    routinely skip.  Drawn per qubit rather than as a unique index list, which
    would make Hypothesis discard colliding draws at small ``n_qubits``."""
    picks = draw(
        st.lists(st.sampled_from("IXYZ"), min_size=n_qubits, max_size=n_qubits),
    )
    return {q: p for q, p in enumerate(picks) if p != "I"}


def render_pauli_term(term_map: dict) -> str:
    """``{0: 'X', 2: 'Z'}`` -> ``"X0 Z2"``; empty map -> ``""`` (identity)."""
    return " ".join(f"{p}{q}" for q, p in sorted(term_map.items()))


@st.composite
def pauli_problems(draw, max_qubits: int = 4) -> tuple:
    """``(n_qubits, term_map)`` drawn together.  Deliberately NOT ``st.data()``
    inside the test: `@example` cannot supply a `data` argument, which would
    make pinned regression cases impossible."""
    n_qubits = draw(st.integers(min_value=1, max_value=max_qubits))
    return n_qubits, draw(pauli_term_maps(n_qubits))


@st.composite
def pauli_sums(draw, max_qubits: int = 8, max_terms: int = 8) -> tuple:
    """``(n_qubits, [(term_map, coeff), …], seed)`` drawn together — a Pauli
    sum with complex coefficients (so it is not Hermitian in general) plus a
    seed for the states it is contracted with.  Repeated term maps are
    allowed on purpose: the sum must add them, not dedupe them.  Up to eight
    qubits so both the sub-block (n < 6) and blocked kernel paths are hit."""
    n_qubits = draw(st.integers(min_value=1, max_value=max_qubits))
    coeffs = st.builds(
        complex,
        st.sampled_from([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]),
        st.sampled_from([-1.0, 0.0, 1.0]),
    )
    terms = draw(
        st.lists(st.tuples(pauli_term_maps(n_qubits), coeffs), min_size=0, max_size=max_terms)
    )
    return n_qubits, terms, draw(unitary_seeds)


@st.composite
def unitaries(draw, n_qubits: int) -> np.ndarray:
    """Draw the *seed*, never the matrix entries: shrinking entries towards
    zero leaves a non-unitary matrix, so every shrink would fail the
    precondition and the reported counterexample stays maximal."""
    return random_unitary(n_qubits, draw(unitary_seeds))


# ── Parameterized-block registry ─────────────────────────────────────────
# Shared inventory: the symbols contract conforms every entry, the pipeline
# sweep composes every entry with primitives/engines/devices.  The
# completeness guard (test_symbols_contract.py::test_registry_is_complete)
# forces every exported parameterized block to appear in exactly one of
# FACTORIES / EXCLUDED, so both suites pick up new blocks automatically.


def _onv4():
    return [1, 1, 0, 0]


def _three_paulis():
    return [QubitOperator("X0 Y1", 1j), QubitOperator("Z0", 1j), QubitOperator("Y0 X1", 1j)]


def _simple_symbolic():
    b = SimpleBlock(2)
    # Adversarial insertion order: reverse-sorted names.
    b.rx(0, qx.Param.symbol("theta_z"))
    b.ry(1, qx.Param.symbol("theta_a"))
    return b


# Symbol names chosen adversarially unsorted (z, y, x) so generation order
# and canonical order differ for every factory that takes explicit symbols.
FACTORIES = {
    "SimpleBlock-fallback": lambda: _simple_symbolic(),
    "UCCBlock": lambda: UCCBlock(occupation_number_vector=_onv4()),
    "UCCBlock-generalised": lambda: UCCBlock(occupation_number_vector=_onv4(), generalised=True),
    "UCCBlock-postfix": lambda: UCCBlock(occupation_number_vector=_onv4(), symbol_postfix="_v0"),
    "UCCBlock-paired": lambda: UCCBlock(occupation_number_vector=_onv4(), paired_doubles=True),
    "UPCCDBlock": lambda: UPCCDBlock([1, 1, 0, 0]),
    "TrotterAnsatzBlock": lambda: TrotterAnsatzBlock(
        n_qubits=2,
        qubit_exponents=_three_paulis(),
        symbols=[Symbol("z"), Symbol("y"), Symbol("x")],
        imaginary=True,
    ),
    # time symbol sorts FIRST ("a_time" < "x") — adversarial against the old
    # append-at-the-end behaviour.
    "TrotterAnsatzBlock-symbolic-time": lambda: TrotterAnsatzBlock(
        n_qubits=2,
        qubit_exponents=_three_paulis(),
        symbols=[Symbol("z"), Symbol("y"), Symbol("x")],
        time=Symbol("a_time"),
        imaginary=True,
    ),
    "TrotterBlock-symbolic-time": lambda: TrotterBlock(
        operator=QubitOperator("Z0"), n_qubits=1, steps=1, time=Symbol("t")
    ),
    "HEABlock": lambda: HEABlock(2, 1, False, True, False, False),
    "BrickworkPCEBlock": lambda: BrickworkPCEBlock(2, 1),
    "SPABlock": lambda: SPABlock(2, 1, False, True, False),
    "GivensBlock": lambda: GivensBlock(theta=Symbol("g0")),
    "LayerBlock": lambda: LayerBlock(qx.GateType.Rx, 2, parameters=[Symbol("z"), Symbol("a")]),
    "QAOABlock": lambda: QAOABlock(2, 1, nx.path_graph(2)),
}

# Symbolic-time blocks defer the C++ buffer until set_time, so a composite
# wrap has no commands to rescan symbols from — wrap agreement is untestable
# for them (a semantic property of deferred build, not of ordering).
UNWRAPPABLE = {"TrotterAnsatzBlock-symbolic-time", "TrotterBlock-symbolic-time"}

# Exported Block classes deliberately not in FACTORIES, with the reason.
# The completeness guard forces every future export to appear in exactly one
# of the two tables.
EXCLUDED = {
    # Structural / parameterless
    "AnyBlock": "hint/isinstance type (= qx.Block), not a constructible block",
    "SimpleBlock": "covered by SimpleBlock-fallback factory",
    "CompositeBlock": "exercised by the wrap test on every factory",
    "CompositeBlockBase": "bare tree primitive behind CompositeBlock",
    "ControlledBlock": "wrapper; symbols come from the wrapped block",
    "MeasureBlock": "parameterless structural primitive",
    "ResetBlock": "parameterless structural primitive",
    "ConditionalBlock": "wrapper; symbols come from the wrapped block",
    "AGateBlock": "fixed-angle two-qubit gate; no free symbols",
    "OrbitalRotationBlock": "Givens angles are classical data from the orthogonal matrix; no free symbols",
    "BrickworkEntanglingBlock": "parameterless entangler",
    "LinearEntanglingBlock": "parameterless entangler",
    "HnBlock": "parameterless",
    "XnBlock": "parameterless",
    "QFTBlock": "parameterless",
    # Exported one level deep (qarp.blocks._primitives, §15), so the
    # completeness guard over qarp.blocks does not reach them; kept for intent.
    "ModularMultiplicationBlock": "parameterless exact permutation synthesizer",
    "OrderFindingBlock": "parameterless modular-order-finding circuit",
    "QROMBlock": "concrete classical lookup data; no free symbols",
    "IdentityBlock": "parameterless",
    "ReadoutBlock": "measurement collection wrapper; no symbols",
    "HadamardTestBlock": "structural; symbols come from wrapped unitary",
    "InterferometricStateBlock": "structural transition-state wrapper; symbols come from bra/ket blocks",
    "InterferometricMeasurementBlock": "measurement wrapper; symbols come from bra/ket blocks",
    "SWAPTestBlock": "structural; no free symbols of its own",
    "RSPBlock": "state prep from concrete amplitudes",
    "PauliBlock": "concrete Pauli string; no symbols",
    "PhaseShiftBlock": "concrete angle input in-tree",
    "ProjectedControlPhaseBlock": "concrete angle input in-tree",
    "ReflectionBlock": "structural",
    "AmplitudeAmplificationBlock": "structural; symbols come from wrapped blocks",
    "AmplitudeEstimationBlock": "structural; symbols come from wrapped blocks",
    "GroverBlock": "structural; symbols come from wrapped oracle",
    "MixedOperatorBlock": "needs operator inputs; angles concrete in-tree",
    "CostOperatorBlock": "exercised inside QAOABlock factory",
    "SelectBlock": "needs operator inputs; no cheap parameterized ctor",
    "BlockEncodingBlock": "needs operator inputs; no cheap parameterized ctor",
    "QubitizationBlock": "needs operator inputs; no cheap parameterized ctor",
    "ParticleNumberProjectorBlock": "fixed quantum-number LCU projector; no free symbols",
    "SzProjectorBlock": "fixed quantum-number LCU projector; no free symbols",
    "SyProjectorBlock": "fixed quantum-number LCU projector; no free symbols",
    "SpinSquaredProjectorBlock": "fixed quantum-number LCU projector; no free symbols",
    "QPEBlock": "needs unitary input; structural",
    "DOSQPEBlock": "needs unitary input; structural",
    "QSPBlock": "concrete phase angles input",
    "QSVTBlock": "concrete phase angles input",
    "SynthesizedTimeEvolutionBlock": "synthesis from concrete matrix",
    "SynthesizedUnitaryBlock": "synthesis from concrete matrix",
    "TrotterBlock": "covered by TrotterBlock-symbolic-time factory",
    # State preparation (concrete amplitude/data inputs)
    "ComputationalBasisStateBlock": "parameterless state prep",
    "DickeStateBlock": "parameterless state prep",
    "CVOQRAMStateBlock": "concrete amplitude data",
    "CVQRAMStateBlock": "concrete amplitude data",
    "MappedONVStateBlock": "parameterless state prep",
    "GHZLikeStateBlock": "parameterless state prep",
    "SynthesizedStateBlock": "synthesis from concrete statevector",
    "SparseStateBlock": "concrete amplitude data",
    "MultiONVStateBlock": "concrete ONV/coefficient data",
    "SlaterDeterminantBlock": "Givens angles are classical data from the orbital matrix; no free symbols",
    "CSFStateBlock": "concrete orbital/coupling-path data; no free symbols",
    "LowRankStateBlock": "concrete amplitude data and SVD-derived unitaries; no free symbols",
    "MPSStateBlock": "concrete MPS tensor data; no free symbols",
    "UniformSuperpositionBlock": "concrete M; no free symbols",
    "PiecewiseLinearStateBlock": "concrete breakpoint/slope/intercept data; no free symbols",
    "HypergraphStateBlock": "parameterless state prep",
    # Re-chained into qarp.blocks by the export-surface plan
    "HaarRandomBlock": "Haar-random unitary drawn from a seed; no free symbols",
    "VUMPOBrickworkBlock": "quimb-gated; angles come bound from a VUMPO instance",
}


def bound_registry_block(name: str, values_seed: int):
    """Registry factory → built → every symbol bound to a seeded value.

    Binding goes through ``parameter_map`` (canonical ``.symbols`` order, §17)
    so the value↔symbol pairing is reproducible from ``values_seed`` alone.
    """
    block = FACTORIES[name]()
    block.build()
    return _bind_all(block, values_seed)


def _bind_all(block, values_seed: int):
    rng = np.random.default_rng(values_seed)
    values = [float(v) for v in rng.uniform(-np.pi, np.pi, size=len(block.symbols))]
    return block.set_symbols(block.parameter_map(values))


def random_gate_block(values_seed: int, n_qubits: int = 3, n_gates: int = 24):
    """Dense concrete random circuit (rotation-rich, so O1/O2 merges actually
    fire — registry ansätze are too structured to exercise the optimizer on
    every draw).  ``generate_random_circuit`` draws from the global ``random``
    module, so it is seeded here, keyed by ``values_seed``."""
    from qarp.utils import generate_random_circuit

    gate_set = [
        qx.GateType.Rz,
        qx.GateType.Rx,
        qx.GateType.Ry,
        qx.GateType.H,
        qx.GateType.X,
        qx.GateType.S,
        qx.GateType.T,
        qx.GateType.CX,
        qx.GateType.CZ,
    ]
    random.seed(values_seed)
    return generate_random_circuit(n_gates, n_qubits, gate_set)


def pipeline_ket(name: str, values_seed: int, opt_level: int):
    """The ket a pipeline config runs: optimize SYMBOLICALLY, then bind.

    Optimize-then-bind is the realistic VQE hot path (ansatz optimized once,
    bound per iteration) and the order that exercises symbolic optimization —
    binding first hides it (the O1 symbol-merge incident class).  Deferred-
    buffer blocks (``UNWRAPPABLE``) have no commands to optimize until bound,
    so they materialise first; ``RandomCircuit`` is concrete throughout.
    """
    if name == "RandomCircuit":
        return random_gate_block(values_seed).optimize(level=opt_level)
    if name in UNWRAPPABLE:
        return bound_registry_block(name, values_seed).optimize(level=opt_level)
    block = FACTORIES[name]()
    block.build()
    return _bind_all(block.optimize(level=opt_level), values_seed)


# ── Pipeline configurations ──────────────────────────────────────────────
# Configs are plain specs
# (strings/ints), never constructed objects — construction happens inside the
# test via build_pipeline, so no qarpx object outlives its example (the
# module-scope leak landmine) and shrunk counterexamples print readably.

PRIMITIVES: tuple = (
    "StateVector",
    "Sampler",
    "PauliAveraging",
    "HadamardTest",
    "TermwiseHadamardTest",
    "SWAPTest",
    "TermwiseSWAPTest",
    "MirrorTest",
)

# The block axis: every registry ansatz plus a dense gate-level random
# circuit (the registry alone is too structured to exercise the optimizer's
# merge passes on every draw).
PIPELINE_BLOCKS: tuple = tuple(sorted(FACTORIES)) + ("RandomCircuit",)

# One member today.  A second engine re-enters here (its tag) and as a
# branch of ``build_pipeline`` — the axis and the dispatch are kept for that.
ENGINES: tuple = ("qarp",)

DEVICE_KINDS: tuple = (None, "line", "ring", "grid")


class PipelineConfig(NamedTuple):
    block_name: str
    primitive: str
    engine: str  # "qarp" (see ENGINES)
    device: Optional[str]  # None | "line" | "ring" | "grid"
    opt_level: int  # 0 | 1 | 2
    shots: Optional[int]  # None → qarp.EXACT
    noise: Optional[tuple]  # None | ("bit_flip" | "depolarizing", p)
    values_seed: int


class BuiltPipeline(NamedTuple):
    primitive: "Runnable"
    engine: "Engine"
    # |result| ceiling for scalar primitives (None for Sampler): Σ|c| for
    # operator expectations, √2 for Hadamard-test estimates (re/im each
    # estimated in [-1, 1]), 1 for overlap probabilities.
    magnitude_bound: Optional[float]


@st.composite
def pipeline_configs(
    draw,
    *,
    primitives: tuple = PRIMITIVES,
    engines: tuple = ENGINES,
    devices: tuple = DEVICE_KINDS,
    opt_levels: tuple = (0, 1, 2),
    shots_axis: tuple = (None, 200),
    noise_axis: tuple = (None, ("bit_flip", 0.05), ("depolarizing", 0.02)),
) -> PipelineConfig:
    """A drawn pipeline configuration.  Deliberately NOT filtered for
    capability-legality — the properties assert invariant-or-CapabilityError,
    which is what pins the capability matrix.  The noise axis applies to
    QarpEngine only: the unified ``NoiseModel`` is a QarpEngine input by type."""
    engine = draw(st.sampled_from(list(engines)))
    noise = draw(st.sampled_from(list(noise_axis))) if engine == "qarp" else None
    return PipelineConfig(
        block_name=draw(st.sampled_from(list(PIPELINE_BLOCKS))),
        primitive=draw(st.sampled_from(list(primitives))),
        engine=engine,
        device=draw(st.sampled_from(list(devices))),
        opt_level=draw(st.sampled_from(list(opt_levels))),
        shots=draw(st.sampled_from(list(shots_axis))),
        noise=noise,
        values_seed=draw(unitary_seeds),
    )


def random_pauli_sum(n_qubits: int, rng: np.random.Generator, max_terms: int = 3):
    """1–``max_terms`` random Pauli terms with real coefficients — Hermitian
    by construction, so expectation values are real and bounded by Σ|c|.
    All-identity draws are kept: the identity term is a legitimate boundary."""
    op = None
    for _ in range(int(rng.integers(1, max_terms + 1))):
        picks = rng.choice(list("IXYZ"), size=n_qubits)
        term = " ".join(f"{p}{q}" for q, p in enumerate(picks) if p != "I")
        t = QubitOperator(term, float(rng.uniform(-2.0, 2.0)))
        op = t if op is None else op + t
    return op


def _aux_state_block(n: int):
    b = SimpleBlock(n, name="aux_bra")
    b.x(0)
    b.build()
    return b


def _aux_unitary_block(n: int):
    b = SimpleBlock(n, name="aux_u")
    b.h(0)
    b.build()
    return b


def _architecture(kind: str, n: int) -> tuple:
    """(qx.Architecture, its qubit count) covering ``n`` logical qubits."""
    if kind == "line":
        return get_nearest_neighbour_architecture(n, 1), n
    if kind == "grid":
        cols = max(1, -(-n // 2))
        return get_nearest_neighbour_architecture(2, cols), 2 * cols
    if kind == "ring":
        # n=2 would duplicate its only edge (the HEA ring-wrap bug shape).
        if n >= 3:
            edges = [(i, (i + 1) % n) for i in range(n)]
        else:
            edges = [(0, 1)] if n == 2 else []
        return qx.Architecture(n, edges, "ring"), n
    raise ValueError(f"unknown device kind {kind!r}")


def build_pipeline(cfg: PipelineConfig) -> BuiltPipeline:
    """Construct (primitive, engine) from a drawn config.

    Raises whatever the constructors raise — call it *inside* the capability
    contract: rejection at construction (e.g. QarpEngine + noise + EXACT) is as valid
    as rejection at build().
    """
    from qarp.algorithms import (
        HadamardTest,
        MirrorTest,
        PauliAveraging,
        Sampler,
        StateVector,
        SWAPTest,
        TermwiseHadamardTest,
        TermwiseSWAPTest,
    )
    from qarp.engines import QarpEngine

    rng = np.random.default_rng(cfg.values_seed)
    ket = pipeline_ket(cfg.block_name, cfg.values_seed, cfg.opt_level)
    n = ket.n_qubits
    op = random_pauli_sum(n, rng)
    sum_abs = sum(abs(c) for c in op.terms.values())
    shots = qarp.EXACT if cfg.shots is None else cfg.shots

    # (constructor, |result| ceiling, compiled circuit width): the Hadamard
    # family adds one ancilla, the SWAP family runs bra+ket+ancilla, the
    # mirror echo stays at ket width — devices must be sized to the circuit,
    # not the ket, or every ancilla primitive degenerates to a capacity
    # rejection and the device axis never exercises routing.
    builders = {
        "StateVector": lambda: (StateVector(ket=ket, operator=op), sum_abs, n),
        "Sampler": lambda: (Sampler(ket=ket, n_shots=shots), None, n),
        "PauliAveraging": lambda: (PauliAveraging(ket=ket, operator=op, n_shots=shots), sum_abs, n),
        "HadamardTest": lambda: (
            HadamardTest(ket=ket, operator=_aux_unitary_block(n), n_shots=shots),
            np.sqrt(2.0),
            n + 1,
        ),
        "TermwiseHadamardTest": lambda: (
            TermwiseHadamardTest(ket=ket, operator=op, n_shots=shots),
            sum_abs * np.sqrt(2.0),
            n + 1,
        ),
        "SWAPTest": lambda: (
            SWAPTest(bra=_aux_state_block(n), ket=ket, n_shots=shots),
            1.0,
            2 * n + 1,
        ),
        "TermwiseSWAPTest": lambda: (
            TermwiseSWAPTest(bra=[_aux_state_block(n)], ket=ket, coefficients=[1.0], n_shots=shots),
            1.0,
            2 * n + 1,
        ),
        "MirrorTest": lambda: (
            MirrorTest(
                bra=_aux_state_block(n), ket=ket, operator=_aux_unitary_block(n), n_shots=shots
            ),
            1.0,
            n,
        ),
    }
    prim, bound, width = builders[cfg.primitive]()

    arch = None
    arch_n = width
    if cfg.device is not None:
        arch, arch_n = _architecture(cfg.device, width)

    eng: "Engine"
    if cfg.engine == "qarp":
        nm = None if cfg.noise is None else getattr(NoiseModel, cfg.noise[0])(cfg.noise[1])
        if arch is None and nm is None:
            # No device fields at all: an effective Device would impose a
            # capacity cap the config never asked for.
            eng = QarpEngine(n_shots=shots, seed=11)
        else:
            # Routing needs a 1q/2q rebase target (the router rejects 3q
            # gates); a connectivity-only Device is a config the taxonomy
            # findings track separately.
            gate_set = qx.full_gateset_1q_2q() if arch is not None else None
            eng = QarpEngine(
                n_qubits=arch_n,
                architecture=arch,
                noise_model=nm,
                gate_set=gate_set,
                n_shots=shots,
                seed=11,
            )
    else:
        # Engine re-entry point: a new ENGINES tag gets its constructor here.
        raise ValueError(f"unknown engine {cfg.engine!r}")
    return BuiltPipeline(primitive=prim, engine=eng, magnitude_bound=bound)


def run_pipeline(cfg: PipelineConfig):
    """Construct, build and run a config; return its single result.

    No exception handling: metamorphic properties that compare two runs use
    this directly (a rejection there is a failure), and callers tolerating
    rejection catch ``CapabilityError`` themselves.
    """
    built = build_pipeline(cfg)
    built.engine.build([built.primitive])
    return built.engine.run()[0]


def results_allclose(a, b, atol: float = 1e-8) -> bool:
    """Engine results equal within ``atol``: scalars compared as complex,
    sampling dictionaries on the union of keys (absent key = probability 0)."""
    if isinstance(a, dict) or isinstance(b, dict):
        if not (isinstance(a, dict) and isinstance(b, dict)):
            return False
        keys = set(a) | set(b)
        return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= atol for k in keys)
    return abs(complex(a) - complex(b)) <= atol
