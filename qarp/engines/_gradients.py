"""Gradient method registry and the shared evaluation core (private module).

``Engine.run_gradient(params, method=...)`` delegates here.  The registry is
a set of plain strings (PennyLane's names, except that the policy entry is
``"default"`` rather than ``"best"`` — it is a per-engine choice that may
change, not a method).  Every non-adjoint method is a *stencil*: a set of
parameter points evaluated in one ``Engine._sweep`` call and combined with
fixed coefficients, so parameter shift, finite differences and SPSA share
one batched evaluation path.

Layering: this module sees primitives only through the ``Runnable`` protocol
(``target.name`` matched by string, ``gradient_kind`` read as a plain
attribute) — never ``qarp.algorithms``.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, NamedTuple, Optional, Sequence

import numpy as np

import qarpx as qx
from qarp.operators import QubitOperator

from ..errors import CapabilityError
from ._engine import _coerce_params, _operator_n_qubits, _qubit_operator_to_observable

GRADIENT_METHODS: tuple[str, ...] = (
    "default",
    "adjoint",
    "parameter-shift",
    "finite-diff",
    "spsa",
)
# Names claimed for later releases; recognised, refused.
RESERVED_METHODS: tuple[str, ...] = ("hadamard", "metric-tensor")
# ``Runnable.gradient_kind`` values — what the shift rules may assume about
# ``prim.run()`` as a function of each compiled circuit's state, separately.
GRADIENT_KINDS: tuple[str, ...] = ("expectation", "amplitude", "squared_overlap", "none")

_OPTIONS: dict[str, frozenset[str]] = {
    "finite-diff": frozenset({"fd_eps", "fd_order"}),
    "spsa": frozenset({"spsa_c0", "num_spsa", "spsa_seed"}),
}

# (offset-from-base-point per symbol, real coefficient)
Stencil = list[tuple[dict[str, float], float]]


def gradient_method_from_flag(gradient: Any) -> Optional[str]:
    """``VQA(gradient=...)`` coercion: ``True`` → ``"default"``, ``False`` →
    ``None``, a registry string → itself.  Typos fail here, at construction."""
    if gradient is True:
        return "default"
    if gradient is False or gradient is None:
        return None
    if isinstance(gradient, str):
        _check_method_name(gradient)
        return gradient
    raise TypeError(f"gradient must be a bool or one of {GRADIENT_METHODS}, got {gradient!r}")


def _check_method_name(method: str) -> None:
    if method in RESERVED_METHODS:
        raise CapabilityError(
            f"gradient method {method!r} is reserved for a later release and not available."
        )
    if method not in GRADIENT_METHODS:
        raise ValueError(f"unknown gradient method {method!r}; choose one of {GRADIENT_METHODS}.")


def adjoint_eligible(prim) -> bool:
    """The C++ adjoint differentiates ⟨ψ|H|ψ⟩ over a QubitOperator and |⟨bra|ket⟩|²."""
    if not getattr(prim, "supports_backprop_gradient", False):
        return False
    target = getattr(getattr(prim, "target", None), "name", None)
    if target == "EXPECTATION_VALUE":
        return isinstance(prim.operator, QubitOperator)
    return target == "OVERLAP"


def resolve_method(engine, prim, method: str) -> str:
    """Registry lookup: policy → concrete method → engine/primitive capability."""
    _check_method_name(method)
    if method == "default":
        method = engine._default_gradient_method(prim)
    if method not in engine.gradient_methods:
        hint = "  QarpEngine provides 'adjoint'." if method == "adjoint" else ""
        raise CapabilityError(
            f"{type(engine).__name__} has no {method!r} gradient; it declares "
            f"{sorted(engine.gradient_methods)}.{hint}"
        )
    if method == "adjoint" and not adjoint_eligible(prim):
        raise CapabilityError(
            f"{type(prim).__name__} (target {getattr(prim.target, 'name', prim.target)}) "
            "is not adjoint-differentiable: the adjoint covers StateVector expectation "
            "values over a QubitOperator and StateVector overlaps.  Use "
            "method='parameter-shift'."
        )
    return method


def _validate_options(method: str, options: Mapping) -> None:
    allowed = _OPTIONS.get(method, frozenset())
    bad = sorted(set(options) - allowed)
    if bad:
        raise ValueError(
            f"options {bad} are not valid for gradient method {method!r}"
            + (f"; it accepts {sorted(allowed)}." if allowed else "; it takes no options.")
        )


def _complex_output(prim) -> bool:
    """dtype by *target*, never by the numeric type of one evaluation."""
    target = getattr(getattr(prim, "target", None), "name", None)
    if target == "TRANSITION_AMPLITUDE":
        return True
    if target == "EXPECTATION_VALUE" and isinstance(getattr(prim, "operator", None), qx.Block):
        return True
    return getattr(prim, "expectation_type", "real") == "complex"


def _value_transform(prim) -> Callable[[complex], complex]:
    """Per-target objective: a StateVector OVERLAP is differentiated as
    |⟨bra|ket⟩|² on every method (the VQD contract) while ``run()`` returns
    the amplitude; every other primitive is differentiated as ``run()`` returns it."""
    if getattr(prim, "gradient_kind", "none") == "squared_overlap":
        return lambda v: abs(v) ** 2
    return lambda v: v


# ── Template entry point ─────────────────────────────────────────────────


def compute_gradients(
    engine, params: Mapping, method: str, options: Optional[Mapping]
) -> list[np.ndarray]:
    """Body of ``Engine.run_gradient`` — see its docstring for the contract."""
    _check_method_name(method)
    opts = dict(options or {})
    _validate_options(method, opts)
    params = _coerce_params(params)
    symbol_names = list(params.keys())

    engine._pre_run()
    prims = list(engine._primitives)
    for prim, l2p_list in zip(prims, engine._l2p_per_primitive, strict=True):
        engine._validate_primitive(prim)
        engine._pre_dispatch_check(prim, l2p_list)
    resolved = [engine.resolve_gradient_method(prim, method) for prim in prims]

    grads: list[np.ndarray] = [
        np.zeros(len(symbol_names), dtype=complex if _complex_output(p) else float) for p in prims
    ]

    adjoint_idx = [i for i, m in enumerate(resolved) if m == "adjoint"]
    if adjoint_idx:
        out = engine._adjoint_gradients([prims[i] for i in adjoint_idx], params, symbol_names)
        for i, g in zip(adjoint_idx, out, strict=True):
            grads[i] = np.asarray(g, dtype=float)

    rest = [i for i, m in enumerate(resolved) if m != "adjoint"]
    if rest:
        methods = {resolved[i] for i in rest}
        assert len(methods) == 1, methods  # one requested method; only "default" branches
        rest_method = methods.pop()
        rest_prims = [prims[i] for i in rest]
        rest_l2p = [engine._l2p_per_primitive[i] for i in rest]
        if rest_method == "parameter-shift":
            out = parameter_shift_gradients(
                engine, rest_prims, rest, rest_l2p, params, symbol_names
            )
        else:
            stencil_fns: dict[str, Callable[..., list[list[Stencil]]]] = {
                "finite-diff": finite_difference_stencils,
                "spsa": spsa_stencils,
            }
            stencils = stencil_fns[rest_method](engine, rest_prims, params, symbol_names, **opts)
            out = evaluate_stencils(
                engine,
                rest_prims,
                [p.compiled_circuits for p in rest_prims],
                rest_l2p,
                params,
                stencils,
            )
        for i, g in zip(rest, out, strict=True):
            grads[i] = g
    return grads


# ── Stencil core ─────────────────────────────────────────────────────────


def evaluate_stencils(
    engine,
    prims: Sequence,
    circuits_per_prim: Sequence[list],
    l2p_per_prim: Sequence,
    base: Mapping[str, float],
    stencils_per_prim: Sequence[Sequence[Stencil]],
) -> list[np.ndarray]:
    """Evaluate every distinct point of every stencil in ONE ``_sweep`` and
    combine.  ``stencils_per_prim[j][k]`` is the stencil of primitive ``j``
    for symbol ``k``; an empty stencil is a zero derivative."""
    points: list[dict[str, float]] = []
    index: dict[tuple, int] = {}

    def point_of(offsets: Mapping[str, float]) -> int:
        pt = dict(base)
        for s, v in offsets.items():
            pt[s] = pt.get(s, 0.0) + v
        key = tuple(sorted(pt.items()))
        if key not in index:
            index[key] = len(points)
            points.append(pt)
        return index[key]

    plan: list[list[list[tuple[int, float]]]] = [
        [[(point_of(off), coeff) for off, coeff in stencil] for stencil in prim_stencils]
        for prim_stencils in stencils_per_prim
    ]
    values = engine._sweep(prims, circuits_per_prim, l2p_per_prim, points, None) if points else []

    out: list[np.ndarray] = []
    for j, prim in enumerate(prims):
        transform = _value_transform(prim)
        g = np.zeros(len(plan[j]), dtype=complex)
        for k, terms in enumerate(plan[j]):
            for point_idx, coeff in terms:
                v = values[point_idx][j]
                if not isinstance(v, (int, float, complex, np.number)):
                    raise CapabilityError(
                        f"{type(prim).__name__} returns a {type(v).__name__}, not a scalar; "
                        "only scalar-valued primitives are differentiable."
                    )
                g[k] += coeff * complex(transform(complex(v)))
        out.append(g if _complex_output(prim) else g.real)
    return out


def _require_scalar_kind(prim, method: str) -> None:
    kind = getattr(prim, "gradient_kind", "none")
    if kind not in GRADIENT_KINDS:
        raise ValueError(f"{type(prim).__name__}.gradient_kind={kind!r} not in {GRADIENT_KINDS}")
    if kind == "none":
        raise CapabilityError(
            f"{type(prim).__name__} declares gradient_kind='none': its value is not a "
            "trigonometric polynomial of each circuit's gate angles, so the "
            f"{method} rule does not apply.  Use method='finite-diff'."
        )


# ── parameter shift: per-occurrence, generator-spectrum rules ─────────────

# Rules in *angle* units as (shift, coefficient) lists, per gate and per
# gradient_kind.  EXPECTATION: the value is bilinear in the circuit's state,
# so the frequencies are the generator's eigenvalue *differences*;
# AMPLITUDE: linear in one circuit's amplitudes, frequencies are the
# eigenvalues themselves.  General rule: Wierichs, Izaac, Wang, Lin,
# Quantum 6, 677 (2022); R=1 is the two-term rule, R=2 with ω₀=½ the
# four-term rule of Anselmetti et al. (2021).  Symbol-unit scaling by the
# affine coefficient c happens in ``_occurrence_stencil``.
_TWO_TERM_HALF = [(np.pi / 2, 0.5), (-np.pi / 2, -0.5)]  # frequency 1
_TWO_TERM_PI = [(np.pi, 0.25), (-np.pi, -0.25)]  # frequency ½
_C_PLUS = (np.sqrt(2.0) + 1.0) / (4.0 * np.sqrt(2.0))
_C_MINUS = (np.sqrt(2.0) - 1.0) / (4.0 * np.sqrt(2.0))
_FOUR_TERM = [  # frequencies {½, 1}
    (np.pi / 2, _C_PLUS),
    (-np.pi / 2, -_C_PLUS),
    (3 * np.pi / 2, -_C_MINUS),
    (-3 * np.pi / 2, _C_MINUS),
]
_RULES: Optional[dict[tuple[str, str], list[tuple[float, float]]]] = None


def _rules() -> dict[tuple[str, str], list[tuple[float, float]]]:
    """Built lazily: no module-scope qarpx objects (nanobind leak rule)."""
    global _RULES
    if _RULES is None:
        pauli = ("Rx", "Ry", "Rz", "RXX", "RYY", "RZZ")  # spectrum {±½}
        projector = ("P", "CP")  # spectrum {0, −1}
        controlled = ("CRx", "CRy", "CRz")  # spectrum {0, ±½}
        rules: dict[tuple[str, str], list[tuple[float, float]]] = {}
        for g in pauli:
            rules[(g, "expectation")] = _TWO_TERM_HALF
            rules[(g, "amplitude")] = _TWO_TERM_PI
        for g in projector:
            rules[(g, "expectation")] = _TWO_TERM_HALF
            rules[(g, "amplitude")] = _TWO_TERM_HALF
        for g in controlled:
            rules[(g, "expectation")] = _FOUR_TERM
            rules[(g, "amplitude")] = _TWO_TERM_PI
        rules[("GPhase", "expectation")] = []  # a global phase drops out of ⟨ψ|H|ψ⟩
        rules[("GPhase", "amplitude")] = _TWO_TERM_HALF
        _RULES = rules
    return _RULES


def shift_rule(gate, kind: str) -> list[tuple[float, float]]:
    """``[(shift_in_angle_units, coefficient), ...]`` for one occurrence."""
    name = gate.name if hasattr(gate, "name") else str(gate)
    key_kind = "expectation" if kind == "squared_overlap" else kind
    try:
        return _rules()[(name, key_kind)]
    except KeyError:
        raise CapabilityError(
            f"gate '{name}' with a symbolic angle has no parameter-shift rule "
            "(symbolic U/CU are rewritten before differentiation; anything else is "
            "unsupported).  Use method='finite-diff'."
        ) from None


class Occurrence(NamedTuple):
    circuit: int
    command: int
    param: int
    symbol: str
    coeff: float  # ∂angle/∂symbol
    private: str  # the occurrence's own symbol name in the rewritten circuit


class ShiftPlan:
    """A primitive's circuits rewritten so that every appearance of every
    symbol carries a private name, plus the affine coefficient of each.
    Built once per ``(primitive, build generation, symbol set)``."""

    def __init__(self, prim, symbol_names: Sequence[str]):
        self.kind = getattr(prim, "gradient_kind", "none")
        self.occurrences: dict[str, list[Occurrence]] = {s: [] for s in symbol_names}
        # Renamed occurrences the symbol does not move (zero affine coefficient):
        # bound at the base point, never shifted.
        self._inert: list[tuple[str, str]] = []
        self.circuits: list[list] = []
        for ci, cmds in enumerate(prim.compiled_circuits):
            current = rebase_differentiable(cmds)
            for sym in symbol_names:
                current, renamed = qx.rename_symbol_occurrences(current, sym, f"{sym}\0{ci}\0")
                for cmd_idx, param_idx, name in renamed:
                    coeffs = qx.affine_coefficients(current[cmd_idx].params[param_idx])
                    if coeffs[name] == 0.0:
                        self._inert.append((sym, name))  # nothing to shift, no division by 0
                        continue
                    self.occurrences[sym].append(
                        Occurrence(ci, cmd_idx, param_idx, sym, coeffs[name], name)
                    )
            self.circuits.append(current)

    def base_point(self, params: Mapping[str, float]) -> dict[str, float]:
        base = dict(params)
        for sym, occs in self.occurrences.items():
            for occ in occs:
                base[occ.private] = params[sym]
        for sym, private in self._inert:
            base[private] = params[sym]
        return base

    def stencil(self, sym: str) -> Stencil:
        out: Stencil = []
        for occ in self.occurrences[sym]:
            gate = self.circuits[occ.circuit][occ.command].gate
            for shift, coeff in shift_rule(gate, self.kind):
                # angle = c·x + b: shift x by s/c, and ∂/∂x = c·∂/∂angle.
                out.append(({occ.private: shift / occ.coeff}, coeff * occ.coeff))
        return out


def differentiable_gateset():
    """``native_gateset`` minus ``U``/``CU``: a symbolic ``U`` or ``CU`` has
    no single generator, so the transpiler's phase-exact decompositions
    (``GPhase((φ+λ)/2)·Rz·Ry·Rz``, and the ``P``/``CX`` ladder for ``CU``)
    rewrite it into gates that do.  Built by *assignment* — in-place mutation
    of ``gs.allowed`` edits a Python copy of the C++ set.  ``native_gateset``
    admits ``Custom``, so O1-fused concrete blobs are not re-opened."""
    gs = qx.native_gateset()
    gs.allowed = gs.allowed - {qx.GateType.U, qx.GateType.CU}
    return gs


def rebase_differentiable(cmds: list) -> list:
    """Rewrite symbolic ``U``/``CU`` into the native ladder; every other
    stream passes through untouched.  Plain ``transpile`` (no optimisation
    pass), so no rotation merge can re-run here."""
    if not any(cmd.gate.name in ("U", "CU") and cmd.is_parametric() for cmd in cmds):
        return cmds
    return qx.Transpiler(differentiable_gateset()).transpile(cmds)


def _shift_plan(engine, prim, prim_idx: int, symbol_names: Sequence[str]) -> ShiftPlan:
    """Cached on the engine, keyed by build generation — never by object ids,
    which Python reuses after garbage collection."""
    generation = getattr(engine, "_build_generation", 0)
    cache = getattr(engine, "_shift_plans", None)
    if cache is None or cache.get("generation") != generation:
        cache = {"generation": generation, "plans": {}}
        engine._shift_plans = cache
    key = (prim_idx, tuple(symbol_names))
    plan = cache["plans"].get(key)
    if plan is None:
        plan = ShiftPlan(prim, symbol_names)
        cache["plans"][key] = plan
    return plan


def parameter_shift_gradients(
    engine, prims, prim_indices, l2p_per_prim, params, symbol_names
) -> list[np.ndarray]:
    """Per-occurrence batched shift: every occurrence of every symbol is
    shifted on its own (2 or 4 points), all points go through one ``_sweep``
    on the rewritten circuits, and the contributions sum per symbol."""
    plans = []
    for prim, idx in zip(prims, prim_indices, strict=True):
        _require_scalar_kind(prim, "parameter-shift")
        plans.append(_shift_plan(engine, prim, idx, symbol_names))
    base = dict(params)
    for plan in plans:
        base.update(plan.base_point(params))
    stencils = [[plan.stencil(sym) for sym in symbol_names] for plan in plans]
    return evaluate_stencils(
        engine, prims, [plan.circuits for plan in plans], l2p_per_prim, base, stencils
    )


# ── finite differences / SPSA (the optimizer helpers' formulas, batched) ──


def finite_difference_stencils(
    engine, prims, params, symbol_names, *, fd_eps: float = 1e-5, fd_order: int = 2
) -> list[list[Stencil]]:
    """``fd_order=1``: forward, ``(f(x+εe_k) − f(x))/ε`` — the formula of
    ``qarp.optimizers.compute_fd_gradients``.  ``fd_order=2``: central."""
    if fd_order not in (1, 2):
        raise ValueError(f"fd_order must be 1 (forward) or 2 (central), got {fd_order!r}")
    eps = float(fd_eps)
    if eps <= 0.0:
        raise ValueError(f"fd_eps must be positive, got {fd_eps!r}")
    if fd_order == 1:
        per_symbol: list[Stencil] = [
            [({}, -1.0 / eps), ({sym: eps}, 1.0 / eps)] for sym in symbol_names
        ]
    else:
        per_symbol = [[({sym: eps}, 0.5 / eps), ({sym: -eps}, -0.5 / eps)] for sym in symbol_names]
    return [list(per_symbol) for _ in prims]


def spsa_stencils(
    engine,
    prims,
    params,
    symbol_names,
    *,
    spsa_c0: float = 1e-2,
    num_spsa: int = 1,
    spsa_seed: Optional[int] = None,
) -> list[list[Stencil]]:
    """Step-0 SPSA of ``qarp.optimizers.compute_spsa_gradients`` (no ``c_k``
    decay): Rademacher directions ``δ``, ``(f(x+cδ) − f(x−cδ))/(2c)·δ`` averaged
    over ``num_spsa`` draws.  ``spsa_seed=None`` draws from the engine's own
    stream (``Engine(seed=...)``, a fixed stream when unseeded), so every MPI
    rank running the same closure perturbs identically."""
    c = float(spsa_c0)
    if c <= 0.0:
        raise ValueError(f"spsa_c0 must be positive, got {spsa_c0!r}")
    num = int(num_spsa)
    if num < 1:
        raise ValueError(f"num_spsa must be ≥ 1, got {num_spsa!r}")
    if spsa_seed is not None:
        rng = np.random.default_rng(spsa_seed)
    else:
        rng = engine._gradient_rng()
    d = len(symbol_names)
    per_symbol: list[Stencil] = [[] for _ in symbol_names]
    for _ in range(num):
        delta = rng.choice([-1.0, 1.0], size=d)
        plus = {s: c * float(dl) for s, dl in zip(symbol_names, delta, strict=True)}
        minus = {s: -v for s, v in plus.items()}
        for k in range(d):
            w = float(delta[k]) / (2.0 * c * num)
            per_symbol[k].append((plus, w))
            per_symbol[k].append((minus, -w))
    return [[list(st) for st in per_symbol] for _ in prims]


# ── adjoint (C++ backprop on a CPU QarpSimulator) ─────────────────────────


def _carries_tracked_symbol(cmds, symbol_names: Sequence[str]) -> bool:
    tracked = set(symbol_names)
    return any(
        p.is_symbolic() and tracked.intersection(p.free_symbols())
        for cmd in cmds
        for p in cmd.params
    )


def adjoint_gradients(sim, prims, params: Mapping[str, float], symbol_names: list[str]) -> list:
    """One array per primitive: ``∂⟨ψ|H|ψ⟩/∂θ`` (EXPECTATION_VALUE over a
    QubitOperator) via ``run_gradient``, or ``∂|⟨bra|ket⟩|²/∂θ`` (OVERLAP) via
    ``run_gradient_phi``: ``2·Re(⟨bra|ket⟩*·⟨bra|∂ket⟩)`` from a sweep over
    the ket with ``|φ⟩ = ⟨bra|ket⟩·|bra⟩``, plus — only when a tracked symbol
    is free in the bra — ``2·Re(⟨ket|bra⟩*·⟨ket|∂bra⟩)`` from a second sweep
    over the bra with ``|φ'⟩ = ⟨ket|bra⟩·|ket⟩``.  Cost is ~2 sweeps per
    differentiated circuit regardless of parameter count; shared symbols sum
    across gates and across the two circuits."""
    out: list[np.ndarray] = []
    for prim in prims:
        grad = np.zeros(len(symbol_names))
        psi0 = prim.initial_state
        if psi0 is not None:
            psi0 = np.ascontiguousarray(psi0, dtype=np.complex128)
        circuits = [rebase_differentiable(c) for c in prim.compiled_circuits]
        if prim.target.name == "EXPECTATION_VALUE":
            if circuits:
                cmds = circuits[0]
                # Pad to the operator width — an observable may act on more
                # qubits than the circuit (idle qubits stay |0>).
                n_q = max(prim._n_qubits_list[0], _operator_n_qubits(prim.operator))
                obs = _qubit_operator_to_observable(prim.operator, n_q)
                grad = np.array(
                    sim.run_gradient(cmds, n_q, obs, params, symbol_names, initial_state=psi0)
                )
        elif len(circuits) >= 2:  # OVERLAP
            ket_cmds, bra_cmds = circuits[0], circuits[1]
            n_q = prim._n_qubits_list[0]
            ket_sub = qx.substitute_all(ket_cmds, params) if params else ket_cmds
            bra_sub = qx.substitute_all(bra_cmds, params) if params else bra_cmds
            # initial_state seeds the ket side only — bra stays |0…0⟩-rooted,
            # matching StateVector's forward semantics.
            ket_sv = np.asarray(sim.statevector(ket_sub, n_q, initial_state=psi0))
            bra_sv = np.asarray(sim.statevector(bra_sub, n_q))
            overlap = complex(np.vdot(bra_sv, ket_sv))
            phi = (overlap * bra_sv).astype(np.complex128)
            grad = np.array(
                sim.run_gradient_phi(
                    ket_cmds, n_q, phi.tolist(), params, symbol_names, initial_state=psi0
                )
            )
            if _carries_tracked_symbol(bra_cmds, symbol_names):
                # The bra term of ∂|⟨bra|ket⟩|²; dropping it was a silent
                # wrong number whenever a symbol drives both circuits.
                phi_bra = (np.conj(overlap) * ket_sv).astype(np.complex128)
                grad = grad + np.array(
                    sim.run_gradient_phi(bra_cmds, n_q, phi_bra.tolist(), params, symbol_names)
                )
        out.append(grad)
    return out
