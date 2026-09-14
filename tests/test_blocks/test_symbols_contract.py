"""Symbols-ordering contract — registry-driven conformance suite.

Pins the symbols-ordering invariant: on any built block,
``.symbols`` is a **tuple** in canonical order (``_sorted_symbols`` — sorted by
string), identical on every surface that republishes the same symbol set, and
preserved by every lifecycle step (rename, postfix, partial bind, deepcopy,
composite wrap).  Symbol↔operator pairing is pinned separately — deliberately
*unsorted* — on the dedicated pairing surfaces.

Completeness guard: every exported parameterized block class must either have
a ``FACTORIES`` entry (in `tests/strategies.py`, shared with the pipeline
property suite) or an ``EXCLUDED`` entry with a reason.  A new block that
builds with symbols and is registered in neither fails the suite — that
clause is what keeps this contract from rotting.
"""

import json
import os
import subprocess
import sys
from copy import deepcopy

import numpy as np
import pytest
from hypothesis import example, given
from hypothesis import strategies as st
from sympy import Symbol

import qarpx as qx
from qarp.blocks import CompositeBlock, SimpleBlock, UCCBlock
from qarp.operators.ucc import ucc_doubles, ucc_singles
from tests.strategies import EXCLUDED, FACTORIES, UNWRAPPABLE, symbol_name_pools


def _onv4():
    return [1, 1, 0, 0]


def _assert_canonical(block, context=""):
    syms = block.symbols
    assert isinstance(syms, tuple), f"{context}: symbols is {type(syms).__name__}, not tuple"
    assert list(syms) == sorted(syms, key=str), (
        f"{context}: symbols not canonically ordered: {syms}"
    )


# ── Completeness guard ──────────────────────────────────────────────────


def test_registry_is_complete():
    """Every exported Block class is either factory-tested or excluded with a
    reason.  New parameterized blocks MUST register here — add a FACTORIES
    entry (preferred) or an EXCLUDED entry explaining why not."""
    import qarp.blocks as qb

    covered = {type(factory()) for factory in FACTORIES.values()}
    unaccounted = []
    for name in dir(qb):
        obj = getattr(qb, name)
        if not (isinstance(obj, type) and issubclass(obj, qx.Block)):
            continue
        if obj in covered or name in EXCLUDED:
            continue
        unaccounted.append(name)
    assert not unaccounted, (
        f"Blocks unregistered in the symbols contract: {unaccounted}. "
        "Add a FACTORIES entry (or EXCLUDED with a reason) in "
        "test_symbols_contract.py."
    )


# ── Lifecycle conformance (parametrized over the registry) ───────────────


@pytest.mark.parametrize("name", FACTORIES)
def test_symbols_canonical_after_build(name):
    block = FACTORIES[name]()
    block.build()
    _assert_canonical(block, f"{name} after build")
    assert len(block.symbols) > 0, f"{name}: factory must produce a parameterized block"


@pytest.mark.parametrize("name", FACTORIES)
def test_symbols_canonical_after_adversarial_rename(name):
    """Rename chosen to invert the sorted order — a naive in-place rename
    would leave the list unsorted."""
    block = FACTORIES[name]()
    block.build()
    old = block.symbols
    n = len(old)
    rename = {s: Symbol(f"r{n - 1 - i:03d}") for i, s in enumerate(old)}
    renamed = block.replace_symbols(rename)
    _assert_canonical(renamed, f"{name} after adversarial rename")
    assert set(renamed.symbols) == set(rename.values())


@pytest.mark.property
@example(name="UCCBlock", pool=["x10", "x2", "x9", "x1"])  # str-order != numeric order
@given(name=st.sampled_from(sorted(FACTORIES)), pool=symbol_name_pools())
def test_symbols_canonical_after_arbitrary_rename(name, pool):
    """Generalization of the adversarial rename above, which it does not
    replace: that one pins a single order-inverting map with zero-padded
    names, this one searches arbitrary target names across every factory."""
    block = FACTORIES[name]()
    block.build()
    old = block.symbols

    names = list(pool)
    i = 0
    while len(names) < len(old):  # pool is unique but may be shorter than needed
        candidate = f"_pad{i}"
        if candidate not in names:
            names.append(candidate)
        i += 1

    rename = {s: Symbol(n) for s, n in zip(old, names[: len(old)], strict=True)}
    renamed = block.replace_symbols(rename)
    _assert_canonical(renamed, f"{name} after arbitrary rename")
    assert set(renamed.symbols) == set(rename.values())


@pytest.mark.parametrize("name", FACTORIES)
def test_symbols_canonical_after_postfix(name):
    block = FACTORIES[name]()
    block.build()
    refreshed = block.refresh_symbols("_p")
    _assert_canonical(refreshed, f"{name} after refresh_symbols")
    assert all(str(s).endswith("_p") for s in refreshed.symbols)


@pytest.mark.parametrize("name", FACTORIES)
def test_symbols_canonical_after_partial_bind(name):
    block = FACTORIES[name]()
    block.build()
    bound_sym = block.symbols[0]
    partial = block.set_symbols({bound_sym: 0.5})
    _assert_canonical(partial, f"{name} after partial set_symbols")
    assert bound_sym not in partial.symbols


@pytest.mark.parametrize("name", FACTORIES)
def test_symbols_survive_deepcopy(name):
    block = FACTORIES[name]()
    block.build()
    copy = deepcopy(block)
    assert copy.symbols == block.symbols
    _assert_canonical(copy, f"{name} after deepcopy")


@pytest.mark.parametrize("name", sorted(set(FACTORIES) - UNWRAPPABLE))
def test_composite_wrap_agrees_with_block(name):
    """The original failure mode: a rebuilt composite ket republishes the SAME
    order as the wrapped ansatz — `dict(zip(...))` against either surface
    binds identically."""
    block = FACTORIES[name]()
    block.build()
    wrap = CompositeBlock([block], block.n_qubits)
    wrap.build()
    assert list(wrap.symbols) == list(block.symbols), (
        f"{name}: composite wrap changed symbol order — the silent-permutation trap"
    )


def test_parameter_map_is_the_blessed_conversion():
    """`Block.parameter_map` is the one vector→map conversion: aligned to
    the canonical `symbols` order, length-checked, and refusing symbol-less
    blocks — so hand-zips (the silent-permutation trap) are never needed."""
    block = FACTORIES["UCCBlock"]()
    block.build()
    x = [0.1 * (i + 1) for i in range(len(block.symbols))]
    pm = block.parameter_map(x)
    assert list(pm) == list(block.symbols)
    assert list(pm.values()) == x
    with pytest.raises(ValueError):
        block.parameter_map(x[:-1])
    with pytest.raises(RuntimeError):
        SimpleBlock(1).parameter_map([])


def test_symbols_mutation_is_sealed():
    block = FACTORIES["UCCBlock"]()
    block.build()
    with pytest.raises(AttributeError):
        block.symbols.append(Symbol("x"))
    with pytest.raises(TypeError):
        block.symbols[0] = Symbol("x")


# ── Reconstruction paths: commands set directly, then mark_built ─────────
# `optimize()` and the container-materialisation helper populate a block from
# a command stream instead of running the build lifecycle.  The invariant is
# stated over *any* built block, so these owe the registry too — skipping it
# leaves `.symbols is None`, which makes `refresh_symbols` / `parameter_map`
# raise "build the block first" on a block that is already built.


@pytest.mark.parametrize("name", sorted(set(FACTORIES) - UNWRAPPABLE))
def test_symbols_canonical_after_optimize(name):
    """Deferred-buffer factories are excluded for the same reason as the wrap
    test: with no commands to rescan, there is nothing to republish."""
    block = FACTORIES[name]()
    block.build()
    optimized = block.optimize(level=0)
    _assert_canonical(optimized, f"{name} after optimize")
    assert list(optimized.symbols) == list(block.symbols), (
        f"{name}: optimize() changed the published symbol set"
    )


def test_optimized_block_still_renames():
    """`.symbols is None` made this raise "build the block first" on a block
    that `optimize()` had just marked built."""
    block = FACTORIES["UCCBlock"]()
    block.build()
    renamed = block.optimize(level=0).refresh_symbols("_opt")
    assert [str(s) for s in renamed.symbols] == [f"{s}_opt" for s in block.symbols]


# ── Compound parameters: one param, two symbols ──────────────────────────
# Renaming a `coeff*sym + offset` param rebuilds it exactly; a param holding
# an expression over *two* symbols is carried by a closure over the old names
# instead.  Renaming only its symbol list leaves a param no map can bind.


def _compound_param_block():
    """Block whose GPhase carries `(phi + lam)/2` — two symbols in one param.

    The U decomposition is the reachable producer: it only fires for a gate
    set without U, so the block is optimized against an explicit one.
    """
    gateset = qx.GateSet()
    gateset.name = "rz_ry_cx"
    gateset.allowed = {
        qx.GateType.Rz,
        qx.GateType.Ry,
        qx.GateType.Rx,
        qx.GateType.CX,
        qx.GateType.GPhase,
    }
    block = SimpleBlock(1)
    block.u(0, Symbol("theta"), Symbol("phi"), Symbol("lam"))
    block.build()
    return block.optimize(target_gateset=gateset, level=0)


def _u_on_ket_zero(theta, phi, lam):
    """OpenQASM 3 `U(θ, φ, λ)` acting on |0⟩ — analytic, phase-exact (§2.5)."""
    return np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)], dtype=complex)


def test_compound_param_block_really_has_a_compound_param():
    """Guard: without a two-symbol param the tests below are vacuous."""
    widths = [
        len(p.free_symbols()) for cmd in _compound_param_block().flatten() for p in cmd.params
    ]
    assert max(widths) > 1, f"no compound param produced, widths={widths}"


def test_compound_param_survives_a_rename():
    block = _compound_param_block()
    values = {"theta": 0.3, "phi": 0.4, "lam": 0.5}
    renamed = block.refresh_symbols("_r")
    bound = renamed.set_symbols({Symbol(f"{k}_r"): v for k, v in values.items()})
    np.testing.assert_allclose(bound.statevector(), _u_on_ket_zero(**values), atol=1e-13)


def test_compound_param_ignores_a_stale_pre_rename_binding():
    """The silent-wrong-value case: a map carrying *both* the new name and the
    stale old one (as a union map over partially-renamed layers does).  A
    rename that left the closure on the old names reads the stale value, all
    symbols resolve, and a wrong angle comes back with no error raised."""
    block = _compound_param_block().replace_symbols({Symbol("phi"): Symbol("phi_r")})
    bound = block.set_symbols(
        {
            Symbol("theta"): 0.3,
            Symbol("phi_r"): 0.4,
            Symbol("lam"): 0.5,
            Symbol("phi"): 99.0,  # stale — must not reach the circuit
        }
    )
    np.testing.assert_allclose(
        bound.statevector(),
        _u_on_ket_zero(theta=0.3, phi=0.4, lam=0.5),
        atol=1e-13,
    )


def test_binding_before_a_rename_is_not_resurrected():
    """The mirror of the stale-binding case: bind, *then* rename the symbol
    just bound.  flatten() applies every rename before any substitution, so a
    binding still keyed on the old name would miss and leave the symbol free
    again — with `.symbols` already reporting it bound, so the two surfaces
    disagree and `unitary_matrix()` raises on a block that claims no symbols."""
    a, x = Symbol("a"), Symbol("x")
    block = SimpleBlock(1, name="r")
    block.rx(0, a)
    block.build()

    bound_then_renamed = block.set_symbols({a: 0.3}).replace_symbols({a: x})
    assert bound_then_renamed.symbols == ()
    assert bound_then_renamed.free_symbols() == []

    renamed_then_bound = block.replace_symbols({a: x}).set_symbols({x: 0.3})
    np.testing.assert_allclose(
        bound_then_renamed.unitary_matrix(),
        renamed_then_bound.unitary_matrix(),
        atol=1e-15,
    )


def test_compound_param_survives_a_collapsing_rename():
    """Tying two symbols to one shared name is a legitimate rename.  The
    compound's closure must then feed both old slots from the shared new one,
    and `free_symbols` must report the tied name once, not once per old
    symbol."""
    block = _compound_param_block()
    tied = block.replace_symbols({Symbol("phi"): Symbol("x"), Symbol("lam"): Symbol("x")})
    assert sorted(tied.free_symbols()) == ["theta", "x"]
    bound = tied.set_symbols({Symbol("theta"): 0.3, Symbol("x"): 0.4})
    np.testing.assert_allclose(
        bound.statevector(), _u_on_ket_zero(theta=0.3, phi=0.4, lam=0.4), atol=1e-13
    )


def test_compound_param_binds_across_successive_calls():
    """A compound param substitutes all-or-nothing, so binding its two symbols
    in separate calls used to leave it symbolic while ``free_symbols`` already
    reported none left — the block claimed to be bound and then failed to run."""
    block = _compound_param_block()
    bound = block.set_symbols({Symbol("phi"): 0.4}).set_symbols(
        {Symbol("lam"): 0.5, Symbol("theta"): 0.3}
    )
    assert not bound.free_symbols()
    np.testing.assert_allclose(
        bound.statevector(), _u_on_ket_zero(theta=0.3, phi=0.4, lam=0.5), atol=1e-13
    )


def test_later_binding_of_a_symbol_wins():
    """§13 (P1.7): re-binding an already-bound symbol replaces the earlier
    value — the compound (phi+lam)/2 param included — so a loop that rebinds
    sees the new angles, not the first ones."""
    block = _compound_param_block()
    bound = block.set_symbols({Symbol("theta"): 0.3, Symbol("phi"): 0.4, Symbol("lam"): 0.5})
    rebound = bound.set_symbols({Symbol("theta"): 1.9, Symbol("phi"): 1.9, Symbol("lam"): 1.9})
    np.testing.assert_allclose(
        rebound.statevector(), _u_on_ket_zero(theta=1.9, phi=1.9, lam=1.9), atol=1e-13
    )
    np.testing.assert_allclose(
        bound.statevector(), _u_on_ket_zero(theta=0.3, phi=0.4, lam=0.5), atol=1e-13
    )


def test_compound_param_rename_leaves_no_stale_symbol():
    block = _compound_param_block().refresh_symbols("_r")
    stale = {
        s
        for cmd in block.flatten()
        for p in cmd.params
        for s in p.free_symbols()
        if not s.endswith("_r")
    }
    assert not stale, f"renamed block still reports pre-rename symbols: {stale}"


# ── Pairing surfaces: pinned as generation-ordered — do NOT "fix" ────────


def test_ucc_pairing_surface_is_generation_ordered():
    """UCCBlock.symbol_qop_pairs carries singles-then-doubles generation
    order, matching the ucc-generator output pairwise.  This order is semantic
    (symbol i drives generator i) and intentionally NOT sorted — sorting it
    scrambles the circuit.  If this test fails because someone canonicalized
    the pairing list, revert that change."""
    onv = _onv4()
    ucc = UCCBlock(occupation_number_vector=onv)
    _, ssyms = ucc_singles(onv, spin_conserving=True, generalised=False)
    _, dsyms = ucc_doubles(onv, spin_conserving=True, generalised=False, paired=False)
    pair_names = [s.name for s, _ in ucc.symbol_qop_pairs]
    assert pair_names == [s.name for s in ssyms + dsyms]
    # And the public registry is the sorted view of the same set.
    assert set(pair_names) == {s.name for s in ucc.symbols}
    _assert_canonical(ucc, "UCCBlock registry")


def test_trotter_ansatz_pairing_preserves_constructor_order():
    tab = FACTORIES["TrotterAnsatzBlock"]()
    assert [s.name for s, _ in tab.symbol_qop_pairs] == ["z", "y", "x"]
    assert [s.name for s in tab.symbols] == ["x", "y", "z"]


def test_trotter_ansatz_symbolic_time_sorted_into_place():
    tab = FACTORIES["TrotterAnsatzBlock-symbolic-time"]()
    assert [s.name for s in tab.symbols] == ["a_time", "x", "y", "z"]


def test_ucc_rename_keeps_pairing_coherent():
    ucc = UCCBlock(occupation_number_vector=_onv4())
    ucc.build()
    target = ucc.symbols[0]
    renamed = ucc.replace_symbols({target: Symbol("zz_renamed")})
    assert Symbol("zz_renamed") in [s for s, _ in renamed.symbol_qop_pairs]
    assert set(s for s, _ in renamed.symbol_qop_pairs) == set(renamed.symbols)


# ── Cross-process determinism (the original MPI motivation) ──────────────

_DETERMINISM_SNIPPET = """
import json
from qarp.blocks import UCCBlock, ComputationalBasisStateBlock, CompositeBlock
onv = [1, 1, 0, 0]
u = UCCBlock(occupation_number_vector=onv)
u.build()
c = CompositeBlock([ComputationalBasisStateBlock([1, 1, 0, 0]).build(), u], 4)
c.build()
print(json.dumps([[str(s) for s in u.symbols], [str(s) for s in c.symbols]]))
"""


def test_symbol_order_is_hash_seed_independent():
    """sympy free_symbols is a set; its iteration order depends on
    PYTHONHASHSEED.  Canonical ordering must erase that — independent
    processes (which get distinct seeds) must agree on parameter order.  Regression test the
    original c86d691e fix never had."""
    outputs = []
    for seed in ("0", "424242"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        res = subprocess.run(
            [sys.executable, "-c", _DETERMINISM_SNIPPET],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        outputs.append(json.loads(res.stdout.strip().splitlines()[-1]))
    assert outputs[0] == outputs[1]
