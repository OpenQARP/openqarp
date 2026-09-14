"""Opt-in diagnostic for the module-scope-qarpx-object landmine (AGENTS.md).

    pytest --leakdiag <targets>

Reports every qarpx-backed object still alive at session end and names the
module global holding it.  Without this, the only symptom is nanobind's
`leaked N instances` at interpreter shutdown, which names unrelated symbols
and points at whichever test happened to run last — not at the module that
actually retains them.

Wired up in conftest.py; inert unless --leakdiag is passed.
"""

import enum
import gc
import sys
import types


def _is_qarpx(obj):
    # qarpx enum members (GateType, Pauli, OptLevel, ...) live on their own
    # class for the process lifetime by design — permanent, never a finding.
    return type(obj).__module__ == "qarpx" and not isinstance(obj, enum.Enum)


def _is_own_machinery(obj, *own):
    # gc.get_referrers surfaces the diagnostic's own frames, list locals, and
    # the calls' argument tuples; counted, they bury the real holder.
    if isinstance(obj, types.FrameType):
        return obj.f_code.co_filename == __file__
    for seq in own:
        if obj is seq:
            return True
        if (
            type(obj) is tuple
            and len(obj) == len(seq)
            and all(a is b for a, b in zip(obj, seq, strict=True))
        ):
            return True
    return False


def report(printer=print):
    """Print live qarpx objects and their module-global holders."""
    gc.collect()
    live = [o for o in gc.get_objects() if _is_qarpx(o)]
    counts: dict[str, int] = {}
    for o in live:
        counts[type(o).__name__] = counts.get(type(o).__name__, 0) + 1

    printer(f"[leakdiag] live qarpx objects at session end: {len(live)}")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        printer(f"[leakdiag]   {n:5d}  {name}")
    if not live:
        return

    printer("[leakdiag] module globals holding them (fix these):")
    live_ids = {id(o) for o in live}
    accounted: set[int] = set()

    def _collect_live(value, out, depth=0):
        # Two container levels: catches {"key": [op, ...]} and namespaces.
        if id(value) in live_ids:
            out.add(id(value))
            return
        if depth >= 2:
            return
        if isinstance(value, dict):
            items = list(value.values())
        elif isinstance(value, (list, tuple, set, frozenset)):
            items = list(value)
        elif hasattr(value, "__dict__") and not isinstance(value, types.ModuleType):
            d = getattr(value, "__dict__", None)
            if not isinstance(d, dict):
                return
            items = list(d.values())
        else:
            return
        for v in items:
            _collect_live(v, out, depth + 1)

    for mod_name, mod in list(sys.modules.items()):
        d = getattr(mod, "__dict__", None)
        if not isinstance(d, dict):
            continue
        for key, value in list(d.items()):
            held: set[int] = set()
            if callable(value) and hasattr(value, "cache_info"):
                # functools.lru_cache wrapper: the cache dict is reachable
                # only through gc, not attributes.
                for referent in gc.get_referents(value):
                    if isinstance(referent, dict):
                        _collect_live(referent, held)
                if held:
                    printer(
                        f"[leakdiag]   {mod_name}.{key}  "
                        f"(lru_cache, {value.cache_info().currsize} entries, "
                        f"{len(held)} leaked objects)"
                    )
            elif not isinstance(value, types.ModuleType):
                _collect_live(value, held)
                if held:
                    printer(f"[leakdiag]   {mod_name}.{key}  ({type(value).__name__})")
            accounted |= held
    if accounted == live_ids:
        return
    live = [o for o in live if id(o) not in accounted]
    if accounted:
        printer(f"[leakdiag] {len(live)} objects not reachable from module globals:")

    # Not a module global: fall back to referrer shapes so the holder is still
    # identifiable (class attribute, closure cell, fixture cache, ...).
    # Two batched heap passes only: per-object gc.get_referrers is a full heap
    # traversal each — O(N × heap) hangs at exactly the leak sizes this
    # fallback exists for.  Owner attribution uses gc.get_referents, which
    # walks direct references and never scans the heap.
    printer("[leakdiag]   none — falling back to referrer shapes:")
    refs = [r for r in gc.get_referrers(*live) if not _is_own_machinery(r, live)]
    containers = [r for r in refs if isinstance(r, (dict, list, tuple, types.CellType))]
    owners: dict[int, set[str]] = {id(c): set() for c in containers}
    for owner in gc.get_referrers(*containers) if containers else []:
        if _is_own_machinery(owner, live, refs, containers):
            continue
        for held in gc.get_referents(owner):
            names = owners.get(id(held))
            if names is not None:
                names.add(type(owner).__name__)
    shapes: dict[str, int] = {}
    for r in refs:
        kind = type(r).__name__
        if isinstance(r, types.FrameType):
            kind = f"frame {r.f_code.co_filename.split('/')[-1]}:{r.f_code.co_name}"
        elif id(r) in owners and owners[id(r)]:
            kind = f"{kind} owned by {sorted(owners[id(r)])[:4]}"
        shapes[kind] = shapes.get(kind, 0) + 1
    for kind, n in sorted(shapes.items(), key=lambda kv: -kv[1]):
        printer(f"[leakdiag]     {n:4d}x  {kind}")
