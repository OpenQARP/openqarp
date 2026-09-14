"""Regenerate the per-(family, stack) workload shims for a track.

The shims are mechanical — one per cell of the family x stack matrix — so they
are generated rather than hand-maintained, and the runner discovers them by
filename.  Run after editing a track's `spec.FAMILIES`:

    python -m benchmarks.gen_shims statevector sampler
"""

import argparse
import importlib
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent

ADAPTERS = {
    "statevector": {
        "numpy": "_np",
        "qarpx": "_qx",
        "qulacs": "_qulacs",
        "aer": "_aer",
        "lightning": "_lightning",
        "qsim": "_qsim",
    },
    "compilation": {
        "reference": "_ref",
        "qarpx": "_qx",
        "qiskit1": "_qk1",
        "qiskit2": "_qk2",
        "qiskit3": "_qk3",
        "tket": "_tket",
    },
    "algorithms": {
        "exact": "_exact",
        "qarpx": "_qx",
        "qulacs": "_qulacs",
        "aer": "_aer",
        "lightning": "_lightning",
        "qsim": "_qsim",
        "ffsim": "_ffsim",
    },
    "sampler": {
        "exact": "_exact",
        "qarpx": "_qx",
        "qulacs": "_qulacs",
        "aer": "_aer",
        "lightning": "_lightning",
        "qsim": "_qsim",
    },
}

_INIT = '"""Thin shims discovered by filename: <family>_<stack>.py.\n\nEach binds one family kernel from _families to one stack adapter, so the stack\nis carried by the filename and the measurement protocol is shared.\n"""\n'


def generate(track: str) -> int:
    spec = importlib.import_module(f"benchmarks.{track}.spec")
    adapters = ADAPTERS[track]
    root = BENCH_DIR / track / "workloads"
    root.mkdir(parents=True, exist_ok=True)
    (root / "__init__.py").write_text(_INIT)

    written = set()
    for family, cfg in spec.FAMILIES.items():
        for stack in cfg["stacks"]:
            module = adapters[stack]
            name = f"{family}_{stack}.py"
            (root / name).write_text(
                f"from benchmarks.{track} import _families, {module}\n\n"
                f'DESCRIPTION, bench, warmup = _families.FACTORIES["{family}"]({module})\n'
            )
            written.add(name)

    stale = [p for p in root.glob("*.py") if p.name not in written and p.name != "__init__.py"]
    for path in stale:
        path.unlink()
    if stale:
        print(f"  removed {len(stale)} stale shim(s): {', '.join(p.name for p in stale)}")
    return len(written)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tracks", nargs="*", choices=sorted(ADAPTERS), default=None)
    args = ap.parse_args()
    for track in args.tracks or sorted(ADAPTERS):
        print(f"{track}: wrote {generate(track)} shims")


if __name__ == "__main__":
    main()
