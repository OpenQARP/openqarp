"""Run every implementation of every algorithm, check they agree, count lines.

Usage:  python benchmarks/code_volume/run.py [--stacks a,b] [--algorithms x,y]
        (exit 1 on any disagreement)

A missing ``impl/<stack>/<algorithm>.py`` is a gap, not a failure -- the matrix
is filled in column by column.  The notebook next to this file imports the same
functions for display.
"""

import argparse
import pathlib
import re
import subprocess
import sys

import numpy as np
from count_loc import effective_lines, imported_from_common, used_functions

ROOT = pathlib.Path(__file__).parent
IMPL = ROOT / "impl"
REFERENCE = "qarp"

# table: which comparison a stack belongs to.  "primitive" ships no algorithm
# layer, so the user writes the plumbing (its common.py); "framework" ships the
# algorithm itself and the comparison is like-for-like.  Both are reported, they
# answer different questions and must not be averaged together.
STACKS = {
    "qarp": ("qarp", None),
    "qulacs": ("qulacs + openfermion", "primitive"),
    "cirq": ("cirq + openfermion", "primitive"),
    "qiskit_raw": ("qiskit (raw)", "primitive"),
    "pennylane": ("pennylane", "framework"),
    "qiskit_nature": ("qiskit-nature", "framework"),
}

# key/oracle: printed labels of the result and of its independent reference.
# cross: stack-vs-qarp tolerance.  oracle_atol: stack-vs-oracle tolerance, None
# when the oracle bounds the result rather than equalling it (UCCSD is not CCSD).
#
# Both are keyed on the stack's table, because the two tables support different
# claims.  A primitive stack hand-writes the *same* ansatz in the same order, so
# it must agree to machine precision -- anything looser would hide a real bug.
# A framework brings its own excitation ordering and therefore spans a slightly
# different variational manifold; demanding 1e-8 of it would be a rigged test, so
# it is held to chemical accuracy (1.6 mHa) against the independent oracle.
CHEMICAL_ACCURACY = 1.6e-3
ALGORITHMS = {
    "vqe": {
        "key": "VQE",
        # CCSD is not what UCCSD-VQE converges to -- it is a different method, and
        # the variational value sits above it.  For this system the gap is 7.2e-5,
        # so chemical accuracy is still a real independent check rather than none.
        "key_note": "UCCSD is a variational bound, not CCSD",
        "oracle": "CCSD",
        "cross": {"primitive": 1e-8, "framework": 2e-3},
        "oracle_atol": CHEMICAL_ACCURACY,
    },
    "ssvqe": {
        "key": "SS-VQE",
        "oracle": "exact",
        "cross": {"primitive": 1e-8, "framework": 2e-3},
        "oracle_atol": {"primitive": 1e-8, "framework": CHEMICAL_ACCURACY},
    },
    "adapt_vqe": {
        "key": "ADAPT-VQE",
        "oracle": "exact",
        "cross": {"primitive": 1e-8, "framework": 2e-3},
        "oracle_atol": {"primitive": 1e-8, "framework": CHEMICAL_ACCURACY},
    },
    "qaoa": {
        "key": "QAOA",
        "oracle": "exact",
        # Oracle is analytic: for MaxCut at p = 1 on a triangle-free 2-regular graph
        # <C> = n/2 + (n/4) sin(4b) sin(2g), whose maximum is 3n/4.  Using the
        # maximum rather than the formula keeps it independent of how each stack
        # parametrises its angles.  The primitive stacks evaluate <C> exactly and
        # reach it to 1e-8; qiskit-algorithms' QAOA is a *sampling* eigensolver with
        # no exact mode, so even at 1e6 shots it lands ~2e-4 short (and at the 1024
        # default, 1.8% short while *reporting* a value above the maximum).
        "cross": {"primitive": 1e-6, "framework": 2e-3},
        "oracle_atol": {"primitive": 1e-6, "framework": 2e-3},
    },
    "qpe": {
        "key": "QPE",
        "oracle": "exact",
        # The Hamiltonian's two terms commute, so exp(-iHt) factorises with no
        # Trotter error, and the phase 11/16 is exactly representable in four
        # ancilla bits.  Every stack must return it exactly, with probability 1 --
        # no statistical tolerance is warranted and none is given.
        "cross": 1e-8,
        "oracle_atol": 1e-8,
    },
    "vqe_shots": {
        "key": "VQE-shots",
        "oracle": "exact",
        # Statistical, not exact, on every stack: each draws its own shots from its
        # own RNG.  100k shots per group on this Hamiltonian give sigma ~ 2.2 mHa
        # (sqrt(sum c^2 / N)), so chemical accuracy would sit inside 1 sigma and
        # flake.  1e-2 is ~4.5 sigma -- wide enough not to flake, tight enough that
        # a real error (a missed group, a sign, an ungrouped identity) still fails.
        "cross": 1e-2,
        "oracle_atol": 1e-2,
    },
}


# Cells whose number cannot be compared across stacks, with the reason.  The
# implementation still runs and still counts lines; only the numeric check is
# skipped, and the reason is printed so the exclusion is never silent.
NOT_COMPARABLE = {
    ("qiskit_nature", "ssvqe"): (
        "qiskit-nature's SCF orbitals differ from openfermion's in ordering and "
        "phase (same spectrum, related by no qubit permutation), and SS-VQE's "
        "result depends on which basis states seed the search"
    ),
}


def table_of(stack):
    """qarp is the reference; hold it to the strict primitive-stack tolerances."""
    return STACKS[stack][1] or "primitive"


def tolerance(spec, field, stack):
    """A per-table dict, or a scalar that applies to every table."""
    value = spec[field]
    return value[table_of(stack)] if isinstance(value, dict) else value


def script_path(stack, algorithm):
    return IMPL / stack / f"{algorithm}.py"


def run_script(stack, algorithm):
    """Execute one implementation in its own directory; parse its ``key = value`` lines."""
    proc = subprocess.run(
        [sys.executable, f"{algorithm}.py"],
        cwd=IMPL / stack,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"impl/{stack}/{algorithm}.py failed:\n{proc.stderr[-2000:]}")
    values = {}
    for line in proc.stdout.splitlines():
        match = re.match(r"^([A-Za-z_ -]+?)\s*=\s*(.+)$", line.strip())
        if match:
            values[match.group(1).strip()] = _parse(match.group(2))
    return values


def _parse(text):
    text = text.strip()
    if text.startswith("["):
        return np.array([float(x) for x in text.strip("[]").split()])
    try:
        return float(text)
    except ValueError:
        return text


def run_all(stacks=None, algorithms=None, verbose=True):
    """Every present cell of the matrix; returns {algorithm: {stack: values}}."""
    stacks = stacks or list(STACKS)
    algorithms = algorithms or list(ALGORITHMS)
    results = {}
    for algorithm in algorithms:
        results[algorithm] = {}
        for stack in stacks:
            if not script_path(stack, algorithm).exists():
                continue
            if verbose:
                print(f"running impl/{stack}/{algorithm}.py ...", flush=True)
            results[algorithm][stack] = run_script(stack, algorithm)
    return results


def check(results):
    """Disagreements as a list of strings; empty means every cell agrees."""
    failures = []
    for algorithm, stacks in results.items():
        spec = ALGORITHMS[algorithm]
        if REFERENCE not in stacks:
            failures.append(f"{algorithm}: no {REFERENCE} cell to compare against")
            continue
        expected = stacks[REFERENCE][spec["key"]]
        for stack, values in stacks.items():
            if (stack, algorithm) in NOT_COMPARABLE:
                continue
            cross = tolerance(spec, "cross", stack)
            if not np.allclose(values[spec["key"]], expected, atol=cross, rtol=0):
                failures.append(
                    f"{algorithm}: {stack} {values[spec['key']]} != {REFERENCE} {expected}"
                    f" (atol {cross})"
                )
            oracle_atol = tolerance(spec, "oracle_atol", stack)
            if oracle_atol is None:
                continue
            oracle = values.get(spec["oracle"])
            if oracle is None:
                failures.append(f"{algorithm}: {stack} printed no '{spec['oracle']}' oracle")
            elif not np.allclose(values[spec["key"]], oracle, atol=oracle_atol, rtol=0):
                failures.append(
                    f"{algorithm}: {stack} {values[spec['key']]} != {spec['oracle']} {oracle}"
                    f" (atol {oracle_atol})"
                )
    return failures


def _common_lines(stack, algorithm):
    """(full, reached) effective lines of this stack's common.py, 0 if it has none."""
    common = IMPL / stack / "common.py"
    if not common.exists():
        return 0, 0
    script = script_path(stack, algorithm)
    reached = used_functions(common, imported_from_common(script))
    return effective_lines(common), effective_lines(common, only=reached)


def loc_rows(table, algorithms=None):
    """One row per algorithm comparing qarp against every stack of one table."""
    algorithms = algorithms or list(ALGORITHMS)
    competitors = [s for s, (_, kind) in STACKS.items() if kind == table]
    rows = []
    for algorithm in algorithms:
        if not script_path(REFERENCE, algorithm).exists():
            continue
        row = {
            "algorithm": ALGORITHMS[algorithm]["key"],
            REFERENCE: effective_lines(script_path(REFERENCE, algorithm)),
        }
        for stack in competitors:
            path = script_path(stack, algorithm)
            if not path.exists():
                row[stack] = None
                continue
            full, reached = _common_lines(stack, algorithm)
            row[stack] = (effective_lines(path), full, reached)
        rows.append(row)
    return rows, competitors


def format_table(table, algorithms=None):
    """Markdown for one comparison: qarp vs every stack of that table."""
    rows, competitors = loc_rows(table, algorithms)
    if not rows:
        return ""
    # Only script and *reached* lines.  Charging one script the whole common.py is
    # meaningless now that it serves six algorithms -- that belongs amortised.
    header = ["algorithm", "qarp"]
    for stack in competitors:
        header += [f"{STACKS[stack][0]} script", "+ reached"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        cells = [row["algorithm"], f"**{row[REFERENCE]}**"]
        for stack in competitors:
            cell = row[stack]
            cells += ["--", "--"] if cell is None else [str(cell[0]), str(cell[0] + cell[2])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def amortised_table(algorithms=None):
    """Every script plus each stack's common.py counted once."""
    algorithms = [a for a in (algorithms or ALGORITHMS) if script_path(REFERENCE, a).exists()]
    reference = sum(effective_lines(script_path(REFERENCE, a)) for a in algorithms)
    lines = [
        "| Stack | scripts | common.py | total | vs qarp |",
        "|---|---|---|---|---|",
        f"| **qarp** | {reference} | 0 | **{reference}** | -- |",
    ]
    for stack, (label, table) in STACKS.items():
        if table is None or not all(script_path(stack, a).exists() for a in algorithms):
            continue
        scripts = sum(effective_lines(script_path(stack, a)) for a in algorithms)
        common = IMPL / stack / "common.py"
        shared = effective_lines(common) if common.exists() else 0
        total = scripts + shared
        lines.append(
            f"| {label} | {scripts} | {shared} | {total} | {100 * (1 - reference / total):.0f} % |"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stacks", help="comma-separated subset of " + ",".join(STACKS))
    parser.add_argument("--algorithms", help="comma-separated subset of " + ",".join(ALGORITHMS))
    args = parser.parse_args()
    stacks = args.stacks.split(",") if args.stacks else None
    algorithms = args.algorithms.split(",") if args.algorithms else None

    results = run_all(stacks, algorithms)
    print()
    for algorithm, cells in results.items():
        key = ALGORITHMS[algorithm]["key"]
        for stack, values in cells.items():
            print(f"{key:<10} {STACKS[stack][0]:<22} {values[key]}")
    for (stack, algorithm), reason in sorted(NOT_COMPARABLE.items()):
        if algorithm in results and stack in results[algorithm]:
            print(f"SKIP {stack}/{algorithm}: {reason}")
    failures = check(results)
    for failure in failures:
        print("FAIL", failure)
    for table in ("primitive", "framework"):
        body = format_table(table, algorithms)
        if body:
            print(f"\n### vs {table} stacks\n")
            print(body)
    print("\n### amortised over every algorithm present\n")
    print(amortised_table(algorithms))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
