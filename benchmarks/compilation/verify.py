"""Independent oracles for the compilation track (§18).

Two properties have to hold for a compiled circuit to be worth timing, and
neither is a comparison against another compiler:

1. **It still computes the same unitary.**  Checked at small width against a
   dense unitary built by kron products in `statevector.verify` — the full
   matrix, not one column of it, so a compiler that happens to agree on
   |0...0> is still caught.
2. **It fits the device.**  Every two-qubit gate lands on an edge of the
   architecture the compiler was given.  This is what catches the most likely
   unfairness in the whole track: handing one stack an easier coupling map.

Both run over every stack, ours included.

Usage:  python -m benchmarks.compilation.verify

The direct module command also runs the full MQT QCEC matrix at every
published width and writes a host-local certification record.  Nightly smoke
calls :func:`run_all` and therefore checks one bounded representative matrix.
"""

import argparse
import importlib.metadata
import itertools
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from benchmarks.common import environment, peak_mem_mib
from benchmarks.compilation import architectures, checks, inputs
from benchmarks.compilation import spec as compilation_spec
from benchmarks.statevector import verify as sv_verify

TOL = 1e-9
STACKS = ("_ref", "_qx", "_qk1", "_qk2", "_qk3", "_tket")
CASES = (
    ("hea", "line", 5),
    ("qv", "grid", 6),
    ("trotter", "line", 5),
    ("mqt", "line", 5),
)
QCEC_SMOKE_CASES = CASES
QCEC_CELL_TIMEOUT_S = 60
CERTIFICATION_SCHEMA = 1
CERTIFICATION_DIR = Path(__file__).resolve().parents[1] / "results" / "compilation_certification"
CERTIFYING_CHECKERS = {
    "decision_diagram_alternating",
    "decision_diagram_construction",
    "preprocessing",
}
# Recorded when QCEC returns no checker record at all.  Deliberately outside
# CERTIFYING_CHECKERS: an unattributed verdict is not a certificate.
PROVENANCE_UNAVAILABLE = "provenance_unavailable"
REPO_ROOT = Path(__file__).resolve().parents[2]

# The numpy reference compiler performs no placement and publishes no layout;
# its identity is asserted by the dense oracle below, not assumed.  Every other
# stack, qarpx included, declares `initial_layout_of` (§14).
_IDENTITY_INITIAL_LAYOUT = {"_ref"}


def _load(name: str):
    return __import__(f"benchmarks.compilation.{name}", fromlist=["x"])


def _input_permutation_matrix(n: int, order) -> np.ndarray:
    """Relabel qubit q -> order[q] as a basis permutation."""
    dim = 1 << n
    out = np.zeros((dim, dim), dtype=complex)
    for index in range(dim):
        moved = 0
        for q in range(n):
            if (index >> q) & 1:
                moved |= 1 << order[q]
        out[moved, index] = 1.0
    return out


def check_unitary_equivalence() -> list[str]:
    """Compiled unitary equals the fixture's, up to global phase and input relabelling.

    A compiler may permute the *inputs* as well as the outputs — that is what
    qiskit's `initial_index_layout` and pytket's `initial_map` are for, and it
    costs nothing because these circuits are executed from |0...0>, which is
    permutation-symmetric.  So the assertion is that the compiled unitary, with
    the reported final layout undone, equals the reference under *some* input
    relabelling, found by exhaustive search.  That is much stronger than the
    per-row |0...0> check: a wrong circuit matches none of the n! candidates.

    Where a stack publishes its initial layout, the discovered permutation must
    also equal it — which is what catches an adapter reading the wrong field.
    """
    failures = []
    for shape, topology, n in CASES:
        ops = inputs.FIXTURES[shape](n)
        reference = sv_verify.circuit_unitary(n, ops)
        for name in STACKS:
            stack = _load(name)
            compiled = stack.compile_circuit(n, stack.prepare(n, ops, topology))
            undone = checks.undo_layout(stack.to_ops(n, compiled), stack.layout_of(compiled))
            candidate = sv_verify.circuit_unitary(n, undone)

            best, best_order = None, None
            for order in itertools.permutations(range(n)):
                delta = _phase_aligned_matrix_diff(
                    candidate @ _input_permutation_matrix(n, order), reference
                )
                if best is None or delta < best:
                    best, best_order = delta, order
            assert best is not None
            if best > TOL:
                failures.append(
                    f"{shape}/{topology}[n={n}]/{stack.LABEL}: compiled unitary matches no "
                    f"input relabelling (closest differs by {best:.2e})"
                )
                continue

            declared = getattr(stack, "initial_layout_of", None)
            stated = declared(compiled) if declared is not None else None
            if stated is None and name in _IDENTITY_INITIAL_LAYOUT:
                stated = list(range(n))
            if stated is not None and tuple(stated) != best_order:
                assert best_order is not None
                failures.append(
                    f"{shape}/{topology}[n={n}]/{stack.LABEL}: declared initial layout "
                    f"{list(stated)} but the unitary needs {list(best_order)}"
                )
    return failures


def _phase_aligned_matrix_diff(a: np.ndarray, b: np.ndarray) -> float:
    pivot = int(np.argmax(np.abs(b)))
    flat_a, flat_b = a.reshape(-1), b.reshape(-1)
    if abs(flat_b[pivot]) < 1e-12:
        return float(np.max(np.abs(a - b)))
    ratio = flat_a[pivot] / flat_b[pivot]
    phase = ratio / abs(ratio)
    return float(np.max(np.abs(a - phase * b)))


def _qcec_verdict_sets():
    """Partition every QCEC verdict under this track's equivalence relation.

    Standalone compilation is compared modulo one global phase (§18).  QCEC's
    user guide defines ``equivalent_up_to_global_phase`` as that exact relation;
    the broader ``equivalent_up_to_phase`` verdict may include relative phase
    and is deliberately not accepted as a certificate.
    """
    from mqt.qcec.pyqcec import EquivalenceCriterion

    certified = {
        EquivalenceCriterion.equivalent,
        EquivalenceCriterion.equivalent_up_to_global_phase,
    }
    uncertain = {
        EquivalenceCriterion.probably_equivalent,
        EquivalenceCriterion.probably_not_equivalent,
        EquivalenceCriterion.equivalent_up_to_phase,
        EquivalenceCriterion.no_information,
    }
    refuted = {EquivalenceCriterion.not_equivalent}
    return certified, uncertain, refuted


def _verdict_status(verdict) -> str:
    certified, uncertain, refuted = _qcec_verdict_sets()
    if verdict in certified:
        return "certified"
    if verdict in uncertain:
        return "uncertain"
    if verdict in refuted:
        return "refuted"
    raise ValueError(f"unclassified QCEC verdict: {verdict!r}")


def _physical_to_logical(layout, n: int):
    """Invert an adapter's logical->physical list for MQT Core metadata."""
    from mqt.core.ir import Permutation

    logical_to_physical = list(range(n)) if layout is None else list(layout)
    if sorted(logical_to_physical) != list(range(n)):
        raise ValueError(f"layout is not a permutation of 0..{n - 1}: {logical_to_physical}")
    return Permutation({physical: logical for logical, physical in enumerate(logical_to_physical)})


def _initial_layout(stack_name: str, stack, compiled, n: int) -> list:
    declared = getattr(stack, "initial_layout_of", None)
    if declared is not None:
        layout = declared(compiled)
        if layout is not None:
            return list(layout)
    if stack_name in _IDENTITY_INITIAL_LAYOUT:
        return list(range(n))
    raise ValueError(f"{stack.LABEL} does not expose its initial layout")


def _checker_provenance(result) -> tuple[list[dict], str]:
    """Return QCEC checker records, tolerating the 3.9 Python binding defect."""
    try:
        raw = result.checker_results
    except RuntimeError:
        # mqt-qcec 3.9 raises std::bad_cast for this documented property.  Its
        # JSON representation exposes the same native checker records.
        return list(result.json().get("checkers", [])), "result.json().checkers"
    if isinstance(raw, dict):
        return [dict(checker=name, **value) for name, value in raw.items()], "checker_results"
    return list(raw), "checker_results"


def _classify(verdict, checker_names: list[str]) -> tuple[str, list[str]]:
    """Certification status of a verdict given the checkers that produced it."""
    names = list(dict.fromkeys(checker_names)) or [PROVENANCE_UNAVAILABLE]
    status = _verdict_status(verdict)
    if status == "certified" and not CERTIFYING_CHECKERS.intersection(names):
        status = "uncertain"
    return status, names


def _case_key(shape: str, topology: str, n: int, stack_label: str) -> str:
    return f"{shape}_{topology}[n={n}]/{stack_label}"


def _certify_case(shape: str, topology: str, n: int, stack_name: str) -> dict:
    """Compile and certify one cell; intended to run in a fresh process."""
    started = time.perf_counter()
    stack = _load(stack_name)
    base = {
        "key": _case_key(shape, topology, n, stack.LABEL),
        "family": f"{shape}_{topology}",
        "shape": shape,
        "topology": topology,
        "n": n,
        "stack": stack.LABEL,
    }
    try:
        from mqt import core
        from mqt.qcec import verify as qcec_verify

        ops = inputs.FIXTURES[shape](n)
        compiled = stack.compile_circuit(n, stack.prepare(n, ops, topology))
        reference = core.load(inputs.to_qiskit(n, ops))
        candidate = core.load(inputs.to_qiskit(n, stack.to_ops(n, compiled)))
        candidate.initial_layout = _physical_to_logical(
            _initial_layout(stack_name, stack, compiled, n), n
        )
        candidate.output_permutation = _physical_to_logical(stack.layout_of(compiled), n)
        result = qcec_verify(
            reference,
            candidate,
            parallel=False,
            run_simulation_checker=False,
            run_zx_checker=False,
            timeout=QCEC_CELL_TIMEOUT_S,
            backpropagate_output_permutation=False,
            reconstruct_swaps=False,
        )
        checkers, provenance_api = _checker_provenance(result)
        status, checker_names = _classify(result.equivalence, [c["checker"] for c in checkers])
        return {
            **base,
            "status": status,
            "verdict": result.equivalence.name,
            "checker": ", ".join(checker_names),
            "checker_results": checkers,
            "provenance_api": provenance_api,
            "qcec_runtime_s": result.preprocessing_time + result.check_time,
            "runtime_s": time.perf_counter() - started,
            "peak_mem_mib": peak_mem_mib(),
        }
    except Exception as exc:
        return {
            **base,
            "status": "error",
            "verdict": "error",
            "checker": "none",
            "checker_results": [],
            "runtime_s": time.perf_counter() - started,
            "peak_mem_mib": peak_mem_mib(),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _isolated_case(shape: str, topology: str, n: int, stack_name: str) -> dict:
    command = [
        sys.executable,
        "-m",
        "benchmarks.compilation.verify",
        "--qcec-cell",
        shape,
        topology,
        str(n),
        stack_name,
    ]
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=QCEC_CELL_TIMEOUT_S + 15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        stack = _load(stack_name)
        return {
            "key": _case_key(shape, topology, n, stack.LABEL),
            "family": f"{shape}_{topology}",
            "shape": shape,
            "topology": topology,
            "n": n,
            "stack": stack.LABEL,
            "status": "uncertain",
            "verdict": "timeout",
            "checker": "timeout",
            "checker_results": [],
            "runtime_s": QCEC_CELL_TIMEOUT_S + 15,
            "peak_mem_mib": None,
        }
    if proc.returncode != 0:
        stack = _load(stack_name)
        detail = (proc.stderr or proc.stdout).strip()[-2000:]
        return {
            "key": _case_key(shape, topology, n, stack.LABEL),
            "family": f"{shape}_{topology}",
            "shape": shape,
            "topology": topology,
            "n": n,
            "stack": stack.LABEL,
            "status": "error",
            "verdict": "error",
            "checker": "none",
            "checker_results": [],
            "runtime_s": 0.0,
            "peak_mem_mib": None,
            "error": detail or f"cell process exited {proc.returncode}",
        }
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _full_qcec_cases() -> tuple[tuple[str, str, int], ...]:
    return tuple(
        (cfg["shape"], cfg["topology"], n)
        for cfg in compilation_spec.FAMILIES.values()
        for n in cfg["sizes"]
    )


def _git_state() -> tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=REPO_ROOT
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    )
    return commit, dirty


def certification_table(cases=None) -> dict:
    """Certify every stack in ``cases`` in fresh, hard-time-limited processes."""
    selected = _full_qcec_cases() if cases is None else tuple(cases)
    commit, dirty = _git_state()
    env = environment()
    env["versions"]["mqt-qcec"] = importlib.metadata.version("mqt.qcec")
    env["versions"]["mqt-core"] = importlib.metadata.version("mqt.core")
    cells = {}
    for shape, topology, n in selected:
        for stack_name in STACKS:
            cell = _isolated_case(shape, topology, n, stack_name)
            cells[cell["key"]] = cell
    return {
        "schema": CERTIFICATION_SCHEMA,
        "track": "compilation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_commit": commit,
        "source_dirty": dirty,
        "equivalence_relation": "unitary equality modulo one global phase",
        "environment": env,
        "cases": cells,
    }


def _certification_failures(record: dict) -> list[str]:
    failures = []
    for key, cell in record["cases"].items():
        if cell["status"] == "certified":
            continue
        detail = cell.get("error", cell["verdict"])
        failures.append(f"{key}: QCEC {cell['status']} ({detail})")
    return failures


def check_qcec_equivalence(cases=QCEC_SMOKE_CASES) -> list[str]:
    """Use exact QCEC checking on a bounded case set, modulo global phase."""
    return _certification_failures(certification_table(cases))


def write_certification_record(record: dict) -> Path:
    """Write a host/version/commit-stamped certification record atomically."""
    CERTIFICATION_DIR.mkdir(parents=True, exist_ok=True)
    host = re.sub(r"[^a-z0-9_.-]+", "-", platform.node().lower()).strip("-") or "host"
    machine = re.sub(r"[^a-z0-9_.-]+", "-", platform.machine().lower()).strip("-")
    path = CERTIFICATION_DIR / f"{host}-{machine}.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return path


def check_qcec_verdict_policy() -> list[str]:
    """Pin QCEC's complete enum partition and the strict phase policy."""
    from mqt.qcec.pyqcec import EquivalenceCriterion

    certified, uncertain, refuted = _qcec_verdict_sets()
    failures = []
    if certified | uncertain | refuted != set(EquivalenceCriterion):
        failures.append("QCEC verdict policy does not cover every EquivalenceCriterion")
    if (certified & uncertain) or (certified & refuted) or (uncertain & refuted):
        failures.append("QCEC verdict policy classes overlap")
    return failures


def check_qcec_negative_controls() -> list[str]:
    """QCEC must accept global phase and refute known wrong unitaries."""
    from mqt.qcec import verify as qcec_verify
    from qiskit import QuantumCircuit

    failures = []
    identity = QuantumCircuit(1)
    global_phase = QuantumCircuit(1, global_phase=0.37)
    global_result = qcec_verify(
        identity,
        global_phase,
        parallel=False,
        run_simulation_checker=False,
        run_zx_checker=False,
    )
    if _verdict_status(global_result.equivalence) != "certified":
        failures.append(f"QCEC did not accept a global phase: {global_result.equivalence.name}")

    wrong_cases: dict[str, tuple[list, list]] = {
        "dropped gate": ([("h", 0)], []),
        "rz sign flip": ([("h", 0), ("rz", 0, 0.4)], [("h", 0), ("rz", 0, -0.4)]),
        "off-|0> phase": ([], [("rz", 0, np.pi)]),
    }
    for label, (reference_ops, candidate_ops) in wrong_cases.items():
        result = qcec_verify(
            inputs.to_qiskit(1, reference_ops),
            inputs.to_qiskit(1, candidate_ops),
            parallel=False,
            run_simulation_checker=False,
            run_zx_checker=False,
        )
        if _verdict_status(result.equivalence) != "refuted":
            failures.append(f"QCEC did not refute {label}: {result.equivalence.name}")
    return failures


def check_qiskit_transport() -> list[str]:
    """The op-list transport must preserve an independently built unitary."""
    from qiskit.quantum_info import Operator

    ops = [
        ("h", 0),
        ("rx", 1, 0.2),
        ("ry", 2, 0.3),
        ("rz", 0, 0.4),
        ("cx", 0, 2),
        ("swap", 1, 2),
    ]
    reference = sv_verify.circuit_unitary(3, ops)
    transported = np.asarray(Operator(inputs.to_qiskit(3, ops)).data)
    delta = _phase_aligned_matrix_diff(transported, reference)
    return [] if delta <= TOL else [f"qiskit transport unitary differs by {delta:.2e}"]


def check_certification_rendering() -> list[str]:
    """The generated table must fail closed for every non-current state."""
    from benchmarks.make_tables import _compilation_certification_cell

    key = "hea_line[n=6]/qarpx"
    cell = {
        "status": "certified",
        "verdict": "equivalent",
        "checker": "decision_diagram_alternating",
        "runtime_s": 0.1,
        "peak_mem_mib": 10.0,
    }
    record = {
        "schema": CERTIFICATION_SCHEMA,
        "track": "compilation",
        "source_commit": "abc",
        "source_dirty": False,
        "equivalence_relation": "unitary equality modulo one global phase",
        "environment": {"host": "host"},
        "cases": {key: cell},
    }

    def render(candidate, commit="abc"):
        return _compilation_certification_cell(
            candidate, key, expected_commit=commit, expected_host="host"
        )

    failures = []
    if not render(record).startswith("✓ "):
        failures.append("current exact certification did not render as certified")
    if render(None).startswith("✓ "):
        failures.append("missing certification rendered as certified")
    if render(record, commit="def").startswith("✓ "):
        failures.append("stale certification rendered as certified")
    noncertified = {
        "probably_equivalent": "uncertain",
        "probably_not_equivalent": "uncertain",
        "equivalent_up_to_phase": "uncertain",
        "no_information": "uncertain",
        "not_equivalent": "refuted",
        "timeout": "uncertain",
        "error": "error",
    }
    for verdict, status in noncertified.items():
        candidate = {
            **record,
            "cases": {key: {**cell, "status": status, "verdict": verdict}},
        }
        if render(candidate).startswith("✓ "):
            failures.append(f"{verdict} rendered as certified")
    absent = {**record, "cases": {}}
    if render(absent).startswith("✓ "):
        failures.append("absent certification cell rendered as certified")
    dirty = {**record, "source_dirty": True}
    if render(dirty).startswith("✓ "):
        failures.append("dirty-tree certification rendered as certified")
    simulation = {
        **record,
        "cases": {key: {**cell, "checker": "decision_diagram_simulation"}},
    }
    if render(simulation).startswith("✓ "):
        failures.append("simulation-only provenance rendered as certified")
    return failures


def check_qcec_provenance_policy() -> list[str]:
    """A verdict with no checker record is never a certificate.

    QCEC 3.9/3.10 raise on ``checker_results`` and the JSON fallback can come
    back empty; "no checker ran" and "provenance unreadable" are then
    indistinguishable, so the empty case is recorded as such and fails closed.
    """
    from mqt.qcec.pyqcec import EquivalenceCriterion as EC

    cases = {
        (EC.equivalent, ("decision_diagram_alternating",)): "certified",
        (EC.equivalent_up_to_global_phase, ("preprocessing",)): "certified",
        (EC.equivalent, ()): "uncertain",
        (EC.equivalent, ("decision_diagram_simulation",)): "uncertain",
        (EC.probably_equivalent, ("decision_diagram_alternating",)): "uncertain",
        (EC.not_equivalent, ()): "refuted",
    }
    failures = []
    for (verdict, checkers), expected in cases.items():
        status, names = _classify(verdict, list(checkers))
        if status != expected:
            failures.append(
                f"{verdict.name} via {checkers or 'nothing'}: {status}, want {expected}"
            )
        if not checkers and names != [PROVENANCE_UNAVAILABLE]:
            failures.append(f"empty provenance recorded as {names}")
    return failures


def check_git_state_is_cwd_independent() -> list[str]:
    """The record's commit stamp must not depend on where the module is invoked."""
    import os
    import tempfile

    here = os.getcwd()
    try:
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            commit, _ = _git_state()
    except Exception as exc:  # noqa: BLE001 — any failure is the finding
        return [f"_git_state failed outside the repo: {type(exc).__name__}: {exc}"]
    finally:
        os.chdir(here)
    return [] if re.fullmatch(r"[0-9a-f]{40}", commit) else [f"bad commit stamp {commit!r}"]


def check_coupling_respected() -> list[str]:
    """Every routed stack's two-qubit gates land on architecture edges."""
    failures = []
    for shape, topology, n in CASES:
        ops = inputs.FIXTURES[shape](n)
        for name in STACKS:
            stack = _load(name)
            if not getattr(stack, "ROUTED", True):
                continue
            compiled = stack.compile_circuit(n, stack.prepare(n, ops, topology))
            bad = architectures.violations(stack.to_ops(n, compiled), topology, n)
            if bad:
                failures.append(
                    f"{shape}/{topology}[n={n}]/{stack.LABEL}: "
                    f"{bad} two-qubit gate(s) off the coupling map"
                )
    return failures


def check_unrouted_input_violates() -> list[str]:
    """The coupling check must be able to fail — a negative control.

    An all-pairs circuit on a line cannot be legal unrouted; if this ever
    reports zero violations, the check is vacuous and every green row above
    means nothing.
    """
    ops = inputs.FIXTURES["trotter"](6)
    if architectures.violations(ops, "line", 6) == 0:
        return ["coupling check is vacuous: unrouted all-pairs circuit reported as legal"]
    return []


CHECKS = (
    check_unitary_equivalence,
    check_coupling_respected,
    check_unrouted_input_violates,
    check_qcec_verdict_policy,
    check_qcec_negative_controls,
    check_qiskit_transport,
    check_certification_rendering,
    check_qcec_provenance_policy,
    check_git_state_is_cwd_independent,
)


def run_all(*, full_qcec: bool = False) -> list[str]:
    failures = []
    for check in CHECKS:
        found = check()
        print(f"  {'FAIL' if found else 'ok  '} compilation/verify/{check.__name__}")
        failures += found
    cases = None if full_qcec else QCEC_SMOKE_CASES
    record = certification_table(cases)
    found = _certification_failures(record)
    print(f"  {'FAIL' if found else 'ok  '} compilation/verify/check_qcec_equivalence")
    failures += found
    if full_qcec:
        path = write_certification_record(record)
        print(f"  wrote {path}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--qcec-cell", nargs=4, metavar=("SHAPE", "TOPOLOGY", "N", "STACK"))
    args = parser.parse_args()
    if args.qcec_cell:
        shape, topology, n, stack_name = args.qcec_cell
        print(json.dumps(_certify_case(shape, topology, int(n), stack_name)))
    else:
        sys.exit(1 if run_all(full_qcec=True) else 0)
