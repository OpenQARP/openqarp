"""Approved, bounded CPU validation; no sampling, GPU, builds or optimization.

Run as ``python -m scripts.validate_selected_pair_scaling --package-root ...
--output-dir NEW_DIRECTORY``. The supervisor runs serial isolated workers with
fixed wall/RSS/address-space ceilings. All ground-state use is validation-only.
"""

import argparse
import importlib.metadata
import json
import math
import os
import platform
import resource
import signal
import subprocess
import sys
import time
import traceback
from hashlib import sha256
from pathlib import Path

MOLECULES = ("h2o", "n2", "lih")
FCI_SECONDS = {"h2o": 600, "n2": 1800, "lih": 300}
PREFIXES = (8, 32, 64, 128)
HISTORICAL_FIDELITIES = {
    "h2o": {"bare": 0.97355334, "one_pair": 0.97946876},
    "n2": {"bare": 0.91730428, "one_pair": 0.93145926, "two_pairs": 0.94727424},
    "lih": {"bare": 0.97452445, "one_pair": 0.98733860},
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def event(output, **values):
    with (output / "events.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(time_utc=time.time(), **values), allow_nan=False) + "\n")


def frozen(package, molecule):
    from scripts.validate_selected_pair_preparations import load_fixture

    fixture, terms, _, _ = load_fixture(package, molecule + "_one_pair")
    arms = {"bare": [], "one_pair": fixture["selection"]["arms"]["one_pair"]["rotations"]}
    if molecule == "n2":
        second, _, _, _ = load_fixture(package, "n2_two_pairs")
        if second["fixture_sha256"] != fixture["fixture_sha256"]:
            raise ValueError("N2 arms must share the audited molecular input")
        arms["two_pairs"] = second["selection"]["arms"]["two_pairs"]["rotations"]
    return fixture, terms, arms


def spin_counts(label, n):
    return tuple(sum((label >> (2 * p + spin)) & 1 for p in range(n)) for spin in (0, 1))


def request_panel(fixture, terms, arms):
    """Matched structural panel, independent of coefficients, angles and FCI.

    Eight diagonal/reference anchors then round-robin off-diagonal keys from
    eight sources. Mask ordering is a stable hash, not amplitude ranking.
    Only bounded mask candidates are traversed; no computational-basis scan.
    """
    from qarp.operators._selected_pairs import _mask_cover

    n = fixture["n_qubits"] // 2
    counts = tuple(fixture["electron_counts"])
    masks = set().union(*(_mask_cover(terms, r, 16384) for r in arms.values()))
    if len(masks) > 16384:
        raise ValueError("Matched mask-union cap exceeded")
    ordered = sorted(masks - {0}, key=lambda m: sha256(str(m).encode()).digest())
    reference = fixture["reference"]
    sources = [reference]
    for mask in ordered:
        label = reference ^ mask
        if spin_counts(label, n) == counts:
            sources.append(label)
        if len(sources) == 8:
            break
    if len(sources) != 8:
        raise ValueError("Insufficient structural sources for the fixed panel")
    keys = [(source, source) for source in sources]
    seen = set(keys)
    for mask in ordered:
        for source in sources:
            target = source ^ mask
            key = (min(source, target), max(source, target))
            if spin_counts(target, n) == counts and key not in seen:
                keys.append(key)
                seen.add(key)
            if len(keys) == 128:
                return keys
    raise ValueError("Insufficient structural keys for the fixed panel")


def determinant_address(label, n, counts):
    """Signed interleaved LSB -> PySCF alpha-major/beta-minor address."""
    from pyscf.fci import cistring

    if not 0 <= label < 1 << (2 * n) or spin_counts(label, n) != tuple(counts):
        raise ValueError("Label outside admitted spin sector")
    occupied = [q for q in range(2 * n) if label & (1 << q)]
    grouped = sorted(occupied, key=lambda q: (q % 2, q // 2))
    permutation = [grouped.index(q) for q in occupied]
    inversions = sum(a > b for i, a in enumerate(permutation) for b in permutation[i + 1 :])
    strings = [sum(1 << (q // 2) for q in occupied if q % 2 == spin) for spin in (0, 1)]
    return (
        int(cistring.str2addr(n, counts[0], strings[0])),
        int(cistring.str2addr(n, counts[1], strings[1])),
        (-1) ** inversions,
    )


def independent_preparation(label, rotations):
    """Embed literal 16-dimensional transfer exponentials, not production algebra."""
    import numpy as np
    from scipy.linalg import expm

    generator = np.zeros((16, 16), dtype=complex)
    generator[12, 3], generator[3, 12] = 1j, -1j
    state = {label: 1 + 0j}
    for p, q, theta in rotations:
        matrix = expm(-1j * theta * generator)
        bits = (2 * p, 2 * p + 1, 2 * q, 2 * q + 1)
        clear = ~sum(1 << bit for bit in bits)
        updated = {}
        for source, amplitude in state.items():
            column = sum(((source >> bit) & 1) << i for i, bit in enumerate(bits))
            for row in range(16):
                value = matrix[row, column] * amplitude
                if value != 0:
                    target = (source & clear) | sum(
                        ((row >> i) & 1) << b for i, b in enumerate(bits)
                    )
                    updated[target] = updated.get(target, 0j) + value
        state = updated
    return state


def independent_jw(h1, eri, nuclear):
    """OpenFermion ladder/JW oracle for the original H, never expanded U†HU."""
    from openfermion import FermionOperator, jordan_wigner

    fermion = FermionOperator((), nuclear)
    branches = 0

    def insert(word, coefficient):
        nonlocal branches, fermion
        if coefficient == 0:
            return
        branches += (1 << (len(word) + 1)) - 2
        if branches > 8_000_000:
            raise ValueError("Independent JW 8-million branch budget exceeded")
        fermion += FermionOperator(word, float(coefficient))

    n = len(h1)
    for p in range(n):
        for q in range(n):
            for spin in (0, 1):
                insert(((2 * p + spin, 1), (2 * q + spin, 0)), h1[p, q])
            for r in range(n):
                for s in range(n):
                    for spin in (0, 1):
                        for other in (0, 1):
                            insert(
                                (
                                    (2 * p + spin, 1),
                                    (2 * r + other, 1),
                                    (2 * s + other, 0),
                                    (2 * q + spin, 0),
                                ),
                                eri[p, q, r, s] / 2,
                            )
    return jordan_wigner(fermion).terms, branches


def assess_roots(energies, roots, action, converged):
    import numpy as np

    order = np.argsort(energies)
    energies = np.asarray(energies)[order]
    roots = [np.asarray(roots[i]) for i in order]
    norms = [float(abs(np.vdot(v, v) - 1)) for v in roots]
    residuals = [
        float(np.linalg.norm(action(v) - e * v)) for e, v in zip(energies, roots, strict=True)
    ]
    gram = np.array([[np.vdot(a, b) for b in roots] for a in roots])
    orthogonality = float(np.max(np.abs(gram - np.eye(len(roots)))))
    ground = np.flatnonzero(energies - energies[0] <= 1e-7)
    accepted = bool(
        converged and max(norms) <= 1e-10 and max(residuals) <= 1e-8 and orthogonality <= 1e-10
    )
    complete = len(ground) < len(roots)
    return {
        "eigenvalues_Ha": energies.tolist(),
        "normalization_errors": norms,
        "residuals_Ha": residuals,
        "orthogonality_error": orthogonality,
        "converged": bool(converged),
        "numerically_accepted": accepted,
        "ground_manifold_complete": bool(complete),
        "ground_multiplicity": len(ground),
    }, [roots[i] for i in ground]


def molecular_check(package, molecule, output):
    import numpy as np
    from pyscf import ao2mo, fci, gto, lib

    from qarp.operators import selected_pair_amplitudes
    from qarp.operators._selected_pairs import _requested_coefficient

    lib.num_threads(2)
    fixture, terms, arms = frozen(package, molecule)
    context = fixture["problem"]
    n, counts = fixture["n_qubits"] // 2, tuple(fixture["electron_counts"])
    shape = tuple(math.comb(n, c) for c in counts)
    if math.prod(shape) > 20000:
        raise ValueError("FCI sector-dimension admission exceeded")
    keys = request_panel(fixture, terms, arms)[:16]
    event(output, stage="molecular", molecule=molecule, phase="integrals", frozen_keys=keys)
    mol = gto.M(
        atom=context["geometry_angstrom"],
        basis=context["basis"],
        unit="Angstrom",
        charge=0,
        spin=counts[0] - counts[1],
        symmetry=False,
        verbose=0,
        max_memory=1024,
    )
    if tuple(mol.nelec) != counts:
        raise ValueError("Molecule electron counts disagree with frozen sector")
    mo = np.asarray(context["mo_coefficients"])
    orbital_error = float(np.linalg.norm(mo.T @ mol.intor("int1e_ovlp") @ mo - np.eye(n)))
    nuclear = float(mol.energy_nuc())
    if orbital_error > 1e-8 or abs(nuclear - context["nuclear_energy_Ha"]) > 1e-10:
        raise ValueError("Frozen MO or nuclear-energy convention mismatch")
    h1 = mo.T @ (mol.intor("int1e_kin") + mol.intor("int1e_nuc")) @ mo
    eri = ao2mo.restore(1, ao2mo.kernel(mol, mo), n)
    rebuilt, branches = independent_jw(h1, eri, nuclear)
    difference = math.fsum(
        abs(rebuilt.get(w, 0) - terms.get(w, 0)) for w in rebuilt.keys() | terms.keys()
    )
    if difference > 1e-8:
        raise ValueError(f"Independent molecular JW mismatch: L1={difference}")
    absorbed = fci.direct_spin1.absorb_h1e(h1, eri, n, counts, 0.5)

    def action(vector):
        real = fci.direct_spin1.contract_2e(absorbed, np.asarray(vector.real, order="C"), n, counts)
        imag = fci.direct_spin1.contract_2e(absorbed, np.asarray(vector.imag, order="C"), n, counts)
        return real + 1j * imag + nuclear * vector

    def vector(label, rotations):
        sparse = independent_preparation(label, rotations)
        production = selected_pair_amplitudes(2 * n, rotations, label)
        error = max(
            abs(sparse.get(s, 0) - production.get(s, 0)) for s in sparse.keys() | production.keys()
        )
        if error > 1e-12:
            raise ValueError("Literal-exponential preparation differs from production")
        result = np.zeros(shape, dtype=complex)
        for s, amplitude in sparse.items():
            a, b, sign = determinant_address(s, n, counts)
            result[a, b] = sign * amplitude
        return result, error

    states, results = {}, {}
    for arm, rotations in arms.items():
        trial, prep_error = vector(fixture["reference"], rotations)
        states[arm] = trial
        energy = float(np.vdot(trial, action(trial)).real)
        recorded = fixture["selection"]["arms"][arm]["energy_Ha"]
        if abs(energy - recorded) > 2e-9:
            raise ValueError("Independent trial energy disagrees with frozen evidence")
        records = []
        for source, target in keys:
            ket, error_k = vector(source, rotations)
            bra, error_b = vector(target, rotations)
            independent = complex(np.vdot(bra, action(ket)))
            production = _requested_coefficient(
                terms, 2 * n, rotations, source, target, max_term_visits=80000
            )
            error = abs(production - independent)
            if error > 1e-8:
                raise ValueError(f"Independent requested K[{target},{source}] mismatch {error}")
            prep_error = max(prep_error, error_k, error_b)
            records.append(
                {
                    "source": source,
                    "target": target,
                    "value": [production.real, production.imag],
                    "independent_value": [independent.real, independent.imag],
                    "error_Ha": error,
                }
            )
        results[arm] = {
            "energy_Ha": energy,
            "recorded_energy_Ha": recorded,
            "energy_error_Ha": abs(energy - recorded),
            "preparation_error": prep_error,
            "rotations": rotations,
            "requested_elements": records,
        }
    event(output, stage="molecular", molecule=molecule, phase="fci", JW_L1_error_Ha=difference)
    calls, root_count = 0, 2
    while True:
        solver = fci.direct_spin1.FCISolver(mol)
        solver.max_memory = 1024
        solver.conv_tol, solver.conv_tol_residual = 1e-12, 1e-9
        solver.lindep, solver.max_cycle, solver.max_space, solver.pspace_size = 1e-20, 200, 48, 0
        solver.nroots = root_count
        energies, roots = solver.kernel(h1, eri, n, counts, ecore=nuclear)
        calls += 1
        assessment, ground = assess_roots(energies, roots, action, bool(np.all(solver.converged)))
        event(output, stage="molecular", molecule=molecule, phase="roots", **assessment)
        if (
            not assessment["numerically_accepted"]
            or assessment["ground_manifold_complete"]
            or root_count == 8
        ):
            break
        root_count *= 2
    valid = assessment["numerically_accepted"] and assessment["ground_manifold_complete"]
    for arm, trial in states.items():
        fidelity = float(sum(abs(np.vdot(root, trial)) ** 2 for root in ground)) if valid else None
        historical = HISTORICAL_FIDELITIES[molecule][arm]
        results[arm].update(
            fidelity=fidelity,
            historical_rounded_fidelity=historical,
            fidelity_difference=None if fidelity is None else fidelity - historical,
            fidelity_status="new_reproduction" if valid else "unavailable",
        )
    return {
        "status": "passed" if valid else "incomplete",
        "kind": "new_reproduction",
        "molecule": molecule,
        "fixture_sha256": fixture["fixture_sha256"],
        "n_qubits": 2 * n,
        "sector_shape": shape,
        "sector_dimension": math.prod(shape),
        "orbital_orthogonality_error": orbital_error,
        "JW_L1_error_Ha": difference,
        "JW_branch_bound_used": branches,
        "FCI_calls": calls,
        "FCI": assessment,
        "results": results,
        "ground_state_used_for_selection": False,
        "settings": 0,
        "submissions": 0,
        "shots": 0,
        "native_gpu": "not_run",
    }


def planning_check(package, molecule, output):
    from qarp import EXACT
    from qarp.algorithms import GroupedTransitionHadamardTest, SelectedPairReadout
    from qarp.engines import QarpEngine
    from qarp.operators import QubitOperator
    from qarp.resources import Stage, estimate

    fixture, terms, arms = frozen(package, molecule)
    keys = request_panel(fixture, terms, arms)
    hamiltonian = QubitOperator()
    hamiltonian.terms = terms
    results = {}
    for arm, rotations in arms.items():
        start = time.monotonic()
        readout = SelectedPairReadout(
            hamiltonian,
            fixture["n_qubits"],
            rotations,
            engine=QarpEngine(seed=101),
            n_shots=EXACT,
            real_only=True,
            max_masks=16384,
            max_term_visits=1_000_000,
            max_grouping_pairs=8_000_000,
        )
        records, prefixes = [], []
        settings = grouping = 0
        status, reason = "complete", None
        for source, target in keys:
            try:
                residual, offset = readout._residual(source, target)
                charge = len(residual.terms) * (len(residual.terms) - 1)
                if grouping + charge > 8_000_000:
                    raise ValueError("Planning grouping-pair budget exceeded")
                grouping += charge
                primitive = GroupedTransitionHadamardTest(
                    bra=readout._state(target),
                    ket=readout._state(source),
                    operator=residual,
                    real=True,
                    imaginary=False,
                    n_shots=4000,
                )
                circuits = primitive.circuits() if residual.terms else []
                gate_counts = []
                for circuit in circuits:
                    count = estimate(circuit)[Stage.LOGICAL]
                    gate_counts.append(
                        {
                            "n_gates": count.n_gates,
                            "n_1q": count.n_1q,
                            "n_2q": count.n_2q,
                            "n_3q_plus": count.n_3q_plus,
                        }
                    )
                settings += len(circuits)
                records.append(
                    {
                        "source": source,
                        "target": target,
                        "residual_terms": len(residual.terms),
                        "exact_offset_Ha": offset.real,
                        "certified_exact": not residual.terms,
                        "certified_zero": not residual.terms and offset == 0,
                        "logical_QWC_settings": len(circuits),
                        "logical_gates_by_setting": gate_counts,
                    }
                )
            except ValueError as exc:
                status, reason = "budget_exhausted", str(exc)
                break
            if len(records) in PREFIXES:
                previous = (
                    prefixes[-1] if prefixes else {"unique_keys": 0, "logical_QWC_settings": 0}
                )
                snapshot = {
                    "unique_keys": len(records),
                    "logical_QWC_settings": settings,
                    "marginal_settings_per_key": (settings - previous["logical_QWC_settings"])
                    / (len(records) - previous["unique_keys"]),
                    "residual_keys": sum(bool(r["residual_terms"]) for r in records),
                    "certified_zero_keys": sum(r["certified_zero"] for r in records),
                    "exact_keys": sum(r["certified_exact"] for r in records),
                    "nonzero_exact_offsets": sum(r["exact_offset_Ha"] != 0 for r in records),
                    "term_visits": readout.diagnostics["term_visits"],
                    "grouping_pair_budget_used": grouping,
                    "planning_seconds": time.monotonic() - start,
                }
                prefixes.append(snapshot)
                event(output, stage="planning", molecule=molecule, arm=arm, **snapshot)
        results[arm] = {
            "status": status,
            "reason": reason,
            "prefixes": prefixes,
            "keys": records,
            "mask_count": len(readout._cover),
            "logical_QWC_settings": settings,
            "term_visits": readout.diagnostics["term_visits"],
            "grouping_pair_budget_used": grouping,
            "planning_seconds": time.monotonic() - start,
            "within_1024_setting_envelope": status == "complete" and settings <= 1024,
            "coefficient_cache_hits": 0,
            "coefficient_cache_reuse": "not_executed",
            "panel_replay_additional_settings_with_full_key_cache": 0,
            "simulator_reuse": "not_executed",
        }
    return {
        "status": "passed"
        if all(r["status"] == "complete" for r in results.values())
        else "incomplete",
        "kind": "new_QWC_planning_reproduction",
        "molecule": molecule,
        "fixture_sha256": fixture["fixture_sha256"],
        "request_policy": "structural_hash_mask_eight_sources_v1",
        "request_keys": keys,
        "results": results,
        "submissions": 0,
        "shots": 0,
        "executed_settings": 0,
        "quadratures": "real_only",
        "planned_shots_per_setting": 4000,
        "compiler_stage": "LOGICAL; no device or transpilation",
        "historical_grouping_comparable": False,
        "ground_state_used_for_selection": False,
        "native_gpu": "not_run",
    }


def worker(args):
    gib = 6 if args.worker == "molecular" else 4
    resource.setrlimit(resource.RLIMIT_AS, (gib * 2**30, gib * 2**30))
    start = time.monotonic()
    try:
        function = molecular_check if args.worker == "molecular" else planning_check
        result = function(args.package_root, args.molecule, args.output_dir)
    except Exception:
        result = {"status": "failed", "traceback": traceback.format_exc()}
    result.update(
        wall_seconds=time.monotonic() - start,
        peak_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        memory_address_space_ceiling_GiB=gib,
    )
    write_json(args.output_dir / f"{args.worker}_{args.molecule}.json", result)
    return 0 if result["status"] == "passed" else 1


def process_tree_rss(pid):
    """Linux process RSS, including descendants; worker threads share its RSS."""
    try:
        status = Path(f"/proc/{pid}/status").read_text()
        rss = next(
            (
                int(line.split()[1]) * 1024
                for line in status.splitlines()
                if line.startswith("VmRSS:")
            ),
            0,
        )
        children = Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
        return rss + sum(process_tree_rss(int(child)) for child in children)
    except FileNotFoundError:
        return 0


def supervised(command, log, seconds, memory_bytes, env):
    start, peak, reason = time.monotonic(), 0, None
    with log.open("x") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, env=env, start_new_session=True
        )
        try:
            while process.poll() is None:
                peak = max(peak, process_tree_rss(process.pid))
                if time.monotonic() - start >= seconds:
                    reason = "wall_budget_exceeded"
                elif peak > memory_bytes:
                    reason = "RSS_budget_exceeded"
                if reason:
                    os.killpg(process.pid, signal.SIGKILL)
                    break
                time.sleep(0.1)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return {
        "returncode": process.returncode,
        "termination_reason": reason,
        "wall_seconds": time.monotonic() - start,
        "observed_tree_RSS_MiB": peak / 2**20,
        "wall_budget_seconds": seconds,
        "memory_budget_GiB": memory_bytes / 2**30,
    }


def supervise(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.update(
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
        QARP_NUM_THREADS="2",
        PYTHONHASHSEED="0",
    )
    sources = [
        Path(__file__),
        Path(__file__).with_name("validate_selected_pair_preparations.py"),
        Path("qarp/operators/_selected_pairs.py"),
        Path("qarp/algorithms/_composite/selected_pair_readout.py"),
        Path("qarp/algorithms/_primitives/grouped_transition_hadamard_test.py"),
    ]
    manifest = {
        "kind": "new_reproduction",
        "authorization": "stages_1_2_only",
        "limits": {
            "molecular_seconds": 2700,
            "planning_seconds": 900,
            "CPU_threads": 2,
            "molecular_GiB": 6,
            "planning_GiB": 4,
            "PySCF_configured_MiB": 1024,
        },
        "source_sha256": {str(p): sha256(p.read_bytes()).hexdigest() for p in sources},
        "versions": {
            p: importlib.metadata.version(p)
            for p in ("numpy", "scipy", "pyscf", "openfermion", "openqarp")
        },
        "python": platform.python_version(),
        "jobs": [],
        "native_gpu": "not_run",
    }
    for stage, total, gib in (("molecular", 2700, 6), ("planning", 900, 4)):
        stage_start = time.monotonic()
        for molecule in MOLECULES:
            remaining = total - (time.monotonic() - stage_start)
            if remaining <= 0:
                manifest["status"] = "budget_exhausted"
                write_json(args.output_dir / "manifest.json", manifest)
                return 1
            seconds = min(remaining, FCI_SECONDS[molecule] if stage == "molecular" else remaining)
            command = [
                sys.executable,
                "-m",
                "scripts.validate_selected_pair_scaling",
                "--package-root",
                str(args.package_root),
                "--output-dir",
                str(args.output_dir),
                "--worker",
                stage,
                "--molecule",
                molecule,
            ]
            observation = supervised(
                command, args.output_dir / f"{stage}_{molecule}.log", seconds, gib * 2**30, env
            )
            observation.update(stage=stage, molecule=molecule)
            manifest["jobs"].append(observation)
            write_json(args.output_dir / "manifest.json", manifest)
            print(json.dumps(observation), flush=True)
            if observation["returncode"] != 0:
                manifest["status"] = "stopped_after_incomplete_or_failed_check"
                write_json(args.output_dir / "manifest.json", manifest)
                return 1
    manifest["status"] = "complete" if len(manifest["jobs"]) == 6 else "budget_exhausted"
    write_json(args.output_dir / "manifest.json", manifest)
    return 0 if manifest["status"] == "complete" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--worker", choices=("molecular", "planning"), help=argparse.SUPPRESS)
    parser.add_argument("--molecule", choices=MOLECULES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker and not args.molecule:
        parser.error("Internal worker requires a molecule")
    return worker(args) if args.worker else supervise(args)


if __name__ == "__main__":
    sys.exit(main())
