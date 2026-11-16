"""Cheap frozen-energy reproduction; no FCI, circuit execution or GPU work.

Supply the separately stored handoff package. This command never reoptimizes
orbitals/angles or claims to recover the missing original fidelity reports.
"""

import argparse
import importlib.metadata
import json
import platform
from hashlib import sha256
from pathlib import Path

from qarp.operators._selected_pairs import _requested_coefficient

_INDEX_SHA256 = "62e9b0accc6806d7402b2551d8150d26f9c4401a85ddff06f5ccc170d4f59034"
_CASES = ("h2o_one_pair", "n2_one_pair", "n2_two_pairs", "lih_one_pair")


def load_fixture(package_root: Path, case: str) -> tuple[dict, dict, dict, bytes]:
    """Load only the audited fixtures and verify their canonical identities."""
    if case not in _CASES:
        raise ValueError("Only the four approved shallow-pair cases are admitted")
    index_bytes = (package_root / "PACKAGE_INDEX.json").read_bytes()
    if sha256(index_bytes).hexdigest() != _INDEX_SHA256:
        raise ValueError("Handoff index differs from the audited package")
    index = json.loads(index_bytes)
    registration = next(item for item in index["fixtures"] if item["case"] == case)
    relative = f"fixtures/{case}.json"
    fixture_path = package_root / relative
    if fixture_path.stat().st_size > 8 * 2**20:
        raise ValueError("Fixture exceeds 8 MiB budget")
    raw = fixture_path.read_bytes()
    fixture = json.loads(raw)
    content = {k: v for k, v in fixture.items() if k != "fixture_sha256"}
    canonical = json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if sha256(canonical).hexdigest() != registration["fixture_sha256"]:
        raise ValueError("Canonical fixture hash mismatch")
    if fixture["fixture_sha256"] != registration["fixture_sha256"]:
        raise ValueError("Fixture identity mismatch")
    terms = {
        tuple((int(bit), letter) for bit, letter in item["word"]): complex(
            item["real"], item["imaginary"]
        )
        for item in fixture["terms"]
    }
    h_bytes = json.dumps(
        [(word, value.real.hex(), value.imag.hex()) for word, value in sorted(terms.items())]
    ).encode()
    if sha256(h_bytes).hexdigest() != fixture["hamiltonian_sha256"]:
        raise ValueError("Hamiltonian fingerprint mismatch")
    return fixture, terms, index, raw


def reproduce(package_root: Path, case: str) -> dict:
    """Verify the frozen input and compare only bare/prepared sparse energies."""
    fixture, terms, index, raw = load_fixture(package_root, case)
    arm = "two_pairs" if case == "n2_two_pairs" else "one_pair"
    results = {}
    for name in ("bare", arm):
        selection = fixture["selection"]["arms"][name]
        energy = _requested_coefficient(
            terms,
            fixture["n_qubits"],
            selection["rotations"],
            fixture["reference"],
            fixture["reference"],
            max_term_visits=80000,
        )
        error = abs(energy - selection["energy_Ha"])
        if error > 2e-9:
            raise ValueError(f"Frozen {name} energy differs by {error} Ha")
        results[name] = {
            "energy_Ha": energy.real,
            "recorded_energy_Ha": selection["energy_Ha"],
            "absolute_error_Ha": error,
            "rotations": selection["rotations"],
        }
    return {
        "kind": "new_reproduction",
        "validation": "frozen_sparse_energy_comparison_only",
        "case": case,
        "source_commit": index["source_commit"],
        "fixture_sha256": fixture["fixture_sha256"],
        "file_sha256": sha256(raw).hexdigest(),
        "hamiltonian_sha256": fixture["hamiltonian_sha256"],
        "n_qubits": fixture["n_qubits"],
        "reference": fixture["reference"],
        "results": results,
        "energy_lowering_mHa": 1000 * (results["bare"]["energy_Ha"] - results[arm]["energy_Ha"]),
        "fidelity": None,
        "fidelity_status": "FCI reproduction pending approved expensive validation",
        "native_gpu_validation": "not_run",
        "settings": 0,
        "submissions": 0,
        "shots": 0,
        "ground_state_used_for_selection": False,
        "versions": {
            "python": platform.python_version(),
            **{
                package: importlib.metadata.version(package)
                for package in ("openqarp", "numpy", "scipy")
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--case", choices=_CASES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; choose a new reproduction filename")
    result = reproduce(args.package_root, args.case)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{args.case}: {result['energy_lowering_mHa']:.7f} mHa; fidelity/GPU unverified")


if __name__ == "__main__":
    main()
