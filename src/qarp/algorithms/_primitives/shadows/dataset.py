"""``ShadowDataset`` — the inert, reusable record of one randomized-measurement
campaign.

Holds only plain data plus a stateless kernel — no ket, circuits, live primitive,
or qarpx objects — so it is safe to keep long-term (leak-free), serialize, and
merge.  ``collect once, estimate many`` lives here: hand it to
:meth:`estimator` for any observable, without re-collecting.

Serialization is a schema-versioned ``to_dict``/``from_dict`` over an **NPZ**
container (integer arrays + a JSON metadata member).  NPZ is chosen over HDF5
because ``h5py`` is not a declared dependency; note the pre-commit hook forbids
*committing* ``.npz`` — this only affects the repo, not a user's runtime
``save()`` (our tests write to a ``tmp_path``).
"""

from __future__ import annotations

import json
import os

import numpy as np

from .kernels import KERNELS, ShadowKernel

SCHEMA_VERSION = 2

# One (setting, counts) pair per measurement setting; counts maps an LSB outcome
# bitmask to an integer shot count (sampled) or a probability (shot-exact).
Record = tuple[np.ndarray, dict[int, float]]


class ShadowDataset:
    """A collected campaign: settings + outcomes + the ensemble kernel."""

    def __init__(
        self,
        kernel: ShadowKernel,
        n_qubits: int,
        records: list[Record],
        shot_exact: bool = False,
    ):
        self.kernel = kernel
        self.n_qubits = n_qubits
        self.records = records
        self.shot_exact = shot_exact

    # --- size / combination ----------------------------------------------------

    def __len__(self) -> int:
        """Total snapshots: shot count for a sampled campaign, number of settings
        for a shot-exact one."""
        if self.shot_exact:
            return len(self.records)
        return int(sum(w for _, counts in self.records for w in counts.values()))

    @property
    def n_settings(self) -> int:
        return len(self.records)

    def __add__(self, other: "ShadowDataset") -> "ShadowDataset":
        return self.merge(other)

    def merge(self, other: "ShadowDataset") -> "ShadowDataset":
        """Concatenate two compatible campaigns (same ensemble, ``n_qubits``,
        ``shot_exact``).  Batching is round-robin by setting index, so the merged
        batches stay evenly mixed between the two campaigns."""
        if self.kernel.ensemble != other.kernel.ensemble:
            raise ValueError(
                f"cannot merge datasets of different ensembles "
                f"({self.kernel.ensemble!r} vs {other.kernel.ensemble!r})"
            )
        if self.n_qubits != other.n_qubits:
            raise ValueError(
                f"cannot merge datasets of different widths ({self.n_qubits} vs {other.n_qubits})"
            )
        if self.shot_exact != other.shot_exact:
            raise ValueError("cannot merge a shot-exact dataset with a sampled one")
        return ShadowDataset(
            self.kernel, self.n_qubits, self.records + other.records, self.shot_exact
        )

    # --- estimation ------------------------------------------------------------

    def estimator(self):
        """Return a :class:`~.estimator.ShadowEstimator` over this dataset."""
        from .estimator import ShadowEstimator

        return ShadowEstimator(self)

    # --- serialization ---------------------------------------------------------

    def to_dict(self) -> dict:
        """Schema-versioned logical form: metadata + flat integer arrays."""
        settings = (
            np.stack([s for s, _ in self.records]).astype(np.int8)
            if self.records
            else np.zeros((0, self.n_qubits), dtype=np.int8)
        )
        # Outcomes are stored as a per-qubit bit-array [n_flat, n_qubits] (LSB:
        # bit q = (outcome >> q) & 1), not a single packed integer — so the schema
        # has no qubit-width ceiling (a packed int would overflow past 63 qubits).
        idx, out_bits, weights = [], [], []
        for i, (_, counts) in enumerate(self.records):
            for outcome, weight in counts.items():
                idx.append(i)
                out_bits.append([(int(outcome) >> q) & 1 for q in range(self.n_qubits)])
                weights.append(weight)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "ensemble": self.kernel.ensemble,
            "n_qubits": self.n_qubits,
            "shot_exact": self.shot_exact,
            "kernel_params": self.kernel.params(),
        }
        return {
            "meta": meta,
            "settings": settings,
            "flat_setting_idx": np.asarray(idx, dtype=np.int64),
            "flat_outcome_bits": np.asarray(out_bits, dtype=np.int8).reshape(-1, self.n_qubits),
            "flat_weight": np.asarray(weights, dtype=np.float64),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ShadowDataset":
        meta = d["meta"]
        version = meta.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ShadowDataset schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION}); the file was written by a "
                "different qarp version."
            )
        ensemble = meta["ensemble"]
        if ensemble not in KERNELS:
            raise ValueError(f"unknown shadow ensemble {ensemble!r} in serialized dataset")
        n_qubits = int(meta["n_qubits"])
        shot_exact = bool(meta["shot_exact"])
        kernel = KERNELS[ensemble].from_params(n_qubits, meta.get("kernel_params", {}))

        settings = np.asarray(d["settings"], dtype=np.int8)
        idx = np.asarray(d["flat_setting_idx"], dtype=np.int64)
        out_bits = np.asarray(d["flat_outcome_bits"], dtype=np.int8)  # [n_flat, n_qubits]
        weights = np.asarray(d["flat_weight"], dtype=np.float64)

        records: list[Record] = [(settings[i], {}) for i in range(settings.shape[0])]
        for i, bits, weight in zip(idx, out_bits, weights, strict=True):
            outcome = int(sum(int(b) << q for q, b in enumerate(bits)))  # LSB re-pack
            w = float(weight) if shot_exact else int(round(float(weight)))
            records[int(i)][1][outcome] = w
        return cls(kernel, n_qubits, records, shot_exact)

    @staticmethod
    def _npz_path(path: str | os.PathLike[str]) -> str:
        """Filename numpy's ``savez`` actually writes: ``.npz`` appended when
        absent.  Applied on both ``save`` and ``load`` so a bare path round-trips
        (``save('run')`` writes ``run.npz`` and ``load('run')`` reads it)."""
        p = os.fspath(path)
        return p if p.endswith(".npz") else p + ".npz"

    def save(self, path: str | os.PathLike[str]) -> None:
        """Write to an NPZ file (schema-versioned).  Accepts any ``os.PathLike``."""
        d = self.to_dict()
        np.savez(
            self._npz_path(path),
            meta=np.array(json.dumps(d["meta"])),
            settings=d["settings"],
            flat_setting_idx=d["flat_setting_idx"],
            flat_outcome_bits=d["flat_outcome_bits"],
            flat_weight=d["flat_weight"],
        )

    @classmethod
    def load(cls, path: str | os.PathLike[str]) -> "ShadowDataset":
        with np.load(cls._npz_path(path), allow_pickle=False) as npz:
            d = {
                "meta": json.loads(str(npz["meta"])),
                "settings": npz["settings"],
                "flat_setting_idx": npz["flat_setting_idx"],
                "flat_outcome_bits": npz["flat_outcome_bits"],
                "flat_weight": npz["flat_weight"],
            }
        return cls.from_dict(d)

    def __repr__(self) -> str:
        return (
            f"ShadowDataset(ensemble={self.kernel.ensemble!r}, n_qubits={self.n_qubits}, "
            f"n_settings={self.n_settings}, shot_exact={self.shot_exact})"
        )
