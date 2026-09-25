"""Post-selection of readout distributions and statevectors.

:class:`PostSelection` is an immutable condition on a subset of qubits,
applied to results after the fact — it never touches circuits or engines.
Two application surfaces:

* :meth:`PostSelection.apply` — condition a sampling distribution (a
  :class:`~qarp.SamplingDistribution`, the :class:`~qarp.algorithms.Sampler`
  output, sampled or ``qarp.EXACT`` alike, or a ``{bits-tuple: probability}``
  dict) and report the kept probability mass as the success rate.
* :meth:`PostSelection.apply_statevector` — project a statevector onto the
  condition, returning the renormalised conditional state and the success
  probability ``‖P|ψ⟩‖²``.

Conventions (qarpx LSB throughout): distribution keys are LSB-first tuples
(qubit ``q`` at position ``q``); statevector index bit ``q`` is qubit ``q``.

Fixed-bit conditions (``PostSelection({qubit: bit})``) collapse the selected
qubits to a definite basis state, so they are removed from the output — keys
shrink to the surviving qubits in ascending order, statevectors compress to
``2^(n-k)`` amplitudes.  Sector conditions (:meth:`PostSelection.hamming_weight`,
:meth:`PostSelection.parity`) project onto a *subspace* in which the selected
qubits generally stay entangled with the rest, so the output keeps the full
register width.

Shot-noise caveat: on sampled input the success rate carries statistical
error (σ ≈ √(p(1−p)/N)) and the conditioned distribution rests on an
effective ``N · success_rate`` shots — quote error bars accordingly.  With
``qarp.EXACT`` input both outputs are exact.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ._sampling_distribution import SamplingDistribution, pack_bits
from ._types import SamplingDictionary


@dataclass(frozen=True)
class PostSelected:
    """Conditioned distribution plus the probability mass that survived."""

    distribution: SamplingDistribution
    success_rate: float


class PostSelection:
    """Immutable post-selection condition on a subset of qubits.

    Construct with a ``{qubit index: required bit}`` mapping for fixed-bit
    conditions, or via the sector constructors :meth:`hamming_weight` /
    :meth:`parity`.  Hashable: fixed-bit specs compare by their conditions,
    sector specs by predicate identity — either way a spec can key a cache
    across a parameter sweep.
    """

    __slots__ = ("qubits", "_expected", "_predicate", "_label")

    def __init__(self, conditions: Mapping[int, int]):
        if not conditions:
            raise ValueError("PostSelection requires at least one condition")
        items = sorted(conditions.items())
        for q, b in items:
            if not isinstance(q, int) or isinstance(q, bool) or q < 0:
                raise ValueError(f"qubit indices must be non-negative ints, got {q!r}")
            if b not in (0, 1):
                raise ValueError(f"required bit for qubit {q} must be 0 or 1, got {b!r}")
        self.qubits: Tuple[int, ...] = tuple(q for q, _ in items)
        self._expected: Optional[Tuple[int, ...]] = tuple(int(b) for _, b in items)
        self._predicate: Optional[Callable[[Tuple[int, ...]], bool]] = None
        self._label = "{" + ", ".join(f"{q}: {b}" for q, b in items) + "}"

    # ── Sector constructors ────────────────────────────────────────────

    @classmethod
    def _sector(
        cls,
        qubits: Sequence[int],
        predicate: Callable[[Tuple[int, ...]], bool],
        label: str,
    ) -> "PostSelection":
        qs = tuple(sorted({int(q) for q in qubits}))
        if len(qs) != len(tuple(qubits)):
            raise ValueError("duplicate qubit indices in sector PostSelection")
        if not qs:
            raise ValueError("PostSelection requires at least one qubit")
        if qs[0] < 0:
            raise ValueError("qubit indices must be non-negative")
        obj = object.__new__(cls)
        obj.qubits = qs
        obj._expected = None
        obj._predicate = predicate
        obj._label = label
        return obj

    @classmethod
    def hamming_weight(cls, qubits: Sequence[int], k: int) -> "PostSelection":
        """Keep outcomes with exactly ``k`` ones across ``qubits``
        (particle-number sector under Jordan-Wigner)."""
        k = int(k)
        return cls._sector(
            qubits,
            lambda bits: sum(bits) == k,
            f"hamming_weight(qubits={list(qubits)}, k={k})",
        )

    @classmethod
    def parity(cls, qubits: Sequence[int], even: bool = True) -> "PostSelection":
        """Keep outcomes whose bit-sum over ``qubits`` is even (or odd)."""
        want = 0 if even else 1
        return cls._sector(
            qubits,
            lambda bits: (sum(bits) & 1) == want,
            f"parity(qubits={list(qubits)}, even={even})",
        )

    # ── Core predicate ─────────────────────────────────────────────────

    @property
    def is_fixed(self) -> bool:
        """True for fixed-bit conditions (selected qubits are removed from
        the output); False for sector conditions (full width preserved)."""
        return self._expected is not None

    def _keep(self, bits: Tuple[int, ...]) -> bool:
        if self._expected is not None:
            return bits == self._expected
        assert self._predicate is not None
        return bool(self._predicate(bits))

    # ── Application: distributions ─────────────────────────────────────

    def apply(self, distribution: SamplingDictionary) -> PostSelected:
        """Condition a distribution keyed by LSB-first bit tuples.

        Keys must share one width covering every selected qubit.  Fixed-bit
        specs drop the (now-constant) selected positions from the output
        keys; sector specs keep full-width keys.  Zero surviving mass yields
        an empty distribution and a success rate of 0 — no raise, so
        parameter sweeps survive nodes with vanishing support.
        """
        dist = SamplingDistribution._from_mapping(distribution)
        n_bits = dist.n_bits_measured
        qmax = self.qubits[-1]
        if len(dist) and qmax >= n_bits:
            raise ValueError(
                f"PostSelection on qubit {qmax}, but outcome keys cover only {n_bits} qubits"
            )
        outcomes, probs = dist.outcomes, dist.probabilities
        # The condition sees only the selected bits, so it is evaluated once
        # per distinct selected pattern.
        patterns, inverse = np.unique(pack_bits(outcomes, self.qubits), return_inverse=True)
        k = len(self.qubits)
        keep_pattern = np.fromiter(
            (self._keep(tuple((p >> i) & 1 for i in range(k))) for p in patterns.tolist()),
            dtype=bool,
            count=len(patterns),
        )
        keep = keep_pattern[inverse]
        success = float(probs[keep].sum())
        if self.is_fixed:
            # Dropping bits that are constant across the kept outcomes keeps
            # the packed keys strictly ascending.
            rest = [q for q in range(n_bits) if q not in set(self.qubits)]
            keys, width = pack_bits(outcomes[keep], rest), len(rest)
        else:
            keys, width = outcomes[keep], n_bits
        if success > 0.0:
            kept = SamplingDistribution._wrap(keys.copy(), probs[keep] / success, width)
        else:
            kept = SamplingDistribution._wrap(keys[:0].copy(), np.zeros(0), width)
        return PostSelected(distribution=kept, success_rate=success)

    def success_rate(self, distribution: SamplingDictionary) -> float:
        """Kept probability mass only (shortcut for ``apply(...).success_rate``)."""
        return self.apply(distribution).success_rate

    # ── Application: statevectors ──────────────────────────────────────

    def apply_statevector(self, statevector, n_qubits: int) -> Tuple[np.ndarray, float]:
        """Project a statevector onto the condition; renormalise.

        Returns ``(conditional_state, success_probability)`` with
        ``success = ‖P|ψ⟩‖²``.  Fixed-bit specs return the state of the
        surviving qubits (ascending order, ``2^(n-k)`` amplitudes — the
        selected qubits collapsed to a product basis state and factor out).
        Sector specs return the projected state on the FULL register
        (``2^n``): a subspace projection leaves the selected qubits
        entangled with the rest, so no reduction exists.  Zero success
        returns the zero vector of the appropriate size with ``p = 0.0``.
        """
        sv = np.asarray(statevector)
        dim = 1 << n_qubits
        if sv.shape != (dim,):
            raise ValueError(
                f"statevector has shape {sv.shape}, expected ({dim},) for n_qubits={n_qubits}"
            )
        if self.qubits[-1] >= n_qubits:
            raise ValueError(
                f"PostSelection on qubit {self.qubits[-1]}, but the register "
                f"has only {n_qubits} qubits"
            )

        idx = np.arange(dim)
        sel_bits = [(idx >> q) & 1 for q in self.qubits]
        # Evaluate the predicate once per selected-bit assignment (2^k calls),
        # then broadcast to the 2^n indices.
        mask = np.zeros(dim, dtype=bool)
        n_sel = len(self.qubits)
        for a in range(1 << n_sel):
            assignment = tuple((a >> i) & 1 for i in range(n_sel))
            if not self._keep(assignment):
                continue
            m = np.ones(dim, dtype=bool)
            for i, sb in enumerate(sel_bits):
                m &= sb == assignment[i]
            mask |= m

        success = float(np.sum(np.abs(sv[mask]) ** 2))

        if self.is_fixed:
            survivors = [q for q in range(n_qubits) if q not in set(self.qubits)]
            out = np.zeros(1 << len(survivors), dtype=complex)
            if success > 0.0:
                ridx = np.zeros(dim, dtype=np.int64)
                for j, q in enumerate(survivors):
                    ridx += ((idx >> q) & 1) << j
                # Fixed bits → exactly one accepted assignment → the kept
                # full indices map bijectively onto the reduced indices.
                out[ridx[mask]] = sv[mask] / np.sqrt(success)
            return out, success

        out = np.zeros(dim, dtype=complex)
        if success > 0.0:
            out[mask] = sv[mask] / np.sqrt(success)
        return out, success

    # ── Value semantics ────────────────────────────────────────────────

    def __eq__(self, other) -> bool:
        if not isinstance(other, PostSelection):
            return NotImplemented
        return (self.qubits, self._expected, self._predicate) == (
            other.qubits,
            other._expected,
            other._predicate,
        )

    def __hash__(self) -> int:
        return hash((self.qubits, self._expected, self._predicate))

    def __repr__(self) -> str:
        return f"PostSelection({self._label})"
