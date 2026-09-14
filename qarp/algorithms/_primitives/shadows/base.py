"""``ShadowProtocol`` — the classical-shadow collector, an ordinary primitive.

A ``PrimitiveAlgorithm`` (like ``TermwiseHadamardTest``): it binds an ``operator``,
builds one measurement circuit per random setting, and ``run(results) -> float``
returns the median-of-means ``<operator>``.  Because the random-measurement
circuits are operator-independent, the same results estimate *any* observable, so
the collector also exposes them as a reusable :class:`~.dataset.ShadowDataset`
(``shadow.dataset``) that a :class:`~.estimator.ShadowEstimator` turns into further
expectations without re-collecting.

Two modes: **collect** (no ``dataset`` passed) samples a fresh campaign from
``ket``; **reuse** (a ``dataset`` passed) estimates ``operator`` from an existing
dataset with no circuits.  ``run()`` never clears the circuits — the engine reuses
them under ``rebuild=False``/``batch_run`` — so freeing memory is an explicit
:meth:`release` (the detached dataset stays valid regardless).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Optional, Union

import numpy as np

from qarp.operators import QubitOperator

from ....blocks import AnyBlock
from ..primitive_algorithm import PrimitiveAlgorithm
from ..target import Target
from ._estimators import n_batches_for
from .dataset import ShadowDataset
from .estimator import _decompose
from .kernels import ShadowKernel


class ShadowProtocol(PrimitiveAlgorithm):
    """Base collector for a randomized-measurement shadow ensemble."""

    supported_targets = frozenset({Target.EXPECTATION_VALUE})
    supports_exact = True  # shot-exact per setting; NOT setting-exact
    gradient_kind = "none"  # median-of-means is not linear in the settings' statistics
    requires_noiseless = True  # the ideal inverse channel would bias a noisy campaign

    def __init__(
        self,
        operator: Union[str, QubitOperator],
        ket: Optional[AnyBlock] = None,
        *,
        dataset: Optional[ShadowDataset] = None,
        n_settings: int = 1000,
        n_shots: int = 1,
        seed: Optional[int] = None,
    ):
        if operator is None:
            raise ValueError("PauliShadow requires an operator.")
        super().__init__(
            ket=ket,
            operator=operator,
            n_shots=n_shots,
            target=Target.EXPECTATION_VALUE,
        )
        self.n_settings = n_settings
        self.seed = seed
        self._passed_dataset = dataset
        self._dataset: Optional[ShadowDataset] = dataset
        self._settings: Optional[np.ndarray] = None

    # --- ensemble hooks --------------------------------------------------------

    @abstractmethod
    def _sample_settings(self, rng: np.random.Generator, n_settings: int, n_qubits: int):
        """Draw ``n_settings`` random settings for an ``n_qubits`` register."""

    @abstractmethod
    def _setting_block(self, setting) -> AnyBlock:
        """Build ``ket -> U(setting) -> measure-all`` for one setting."""

    @abstractmethod
    def _kernel(self, n_qubits: int) -> ShadowKernel:
        """Return this ensemble's stateless estimation kernel."""

    # --- primitive lifecycle ---------------------------------------------------

    def build(self):
        if self._passed_dataset is not None:  # reuse mode
            # Reuse runs no circuits — run() is pure arithmetic on stored snapshots,
            # never touching the engine's simulator or noise.  So the noiseless
            # guard (which protects live measurement) is moot here; clear it on this
            # instance so a pre-collected dataset re-estimates on any engine.
            self.requires_noiseless = False
            self.n_qubits = self._passed_dataset.n_qubits
            _, terms, _ = _decompose(self.operator)
            for term in terms:
                for qubit, _ in term:
                    if qubit >= self.n_qubits:
                        raise ValueError(
                            f"operator acts on qubit {qubit}, but the passed dataset "
                            f"has only {self.n_qubits} qubits."
                        )
            self.sub_blocks = []
            self._dataset = self._passed_dataset
            return self

        # collect mode
        if self.ket is None:
            raise ValueError(
                "PauliShadow in collect mode needs a ket (or pass dataset= to reuse "
                "an existing campaign)."
            )
        self.ket.build()
        if self.ket.symbols:
            raise ValueError(
                "PauliShadow requires a concrete (fully-bound) ket; bind the "
                f"parameters first. Free symbols: {self.ket.symbols}"
            )
        self.n_qubits = self.ket.n_qubits
        floor = n_batches_for(0.05)
        if self.n_settings < floor:
            raise ValueError(
                f"n_settings={self.n_settings} is below the default batch count "
                f"({floor}); use at least {floor} settings."
            )
        rng = np.random.default_rng(self.seed)
        self._settings = self._sample_settings(rng, self.n_settings, self.n_qubits)
        self.sub_blocks = [self._setting_block(s) for s in self._settings]
        self._dataset = None  # invalidate any previously collected dataset
        return self

    def run(self, results: list) -> float:
        if self._passed_dataset is not None:  # reuse mode — no circuits ran
            dataset = self._passed_dataset
        else:
            if self._settings is None:
                raise RuntimeError("run() called before build() sampled settings")
            shot_exact = getattr(results[0], "is_exact", False) if results else False
            records = [
                (np.asarray(setting, dtype=np.int8), dict(sr.counts))
                for setting, sr in zip(self._settings, results, strict=True)
            ]
            dataset = ShadowDataset(self._kernel(self.n_qubits), self.n_qubits, records, shot_exact)
            self._dataset = dataset
        return dataset.estimator().expval(self.operator).value

    # --- reuse layer + memory --------------------------------------------------

    @property
    def dataset(self) -> ShadowDataset:
        """The collected (or passed) :class:`ShadowDataset`.  Raises if neither a
        dataset was passed nor a campaign collected yet."""
        if self._dataset is None:
            raise RuntimeError(
                "no ShadowDataset yet: run the primitive (engine.run([shadow])) to "
                "collect one, or pass dataset= at construction."
            )
        return self._dataset

    def release(self) -> None:
        """Free the compiled circuits in place (the detached dataset stays valid).

        ``run()`` never does this automatically — the engine reuses the compiled
        circuits under ``rebuild=False`` and across ``batch_run`` parameter sets.
        After ``release()``, a re-run needs a rebuild.
        """
        self.sub_blocks = []
        self.compiled_circuits = []
        self._n_qubits_list = []
