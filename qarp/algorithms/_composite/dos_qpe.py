from typing import Optional

import numpy as np
from matplotlib.ticker import MaxNLocator

from ...blocks import AnyBlock, DickeStateBlock, DOSQPEBlock, HnBlock
from ...engines import Engine
from .._composite import CompositeAlgorithm
from .._primitives import PrimitiveAlgorithm, Sampler
from .._utils import map_binary_to_integer_keys


class DOSQPE(CompositeAlgorithm):
    def __init__(
        self,
        unitary: AnyBlock,
        n_ancilla: int,
        hamming_weight: Optional[int] = None,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        """Density of States Quantum Phase Estimation (DOS-QPE) algorithm (arXiv:2510.14744).

        DOS-QPE is a variant of the Quantum Phase Estimation algorithm that estimates the density of states
        of a given unitary operator using a probe state. The probe state will be a mixed state prepared by
        a unitary, an entangling layer with a purification qubit register, and then tracing out the purification qubits.
        The probe state can either be a maximally mixed state (using the HnBlock), or a Dicke state of given Hamming weight.

        Args:
            unitary: The Block of the unitary operator whose DOS is to be estimated.
            n_ancilla: Number of ancilla qubits for phase estimation precision.
            hamming_weight: Hamming weight for Dicke state preparation. If None, uses maximally mixed state (via Hn layer).
            primitive: Primitive algorithm for circuit execution.
            engine: Quantum engine for simulation.
        """
        if primitive is None:
            primitive = Sampler()
        super().__init__(engine=engine, primitive=primitive)

        self.unitary = unitary
        self.n_ancilla = n_ancilla
        self.hamming_weight = hamming_weight

        self.block = None
        self.n_qubits = None
        self.state = None
        self.distribution = None
        self._plan = None

    @property
    def freqs(self):
        """Get the frequency grid for the ancilla register."""
        if self.n_ancilla is None:
            raise ValueError("n_ancilla not set. Build the algorithm first.")
        return np.array([x / (2**self.n_ancilla) for x in range(2 ** (self.n_ancilla))])

    def build(self):
        """
        Build the DOS Phase Estimation circuit.

        First offers the problem to the engine's structured fast path
        (``Engine.prepare_structured_qpe`` — matrix exponentiation, the
        controlled-U ladder is never compiled).  Engines without the path,
        or refusing it (EXACT readout, noise, routing, parametric U — see
        ``QarpEngine.prepare_structured_qpe``), return None and the full
        DOSQPEBlock circuit is built instead.

        Returns:
            self: The instance of the class.
        """
        self.n_qubits = self.unitary.build().n_qubits
        if self.n_qubits is None:
            raise ValueError("Unitary block must have a defined number of qubits.")

        if self.hamming_weight is None:
            self.state = HnBlock(self.n_qubits).build()
        else:
            self.state = DickeStateBlock(self.n_qubits, self.hamming_weight).build()

        self._plan = self.engine.prepare_structured_qpe(
            "dosqpe", self.unitary, self.state, self.n_ancilla, self.primitive
        )
        if self._plan is not None:
            return self

        # Generic path: build and compile the full DOSQPE circuit.
        self.block = DOSQPEBlock(
            self.state,
            self.unitary,
            self.n_ancilla,
            self.n_qubits,
            measure=True,
        ).build()

        self.primitive.ket = self.block
        if isinstance(self.primitive, Sampler) and self.primitive.measured_qubits is None:
            # The block records measurements on the ancilla register only
            # (qubits 0..n_ancilla-1); marginalise the system register out so
            # the phase extraction reads pure ancilla bits — otherwise any
            # eigenstate ≠ |0…0⟩ shifts the result by whole integers.
            self.primitive.measured_qubits = list(range(self.n_ancilla))
        # engine.build() builds the primitive itself — building here too would
        # compile every circuit twice.
        self.engine.build([self.primitive])

        return self

    def run(self):
        """
        Run the DOS Phase Estimation algorithm.

        Returns:
            distribution: The distribution of the measurement results.
        """
        if self._plan is not None:
            # Fast path: matrix-exponentiation DOSQPE.
            self.distribution = self._plan.sample()
        else:
            if self.block is None or not self.block.is_built:
                raise ValueError("Circuit not built. Call build() before run().")
            self.distribution = self.engine.run()[0]

        return self.distribution

    def _auto_figsize(self):
        """Default figure size scaled to the ancilla register.

        The x-axis carries ``2**n_ancilla`` phase bins, so the figure widens with
        ``n_ancilla`` to keep the bars resolvable, capped at 20" so it stays on
        screen.  Height is fixed.
        """
        width = min(5.5 + 1.4 * max(0, self.n_ancilla - 3), 20.0)
        return (width, 4.0)

    def _draw_distribution(self, ax, max_xticks):
        """Draw the DOS-QPE probability distribution onto ``ax`` as bars whose
        width tracks the ``2**n_ancilla`` bin spacing, with a bounded number of
        x-ticks.  Both adapt to ``n_ancilla`` so the plot does not get crowded as
        the ancilla register grows.
        """
        n_bins = 2**self.n_ancilla
        x_freqs = np.arange(n_bins) / n_bins
        # Sampler keys are LSB-first; reverse each key so that
        # map_binary_to_integer_keys (which reads MSB-first) maps them correctly.
        res_spectrum = map_binary_to_integer_keys(
            {k[::-1]: v for k, v in self.distribution.items()}
        )
        heights = np.array([res_spectrum.get(i, 0.0) for i in range(n_bins)])

        # One bar per bin, width in data coordinates -> scales with n_ancilla
        # automatically (unlike a fixed point-size linewidth, which overlaps).
        ax.bar(x_freqs, heights, width=1.0 / n_bins, color="C0", alpha=0.6, linewidth=0)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Eigenvalues")
        ax.set_ylabel("Probabilities")
        # Bounded, evenly spaced ticks instead of one per bin (which is unreadable
        # once n_ancilla is large).
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks, steps=[1, 2, 2.5, 5, 10]))

        if self.hamming_weight is None:
            ax.set_title("probe: Maximally mixed state")
        else:
            ax.set_title(f"probe: Dicke state |{self.n_qubits}, {self.hamming_weight}>")

    def plot(self, figsize=None, return_fig=False, max_xticks=11):
        """
        Plot the results of the DOS Phase Estimation algorithm.

        Args:
            figsize (tuple): Size of the figure. If None, scales with n_ancilla.
            return_fig (bool): If True, return (fig, ax) for external saving/customization.
            max_xticks (int): Upper bound on the number of x-axis ticks.

        Returns:
            (fig, ax) if return_fig is True, otherwise None.
        """
        import matplotlib.pyplot as plt  # deferred: ~0.2 s of import, plotting only

        if self.distribution is None:
            raise ValueError("No distribution to plot. Run the algorithm first.")

        fig, ax = plt.subplots(figsize=figsize or self._auto_figsize())
        self._draw_distribution(ax, max_xticks)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if return_fig:
            return fig, ax
        else:
            plt.show()
            return None

    def plot_against_spectrum(
        self,
        unique_eigs,
        normalized_degeneracy,
        unique_occ_numbers,
        figsize=None,
        return_fig=False,
        max_xticks=11,
    ):
        """
        Plot the results of the DOS Phase Estimation algorithm against the spectrum.

        Args:
            unique_eigs (list): Unique eigenvalues.
            normalized_degeneracy (list): Normalized degeneracy.
            unique_occ_numbers (list): Unique occupation numbers.
            figsize (tuple): Size of the figure. If None, scales with n_ancilla.
            return_fig (bool): If True, return (fig, ax) for external saving/customization.
            max_xticks (int): Upper bound on the number of x-axis ticks.

        Returns:
            (fig, ax) if return_fig is True, otherwise None.
        """
        import matplotlib.pyplot as plt  # deferred: ~0.2 s of import, plotting only

        if self.distribution is None:
            raise ValueError("No distribution to plot. Run the algorithm first.")

        fig, ax = plt.subplots(figsize=figsize or self._auto_figsize())
        self._draw_distribution(ax, max_xticks)

        colors = [f"C{i}" for i in range(16)]
        # Thinner reference lines when many eigenvalues are overlaid, so a dense
        # spectrum does not wash the axis out.
        eig_lw = max(1.0, min(3.0, 300.0 / max(1, len(unique_eigs))))
        ax.vlines(
            unique_eigs,
            ymin=np.zeros(len(unique_eigs)),
            ymax=normalized_degeneracy,
            colors=[colors[i] for i in unique_occ_numbers],
            linestyles="dashed",
            lw=eig_lw,
        )

        # manually add legend entries
        for i in range(self.n_qubits + 1):  # type: ignore[arg-type]
            if i in unique_occ_numbers:
                ax.plot([], [], color=colors[i], lw=3, label=f"Occ. #: {i}")

        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()
        plt.tight_layout()

        if return_fig:
            return fig, ax
        else:
            plt.show()
            return None
