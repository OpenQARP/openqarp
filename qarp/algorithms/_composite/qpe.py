from typing import Optional

import numpy as np
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

from ...blocks import AnyBlock, QPEBlock
from ...endianness import bits_to_label, label_to_bits
from ...engines import Engine
from .._composite import CompositeAlgorithm
from .._primitives import PrimitiveAlgorithm, Sampler
from .._utils import dirichlet_kernel_squared, map_binary_to_integer_keys


class QPE(CompositeAlgorithm):
    def __init__(
        self,
        state: AnyBlock,
        unitary: AnyBlock,
        n_ancilla: int,
        primitive: Optional[PrimitiveAlgorithm] = None,
        engine: Optional[Engine] = None,
    ):
        """Quantum Phase Estimation (QPE) algorithm for eigenvalue extraction.

        QPE is a quantum algorithm that estimates the phase (eigenvalue) of an eigenvector of a unitary operator.
        It prepares ancilla qubits in superposition, applies controlled-unitary operations with increasing powers,
        and uses the inverse quantum Fourier transform to extract phase information. The precision of the estimate
        scales exponentially with the number of ancilla qubits.

        Args:
            state: The Block preparing the probe state (typically an eigenstate).
            unitary: The Block of the unitary operator whose eigenvalues are to be estimated.
            n_ancilla: Number of ancilla qubits for phase estimation precision.
            primitive: Primitive algorithm for circuit execution.
            engine: Quantum engine for simulation.
        """
        if primitive is None:
            primitive = Sampler()
        super().__init__(engine=engine, primitive=primitive)

        self.state = state
        self.unitary = unitary
        self.n_ancilla = n_ancilla
        self.freqs = [x / (2**self.n_ancilla) for x in range(2 ** (self.n_ancilla))]

        self.block = None
        self._structured_run = None
        self.distribution = None
        self.bitstring = None
        self.result = None
        self.result_probability = None

    def build(self):
        """
        Build the Canonical Phase Estimation circuit.

        The engine is first offered the block's declared structure
        (``Engine.prepare_structured``, §14 *Structured execution*): an engine
        that can run it never builds the controlled-U ladder, and ``block``
        stays None.  Otherwise the full ``QPEBlock`` is built and compiled.

        Returns:
            self: The instance of the class.
        """
        # Ensure unitary and state are built so their commands are available.
        self.unitary.build()
        self.state.build()

        block = QPEBlock(self.state, self.unitary, self.n_ancilla, self.unitary.n_qubits)
        if isinstance(self.primitive, Sampler) and self.primitive.measured_qubits is None:
            # The block records measurements on the ancilla register only
            # (qubits 0..n_ancilla-1); marginalise the system register out so
            # the phase extraction reads pure ancilla bits — otherwise any
            # eigenstate ≠ |0…0⟩ shifts the result by whole integers.
            self.primitive.measured_qubits = list(range(self.n_ancilla))

        self._structured_run = self.engine.prepare_structured(block, self.primitive)
        if self._structured_run is not None:
            self.block = None
            return self

        self.block = block.build()
        self.primitive.ket = self.block
        # engine.build() builds the primitive itself — building here too would
        # compile every circuit twice.
        self.engine.build([self.primitive])

        return self

    def run(self):
        """
        Run the Canonical Phase Estimation algorithm.

        Returns:
            result: The estimated eigenvalue.
        """
        if self._structured_run is not None:
            self.distribution = self._structured_run.sample()
        else:
            if self.block is None or not self.block.is_built:
                raise ValueError("Circuit not built. Call build() before run().")
            self.distribution = self.engine.run()[0]

        self.bitstring = max(self.distribution, key=self.distribution.get)
        self.result_probability = self.distribution[self.bitstring]
        # Sampler emits LSB-first tuples.
        self.result = bits_to_label(self.bitstring) / 2**self.n_ancilla

        return self.result

    def estimate_phase(self, fit_range=None, verbose=False):
        """
        Estimation of the phase done by fitting the Dirichlet kernel squared to the distribution obtained from the run.

        Args:
            fit_range (tuple or None): Optional (min, max) range between 0 and 1 for fitting. If None, use on the entire histogram.
            verbose (bool): If True, print the fitted phase.

        Returns:
            phi_fit: The estimated phase.
        """
        if self.distribution is None:
            raise ValueError("No distribution to estimate phase from. Run the algorithm first.")
        N = len(self.freqs)
        x = self.freqs
        # Build y from distribution, ensuring all frequencies have a value (0 if not observed)
        y = []
        for i in range(2**self.n_ancilla):
            # Sampler keys are LSB-first tuples.
            bitstring = tuple(label_to_bits(i, self.n_ancilla))
            y.append(self.distribution.get(bitstring, 0.0))

        if fit_range is not None:
            min_range, max_range = fit_range
            selected_indices = [i for i, xi in enumerate(x) if min_range <= xi <= max_range]
            x_selected = [x[i] for i in selected_indices]
            y_selected = [y[i] for i in selected_indices]
        else:
            selected_indices = list(range(len(x)))
            x_selected = x
            y_selected = y

        phi_centroid = np.sum(np.multiply(y_selected, x_selected)) / np.sum(y_selected)
        popt, _ = curve_fit(
            lambda x, phi: dirichlet_kernel_squared(x, phi, N),
            x_selected,
            y_selected,
            p0=phi_centroid,
        )
        phi_fit = popt[0]

        if verbose:
            if fit_range is not None:
                print(f"Fit range: {fit_range}")
            else:
                print("Fit range: (0, 1)")
            print(f"The estimated phase using the Dirichlet kernel is: {phi_fit}")

        return phi_fit

    def plot(self, figsize=None, return_fig=False, max_xticks=11):
        """
        Plot the results of the Canonical Phase Estimation algorithm.

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

        n_bins = 2**self.n_ancilla
        if figsize is None:
            # Widen with the ancilla register so the 2**n_ancilla phase bins stay
            # resolvable; cap so the figure stays on screen.
            figsize = (min(5.5 + 1.4 * max(0, self.n_ancilla - 3), 20.0), 4.0)
        fig, ax = plt.subplots(figsize=figsize)

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
        # Bounded, evenly spaced ticks instead of one per bin (unreadable once
        # n_ancilla is large).
        ax.xaxis.set_major_locator(MaxNLocator(nbins=max_xticks, steps=[1, 2, 2.5, 5, 10]))
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()

        if return_fig:
            return fig, ax
        else:
            plt.show()
            return None
