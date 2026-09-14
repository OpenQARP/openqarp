from typing import Dict, Optional, Tuple

import numpy as np

from qarp.operators._qubit_operator import QubitOperator


class qDRIFT:
    def __init__(
        self,
        H: QubitOperator,
        samples: int,
        ratio: Optional[float] = None,
        verbose: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Implements qDRIFT approximation for QubitOperators.

        qDRIFT provides a randomized compilation technique that approximates
        a Hamiltonian by sampling its Pauli terms according to their weights.
        Supports both fully randomized and partially randomized variants.

        References:
            Campbell, E. (2019). Random compiler for fast Hamiltonian simulation.
            Physical Review Letters, 123(7), 070503.

        Args:
            H: QubitOperator to approximate
            samples: Number of samples to draw from the operator
            ratio: Fraction of terms to keep deterministically for partially
                   randomized qDRIFT (between 0 and 1). If None, uses fully
                   randomized qDRIFT.
            verbose: If True, print diagnostic information
            seed: Optional seed for this instance's local random generator

        Raises:
            ValueError: If samples < 0 or ratio not in [0, 1]
            TypeError: If H is not a QubitOperator
        """
        if not isinstance(H, QubitOperator):
            raise TypeError(f"H must be a QubitOperator, got {type(H)}")
        if not isinstance(samples, int) or samples < 0:
            raise ValueError(f"samples must be a positive integer, got {samples}")
        if ratio is not None and (not isinstance(ratio, float) or not 0 <= ratio <= 1):
            raise ValueError(f"ratio must be a float between 0 and 1, got {ratio}")

        self.H = H
        self.samples = samples
        self.ratio = ratio
        self.verbose = verbose
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def _compute_normalization(self) -> Tuple[float, Dict, Dict]:
        """Compute normalization constant and probability distributions.

        Returns:
            Tuple containing:
                - lambda_val: 1-norm of coefficients
                - probabilities: Dict mapping terms to their probabilities
                - terms_normalized: Dict mapping terms to normalized coefficients
        """
        terms = self.H.terms.items()

        # Compute 1-norm (lambda)
        lambda_val = sum(abs(coeff) for _, coeff in terms)

        if lambda_val == 0:
            raise ValueError("Hamiltonian has zero norm")

        # Build probability distribution and normalized terms
        probabilities = {term: abs(coeff) / lambda_val for term, coeff in terms}
        terms_normalized = {term: coeff / lambda_val for term, coeff in terms}

        return lambda_val, probabilities, terms_normalized

    def qdrift(self) -> QubitOperator:
        """Apply fully randomized qDRIFT approximation.

        Samples Pauli terms according to their weight distribution and
        constructs an approximate Hamiltonian.

        Returns:
            Approximate QubitOperator with sampled terms

        Raises:
            ValueError: If samples is 0 (at least 1 sample required)
        """
        if self.samples == 0:
            raise ValueError(
                "qdrift() requires at least 1 sample. Use partially_randomized() for deterministic-only mode."
            )

        lambda_val, probabilities, terms_normalized = self._compute_normalization()

        # Extract terms and probabilities
        terms_list = list(probabilities.keys())
        probs_list = np.array(list(probabilities.values()))

        # Normalize probabilities
        probs_list /= np.sum(probs_list)

        # Sample terms according to probabilities
        sampled_indices = self._rng.choice(
            len(terms_list), size=self.samples, replace=True, p=probs_list
        )
        sampled_terms = [terms_list[int(index)] for index in sampled_indices]
        sampled_terms_unique = list(dict.fromkeys(sampled_terms))

        if self.verbose:
            print(f"Sampled {len(sampled_terms_unique)} unique terms out of {self.samples} samples")
            print(f"Original Hamiltonian had {len(terms_list)} terms")

        # Build effective Hamiltonian
        Heff = QubitOperator()
        for term in sampled_terms_unique:
            Heff += QubitOperator(term, terms_normalized[term])

        return lambda_val * Heff

    def partially_randomized(self) -> QubitOperator:
        """Apply partially randomized qDRIFT approximation.

        Keeps the most significant terms deterministically and samples
        the remaining terms randomly according to their weights.

        Returns:
            Approximate QubitOperator with deterministic + sampled terms

        Raises:
            ValueError: If ratio was not provided during initialization
        """
        if self.ratio is None:
            raise ValueError(
                "ratio parameter required for partially randomized qDRIFT. "
                "Provide ratio in __init__ or use qdrift() method instead."
            )

        lambda_val, probabilities, terms_normalized = self._compute_normalization()

        # Sort terms by probability (descending)
        sorted_terms = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

        # Split into deterministic and random parts
        num_deterministic = max(1, round(self.ratio * len(sorted_terms)))
        deterministic_terms = dict(sorted_terms[:num_deterministic])
        random_terms = dict(sorted_terms[num_deterministic:])

        # Build deterministic Hamiltonian
        H_deterministic = QubitOperator()
        for term in deterministic_terms:
            H_deterministic += QubitOperator(term, terms_normalized[term])

        # Compute lambda for random part (hoist the .terms view out of the
        # genexpr — the attribute access re-validates the cached dict view on
        # every iteration otherwise)
        h_terms = self.H.terms
        lambda_random = sum(abs(h_terms[term]) for term in random_terms) / lambda_val

        if self.verbose:
            print(f"Deterministic terms: {num_deterministic}/{len(sorted_terms)}")
            print(f"λ_random = {lambda_random:.4f} (should be << 1 for optimal performance)")

        # Sample from random terms
        if random_terms and self.samples > 0:
            terms_list = list(random_terms.keys())
            probs_list = np.array(list(random_terms.values()))
            probs_list /= np.sum(probs_list)  # Normalize

            sampled_indices = self._rng.choice(
                len(terms_list), size=self.samples, replace=True, p=probs_list
            )
            sampled_random = [terms_list[int(index)] for index in sampled_indices]
            sampled_random_unique = list(dict.fromkeys(sampled_random))

            # Build random Hamiltonian
            H_random = QubitOperator()
            for term in sampled_random_unique:
                H_random += QubitOperator(term, terms_normalized[term])

            if self.verbose:
                print(f"Sampled {len(sampled_random_unique)} random terms")
        else:
            H_random = QubitOperator()
            if self.verbose:
                if self.samples == 0:
                    print("No sampling (samples=0, using deterministic terms only)")
                else:
                    print("No random terms to sample (all terms kept deterministically)")

        # Combine deterministic and random parts
        Heff = H_deterministic + H_random

        return lambda_val * Heff
