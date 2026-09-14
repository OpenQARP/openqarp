from typing import List, Optional, Tuple

import numpy as np

from ..._postselection import PostSelection
from .._block import SimpleBlock
from .._prepares_known_state import prepares_known_state
from ._multi_controlled import _apply_mc_ry, _apply_mcx_path


def _apply_comparator(block, domain_qubits: List[int], constant: int, flag_qubit: int) -> None:
    """Toggle ``flag_qubit`` iff the domain register's value is ``>=
    constant`` (``0 <= constant < 2**len(domain_qubits)``) — a "prefix
    match" walk over ``constant``'s bits (MSB first): at each bit where
    ``constant`` has a 0, the domain register having a 1 there (with every
    higher bit tied so far) means it has already exceeded ``constant``, so
    that branch's contribution is toggled in; the final call (after the
    full walk) handles exact equality. Self-inverse — applying it twice
    with the domain register unchanged exactly undoes it, since every gate
    here is its own inverse and neither the domain register nor
    ``flag_qubit``'s *control* role changes between the two calls (only
    `flag_qubit` itself toggles, which each individual mcx already accounts
    for). Verified against a brute-force truth table across many
    ``(n, constant)`` pairs before use — see this block's test file.
    """
    n = len(domain_qubits)
    path: List[Tuple[int, int]] = []
    for i in range(n - 1, -1, -1):
        bit = (constant >> i) & 1
        q = domain_qubits[i]
        if bit == 0:
            _apply_mcx_path(block, path + [(q, 1)], flag_qubit)
            path = path + [(q, 0)]
        else:
            path = path + [(q, 1)]
    _apply_mcx_path(block, path, flag_qubit)  # domain register == constant exactly


def _piecewise_theta(domain_size: int, breakpoints: List[int], slopes, intercepts) -> np.ndarray:
    bounds = [0, *breakpoints, domain_size]
    theta = np.zeros(domain_size)
    for i, (slope, intercept) in enumerate(zip(slopes, intercepts, strict=True)):
        lo, hi = bounds[i], bounds[i + 1]
        theta[lo:hi] = slope * np.arange(lo, hi) + intercept
    return theta


@prepares_known_state
class PiecewiseLinearStateBlock(SimpleBlock):
    def __init__(
        self,
        n_domain_qubits: int,
        breakpoints: List[int],
        slopes: List[float],
        intercepts: List[float],
        target_qubits: Optional[List[int]] = None,
        name: str = "PiecewiseLinear",
    ):
        r"""The piecewise-linear "payoff operator" of Woerner & Egger,
        *npj Quantum Inf.* **5**, 15 (2019): from :math:`|0\rangle^{\otimes
        n}|0\rangle`, put the domain register in uniform superposition and
        rotate a flag qubit by an angle linear in :math:`x` within each
        piece,

        .. math::

            |0\ldots0\rangle \;\mapsto\; 2^{-n/2}\sum_{x=0}^{2^n-1}
            |x\rangle\bigl(\cos(\theta(x)/2)|0\rangle
            + \sin(\theta(x)/2)|1\rangle\bigr),
            \qquad \theta_i(x) = \text{slopes}_i\, x + \text{intercepts}_i
            \text{ for } x \text{ in piece } i.

        The domain register is split into :math:`P` pieces by the interior
        breakpoints; :math:`P - 1` comparator ancillas (``x >= breakpoint``,
        a prefix-match walk) select the piece, the weighted-sum rotation
        (one ``Ry`` per domain qubit, angle :math:`\text{slope}_i 2^q`,
        plus one for the intercept — they accumulate additively since they
        all act on the same target about the same axis) lands on the flag,
        and the comparators are uncomputed.  The flag's :math:`|1\rangle`
        amplitude is :math:`\sin(\theta(x)/2)`, which is what amplitude
        estimation reads out: with this block as the state-preparation
        operator, the estimated probability is :math:`2^{-n}\sum_x
        \sin^2(\theta(x)/2)`.

        The block is deterministic and control-safe — ``state_qubits`` are
        the domain qubits plus the flag (block qubit ``self.flag_qubit`` =
        ``n + P - 1``, i.e. *after* the comparators; in ``target_statevector``
        it is bit ``n`` because ``state_qubits`` skips them), the comparators
        are the ancillas and return to :math:`|0\rangle` with probability 1,
        so ``ancilla_postselection`` is ``None``.  Whoever wants the
        *postselected* state :math:`\propto \sum_x \sin(\theta(x)/2)|x\rangle`
        on the domain applies ``PostSelection({self.flag_qubit: 1})`` to the
        result; the block itself never postselects.  The angle is linear in
        :math:`x`, not the amplitude or the probability: callers wanting
        :math:`\sin^2(\theta/2) \approx f(x)` choose slopes/intercepts under
        the small-angle convention of the amplitude-estimation literature
        themselves — this block is exact for whatever piecewise-linear
        :math:`\theta` it is given.

        **Cost (measured, ``qx.clifford_t_rz_gateset()`` CNOTs):** n=3, P=1:
        6; n=3, P=2: 100; n=4, P=3: 630; n=6, P=4: 5 970; n=8, P=4:
        13 000–23 000 depending on the breakpoints' bit patterns (20 080 for
        ``[50, 120, 200]``, the case pinned in ``test_state_prep_cost.py``).
        Every rotation is controlled on all :math:`P - 1` comparator bits
        plus a domain bit, and the prefix-match comparator is
        :math:`O(n^2)` Toffolis per breakpoint.  Woerner & Egger's per-piece
        *delta* form (rotate by :math:`\text{slope}_i - \text{slope}_{i-1}`
        controlled on comparator :math:`i` alone — two controls per rotation
        regardless of :math:`P`, ≈ 700 CNOTs at n=8, P=4) and a ripple-carry
        comparator are declared follow-ups.

        Args:
            n_domain_qubits: width of the domain register,
                :math:`x \in [0, 2^{n})`.
            breakpoints: :math:`P - 1` interior breakpoints, integers,
                strictly ascending, each in :math:`(0, 2^n)` — splits the
                domain into :math:`P` pieces.
            slopes, intercepts: length :math:`P`, the per-piece angle
                coefficients :math:`\theta_i(x) = \text{slopes}_i x +
                \text{intercepts}_i` (radians).
            target_qubits, name: standard Block kwargs.
        """
        domain_size = 2**n_domain_qubits
        for b in breakpoints:
            if isinstance(b, bool) or not isinstance(b, (int, np.integer)):
                raise ValueError(f"breakpoints must be integers, got {b!r}.")
        breakpoints = [int(b) for b in breakpoints]
        if not breakpoints == sorted(set(breakpoints)):
            raise ValueError("breakpoints must be strictly ascending.")
        if breakpoints and not (0 < breakpoints[0] and breakpoints[-1] < domain_size):
            raise ValueError(f"breakpoints must lie strictly inside (0, {domain_size}).")
        n_pieces = len(breakpoints) + 1
        if len(slopes) != n_pieces or len(intercepts) != n_pieces:
            raise ValueError(
                f"slopes/intercepts must have length {n_pieces} (= len(breakpoints) + 1)."
            )

        self.n_domain_qubits = n_domain_qubits
        self.breakpoints = breakpoints
        self.slopes = list(slopes)
        self.intercepts = list(intercepts)
        self.n_comparator_qubits = len(breakpoints)
        self.flag_qubit = n_domain_qubits + self.n_comparator_qubits

        theta = _piecewise_theta(domain_size, self.breakpoints, self.slopes, self.intercepts)
        # LSB column on (domain..., flag): index x + 2^n f; unit norm by construction.
        self._target = np.concatenate([np.cos(theta / 2), np.sin(theta / 2)]).astype(complex)
        self._target /= np.sqrt(domain_size)

        super().__init__(
            n_domain_qubits + self.n_comparator_qubits + 1,
            target_qubits=target_qubits,
            name=name,
        )

    @property
    def state_qubits(self) -> Tuple[int, ...]:
        return (*range(self.n_domain_qubits), self.flag_qubit)

    @property
    def ancilla_postselection(self) -> Optional[PostSelection]:
        return None

    def build_vanilla(self) -> None:
        domain_qubits = list(range(self.n_domain_qubits))
        comparator_qubits = list(range(self.n_domain_qubits, self.flag_qubit))

        self.h(domain_qubits)

        for breakpoint_value, comparator_qubit in zip(
            self.breakpoints, comparator_qubits, strict=True
        ):
            _apply_comparator(self, domain_qubits, breakpoint_value, comparator_qubit)

        for i in range(len(self.slopes)):
            piece_path = [(q, 1 if j < i else 0) for j, q in enumerate(comparator_qubits)]
            if abs(self.intercepts[i]) > 1e-15:
                _apply_mc_ry(self, piece_path, self.flag_qubit, self.intercepts[i])
            for bit_position, domain_qubit in enumerate(domain_qubits):
                angle = self.slopes[i] * (1 << bit_position)
                if abs(angle) > 1e-15:
                    _apply_mc_ry(self, [*piece_path, (domain_qubit, 1)], self.flag_qubit, angle)

        for breakpoint_value, comparator_qubit in zip(
            self.breakpoints, comparator_qubits, strict=True
        ):
            _apply_comparator(self, domain_qubits, breakpoint_value, comparator_qubit)

    def target_statevector(self) -> np.ndarray:
        """``2^{-n/2} Σₓ |x⟩(cos(θ(x)/2)|0⟩ + sin(θ(x)/2)|1⟩)`` on
        ``(domain..., flag)``, LSB: index ``x + 2^n·f``."""
        return self._target
