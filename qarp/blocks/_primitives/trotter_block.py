from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sympy import Symbol

import qarpx as qx
from qarp.blocks._block import SimpleBlock, _sorted_symbols
from qarp.operators import FullyCommuting, GroupingStrategy, QubitOperator

PauliTuple = Tuple[str, ...]  # ('X', 'I', 'Z', ...) — len == n_qubits

# ── Yoshida-style higher-order weight tables ────────────────────────────
#
# Used by ``TrotterBlock(composition="yoshida", order ∈ {6, 8})``.  Weights
# are literature constants; the symplectic Yoshida constraint
# ``Σ w_j = 1`` fixes ``w[0]`` once the others are known.
#
# References:
#   * Yoshida, "Construction of higher order symplectic integrators",
#     Phys. Lett. A 150 (1990) 262-268.  doi:10.1016/0375-9601(90)90092-3.
#     6-th order weights from Table 1, solution A.
#   * Morales et al., "Greatly improved higher-order product formulae for
#     quantum simulation", Quantum Info. & Comp. 25 (2025), arXiv:2210.15817.
#     8-th order weights from Table I, right column.

_YOSHIDA_W6 = np.asarray([0.0, -0.117767998417887e1, 0.235573213359, 0.784513610477])
_YOSHIDA_W8 = np.asarray(
    [
        0.0,
        0.10467636532245895252340732579853,
        -0.57896999331780988041471955125778,
        0.57503350160061785946141563279891,
        0.12231011868707029786561397542663,
        0.27793149999039524816733903301747,
        -0.37349605088056728482635987352576,
        0.11575566589480463220616543972403,
        0.1464645610975800618712569230326,
        -0.39443578322284085764474498594073,
        0.44370228726021218923197141183196,
    ]
)


def _yoshida_resolve_weights(order: int) -> np.ndarray:
    if order == 6:
        w = _YOSHIDA_W6.copy()
    elif order == 8:
        w = _YOSHIDA_W8.copy()
    else:
        raise ValueError("Yoshida composition is defined only for order 6 and 8")
    # Yoshida symplectic constraint: Σ w_j = 1 (the chain represents one
    # full t-step).  Resolve w[0] from this once the others are pinned.
    w[0] = 1.0 - 2.0 * np.sum(w[1:])
    return w


def _pauli_str(t: PauliTuple) -> str:
    """Pauli tuple → concatenated string accepted by ``parse_pauli_string`` (e.g. ``('X','I','Z') → 'XIZ'``)."""
    return "".join(t)


def _yoshida_s2_commuting(
    groups: List[Tuple[List[PauliTuple], List[float]]],
    weight: float,
) -> List[Tuple[List[PauliTuple], List[float]]]:
    """Strang split ``S₂(weight·t)`` over commuting-set-grouped Pauli terms.

    The pre/post halves emit a half-exponential (Pauli-exp angle = ``c·t``);
    the middle group emits the full exponential (angle = ``2·c·t``).  ``weight``
    scales every coefficient — the chain consumer multiplies by ``time/steps``
    at gate-emission time to materialise the actual Rz angle.
    """
    if not groups:
        return []
    pre = [(paulis, [c * weight for c in coeffs]) for paulis, coeffs in groups[:-1]]
    mid = (groups[-1][0], [2.0 * c * weight for c in groups[-1][1]])
    post = list(reversed(pre))
    return pre + [mid] + post


def _yoshida_iterative_build_commuting(
    groups: List[Tuple[List[PauliTuple], List[float]]],
    order: int,
) -> List[Tuple[List[PauliTuple], List[float]]]:
    """Yoshida-weighted chain ``∏_j S₂(w_j · t)`` for the grouped path."""
    w = _yoshida_resolve_weights(order)
    chain = _yoshida_s2_commuting(groups, float(w[0]))
    layers = [_yoshida_s2_commuting(groups, float(w_j)) for w_j in w[1:]]
    for layer in layers:
        chain = layer + chain
    for layer in layers:
        chain = chain + layer
    return chain


def _pauli_tuple_from_term(term, n_qubits: int) -> PauliTuple:
    """Convert an OpenFermion ``term`` (tuple of (qubit_idx, pauli_letter))
    into a length-``n_qubits`` tuple of Pauli letters with 'I' for unused
    qubits.
    """
    paulis = ["I"] * n_qubits
    for q, p in term:
        paulis[q] = p
    return tuple(paulis)


def _coerce_coeff(coeff, imaginary: bool) -> float:
    """Coefficient → real ``float`` according to the imaginary-flag convention."""
    if isinstance(coeff, complex):
        return float(coeff.imag if imaginary else coeff.real)
    return float(coeff)


def _pauli_tuple_to_dict(p: PauliTuple) -> Dict[int, str]:
    """Dense letter tuple → sparse ``{qubit: letter}`` (grouping-module layout)."""
    return {q: letter for q, letter in enumerate(p) if letter != "I"}


def list_commuting_paulis(
    operators, n_qubits: int, imaginary: bool, grouping: Optional[GroupingStrategy] = None
) -> List[Tuple[List[PauliTuple], List[float]]]:
    """Partition every Pauli term in ``operators`` into commuting groups via
    ``grouping`` (``None`` → general commutation).  Returns a list of
    ``(group_paulis, group_coeffs)`` where each Pauli is a length-``n_qubits``
    tuple of letters.

    General (not qubit-wise) commutation is the correct default for
    exponentiation: it keeps generally-commuting pairs like the JW hopping
    ``X_pX_q``/``Y_pY_q`` in one group, preserving particle-number
    conservation at finite Trotter steps (QWC scatters them and leaks
    Hamming weight).
    """
    if grouping is None:
        grouping = FullyCommuting()
    paulis: List[PauliTuple] = []
    coeffs: List[float] = []
    for operator in operators:
        for term, coeff in operator.terms.items():
            paulis.append(_pauli_tuple_from_term(term, n_qubits))
            coeffs.append(_coerce_coeff(coeff, imaginary))

    groups_idx = grouping.group([_pauli_tuple_to_dict(p) for p in paulis], n_qubits)
    return [([paulis[i] for i in grp], [coeffs[i] for i in grp]) for grp in groups_idx]


class TrotterBlock(SimpleBlock):
    def __init__(
        self,
        n_qubits: int,
        operator: Optional[QubitOperator],
        steps: int = 1,
        time: Optional[Union[float, Symbol]] = None,
        order: int = 1,
        imaginary: bool = False,
        composition: Literal["suzuki", "yoshida"] = "suzuki",
        grouping: Optional[GroupingStrategy] = None,
        target_qubits: Optional[List[int]] = None,
        name: Optional[str] = None,
    ):
        r"""Trotterised circuit for exp(-iHt) where H is a Qubit operator.

        Terms are partitioned via :func:`list_commuting_paulis` and each group
        is emitted as a single ``commuting_pauli_set_exp`` — one shared
        basis-change Clifford per group rather than per term.

        Args:
            n_qubits: Number of qubits the operator acts on.
            operator: A QubitOperator object.
            steps: Number of Trotter steps.
            time: Trotter evolution time (float or sympy Symbol).
            order: Trotter expansion order (1 or any even integer for
                ``composition="suzuki"``; one of {6, 8} for
                ``composition="yoshida"``).
            imaginary: Use imaginary part of coefficients.
            grouping: Term-partitioning strategy; ``None`` → ``FullyCommuting()``
                (general commutation).  ``NoGrouping()`` gives one term per
                group — the termwise circuit.  The strategy is
                physics-relevant: qubit-wise grouping separates
                generally-commuting pairs (e.g. JW hopping ``XX``/``YY``) and
                breaks particle-number conservation at finite steps.  A strategy
                emitting a non-commuting group is rejected by
                ``commuting_pauli_set_exp``.
            composition: Product-formula composition method.

                * ``"suzuki"`` (default): Suzuki 5-term recursion
                  ``S_p(t) = S_{p-2}(u·t)² · S_{p-2}((1-4u)·t) · S_{p-2}(u·t)²``
                  with ``u = 1/(4 - 4^(1/(p-1)))``.  Works at order 1, 2, 4,
                  6, 8, 10, ... at growing cost (5^((p-2)/2)× the base count).

                * ``"yoshida"``: literature-tuned weight chain
                  ``∏_j S₂(w_j · t)`` from Yoshida 1990 (order 6) and
                  Morales 2025 (order 8).  Smaller leading-error constant than
                  Suzuki at the same order; only defined for orders 6 and 8.

            target_qubits, name: see Block.
        """
        if not isinstance(order, int) or order < 1 or (order > 2 and order % 2 != 0):
            raise ValueError(f"order must be 1 or an even integer >= 2; got order={order}.")
        if composition not in ("suzuki", "yoshida"):
            raise ValueError(f"composition must be 'suzuki' or 'yoshida'; got {composition!r}.")
        if composition == "yoshida" and order not in (6, 8):
            raise ValueError(
                f"composition='yoshida' is only defined for order 6 or 8 (literature "
                f"weights); got order={order}.  Use composition='suzuki' for other orders."
            )
        if composition == "suzuki" and order >= 8:
            import warnings

            sub_block_factor = 5 ** ((order - 2) // 2)
            warnings.warn(
                f"Trotter order={order} (composition='suzuki') unfolds the 5-term "
                f"recursion into ~{sub_block_factor}× the base Pauli-exp count per "
                "Trotter step. Consider composition='yoshida' at order 6 or 8 for a "
                "shorter circuit with smaller leading-error constant.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.operator = operator
        self.steps = steps
        self.time = Symbol("t") if time is None else time
        self.order = order
        self.grouping = grouping if grouping is not None else FullyCommuting()
        self.imaginary = imaginary
        self.composition = composition
        if name is None:
            name = f"Trotter (order={self.order}, {composition})"

        super().__init__(n_qubits, target_qubits, name=name)

        if isinstance(self.time, Symbol):
            self.symbols = [self.time]
            self._time_symbol = self.time
        else:
            self.symbols = []
            self._time_symbol = None

    def set_time(self, time_value: float) -> "SimpleBlock":
        """Substitute the symbolic time with a concrete value (radians)."""
        if self._time_symbol is None:
            raise ValueError("Cannot set time: block was not built with a symbolic time parameter")
        return self.set_symbols({self._time_symbol: time_value})

    # ── Build ───────────────────────────────────────────────────────────────

    def build_vanilla(self) -> None:
        if self.operator is None:
            raise AttributeError(
                "The operator attribute must be set before calling .build(). "
                "Either set the attribute directly, or pass an operator in the constructor."
            )

        operator_terms = self.operator.terms
        if () in operator_terms:
            op = QubitOperator()
            op.terms = {k: v for k, v in operator_terms.items() if k != ()}
            single_terms = list(op.get_operators())
            constant = operator_terms.get((), 0)
        else:
            single_terms = list(self.operator.get_operators())
            constant = 0

        if self.composition == "yoshida":
            self._emit_yoshida_chain(single_terms)
        else:
            self._emit_suzuki_sequences(single_terms)

        # Constant term contributes a global phase exp(-i · constant · time)
        # (matching the exp(-iHt) convention of the Pauli-exp sequences).
        constant_coeff = _coerce_coeff(constant, self.imaginary)
        if constant_coeff != 0.0:
            phase_angle = constant_coeff * self._time_value()
            self.gphase(-phase_angle)

    def _synthesise_group(self, pauli_strs, params):
        """One synthesis-boundary crossing: synthesize into a scratch block and
        capture the commands for replay.  Steps repeat identical sequences, so
        re-entering the C++ synthesis per step would redo the O(k²·n²)
        commutation/elimination work for byte-identical output."""
        scratch = qx.SimpleBlock(self.n_qubits, "trotter_synth")
        scratch.commuting_pauli_set_exp(pauli_strs, params)
        return list(scratch.commands())

    def _replay_steps(self, step_cmds) -> None:
        """Append ``step_cmds`` ``self.steps`` times to the command buffer."""
        all_cmds = list(self.commands())
        for _ in range(self.steps):
            all_cmds.extend(step_cmds)
        self.set_commands(all_cmds)

    def _emit_suzuki_sequences(self, single_terms) -> None:
        """Suzuki 5-term recursion path — one synthesis per distinct group,
        replayed per step."""
        terms = list_commuting_paulis(single_terms, self.n_qubits, self.imaginary, self.grouping)
        sequences = self._make_sequences(terms, self._time_value(), self.order, self.steps)
        step_cmds: list = []
        for seq in sequences:
            step_cmds.extend(
                self._synthesise_group(
                    [_pauli_str(p) for p, _ in seq], [self._as_param(a) for _, a in seq]
                )
            )
        self._replay_steps(step_cmds)

    def _emit_yoshida_chain(self, single_terms) -> None:
        """Yoshida-weighted chain ``∏_j S₂(w_j · t)``.  Yields a shorter circuit
        than Suzuki at orders 6/8 with smaller leading-error constant
        (literature-tuned weights).
        """
        groups = list_commuting_paulis(single_terms, self.n_qubits, self.imaginary, self.grouping)
        chain = _yoshida_iterative_build_commuting(groups, self.order)
        per_step = self._time_value() / self.steps
        step_cmds: list = []
        for paulis_in_group, coeffs in chain:
            step_cmds.extend(
                self._synthesise_group(
                    [_pauli_str(p) for p in paulis_in_group],
                    [self._as_param(float(c) * per_step) for c in coeffs],
                )
            )
        self._replay_steps(step_cmds)

    # ── Sequence generation ─────────────────────────────────────────────────

    @staticmethod
    def _make_sequences(terms, time, order: int, steps: int):
        """Per-step Pauli-exp sequence list at the requested order.

        Pure helper — no ``self`` so the higher-order Suzuki recursion can
        re-enter with scaled times without instantiating temporary blocks.

        ``terms`` is the ``(paulis_in_group, coeffs_in_group)`` list from
        :func:`list_commuting_paulis`.  Returns a list of groups, each a list
        of ``(pauli_tuple, angle)`` ready for ``commuting_pauli_set_exp``.
        ``angle`` carries the Rz radians factor ``2·coeff·time/steps`` baked
        in; with a symbolic ``time`` it is a sympy monomial ``k·t`` that
        ``Block._as_param`` materialises.
        """
        # Base layer: order-1 unsymmetrised emission.
        angle_factor = 2.0 * time / steps
        flat = [
            [(paulis, coeffs[i] * angle_factor) for i, paulis in enumerate(cps)]
            for cps, coeffs in terms
        ]

        if order == 1:
            return flat
        if order == 2:
            return TrotterBlock._symmetrize_sequence(flat)

        # Suzuki 5-term recursion: S_p(t) = S_{p-2}(u·t)² · S_{p-2}((1−4u)·t)
        # · S_{p-2}(u·t)² with u = 1/(4 − 4^(1/(p−1))).  Validated for
        # ``order ∈ {2, 4, 6, 8}`` in ``test_trotter_block.py`` (convergence
        # rate + explicit S₆ reference).
        reduction = 1.0 / (4.0 - (4.0 ** (1.0 / (order - 1))))
        outer = TrotterBlock._make_sequences(terms, time * reduction, order - 2, steps)
        inner = TrotterBlock._make_sequences(
            terms, time * (1.0 - 4.0 * reduction), order - 2, steps
        )
        return outer + outer + inner + outer + outer

    @staticmethod
    def _symmetrize_sequence(flat):
        """Second-order symmetrisation: halved pre-groups, full middle,
        halved reversed post-groups.  Emits S_2(t) = ∏_k exp(-i·t/2·H_k) ·
        exp(-i·t·H_last) · ∏_k exp(-i·t/2·H_k) so the leading-order
        commutator vanishes (Strang splitting).
        """
        if not flat:
            return []
        pre = [[(p, a / 2) for p, a in seq] for seq in flat[:-1]]
        mid = [flat[-1]]
        post = list(reversed(pre))
        return pre + mid + post

    def _time_value(self):
        """Return ``self.time`` ready for gate-angle arithmetic.

        Symbolic time stays a sympy Symbol (the gate methods promote it to
        a qx.Param).  Float/complex inputs are cast to a real float.
        """
        t = self.time
        if isinstance(t, complex):
            return float(t.real)
        return t


class TrotterAnsatzBlock(SimpleBlock):
    """Symbol-per-term Trotterised ansatz: ``∏_k exp(-i s_k Q_k)``.

    Each ``(s_k, Q_k)`` pair contributes one factor in the ansatz.  Per-term
    Pauli coefficients absorb into the symbol's effective angle, so the
    ansatz parameters ``s_k`` are the only free parameters at run time.

    Radians convention: a symbol value ``s_k`` contributes ``exp(+i s_k c
    time P)`` per Pauli term ``P`` with coefficient ``c`` — emitted as a
    ``commuting_pauli_set_exp`` angle of ``-2·c·time/steps · s_k`` radians
    (the builder realises ``exp(-i/2 · Σ angle·P)``).
    """

    def __init__(
        self,
        n_qubits: int,
        qubit_exponents: List[QubitOperator],
        symbols: List[Symbol],
        steps: int = 1,
        time: Union[float, Symbol] = 1.0,
        order: int = 1,
        imaginary: bool = False,
        grouping: Optional[GroupingStrategy] = None,
        target_qubits=None,
        name="TrotterAnsatz",
    ):
        if not isinstance(order, int) or order < 1 or (order > 2 and order % 2 != 0):
            raise ValueError(f"order must be 1 or an even integer >= 2; got order={order}.")
        if order >= 8:
            import warnings

            sub_block_factor = 5 ** ((order - 2) // 2)
            warnings.warn(
                f"TrotterAnsatz order={order} unfolds the Suzuki 5-term recursion "
                f"into ~{sub_block_factor}× the base Pauli-exp count per Trotter "
                "step. Expect a large gate count.",
                RuntimeWarning,
                stacklevel=2,
            )

        super().__init__(
            n_qubits=n_qubits,
            target_qubits=target_qubits,
            name=name,
        )
        self.qubit_exponents = qubit_exponents
        # Pairing contract: one symbol per exponent — silent truncation would
        # advertise all symbols in .symbols while dropping exponents.
        self.symbol_qop_pairs = list(zip(symbols, qubit_exponents, strict=True))
        self.symbols = _sorted_symbols(symbols)
        self.steps = steps
        self.time = time
        self.order = order
        self.grouping = grouping if grouping is not None else FullyCommuting()
        self.imaginary = imaginary

        if isinstance(time, Symbol):
            if time not in self.symbols:
                self.symbols = (*self.symbols, time)
            self._time_symbol = time
        else:
            self._time_symbol = None

    def set_time(self, time_value: float) -> "SimpleBlock":
        if self._time_symbol is None:
            raise ValueError("Cannot set time: block was not built with a symbolic time parameter")
        # Symbolic-time path: the C++ buffer is empty (build deferred because
        # qx.Param is single-variable).  Construct and build a fresh block at
        # the concrete time, preserving structural settings.
        if isinstance(self.time, Symbol):
            symbols_only = [s for s, _ in self.symbol_qop_pairs]
            new_block = TrotterAnsatzBlock(
                n_qubits=self.n_qubits,
                qubit_exponents=list(self.qubit_exponents),
                symbols=symbols_only,
                steps=self.steps,
                time=float(time_value),
                order=self.order,
                imaginary=self.imaginary,
                grouping=self.grouping,
                target_qubits=list(self.target_qubits) if self.target_qubits else None,
                name=self.name,
            )
            new_block.build()
            return new_block
        # Concrete-time block built with a symbolic time?  Not reachable today —
        # _time_symbol is only set when self.time is a Symbol — but defer to
        # the canonical pending-substitution path for safety.
        return self.set_symbols({self._time_symbol: time_value})

    def build_vanilla(self) -> None:
        if self.symbols is None:
            raise ValueError("Symbols must be provided for TrotterAnsatzBlock")

        # Symbolic-time path: qx.Param is single-variable, so we cannot bake
        # ``t · s_k`` into one Param at build time.  Leave the C++ buffer empty
        # — ``set_time(value)`` returns a fresh block with the substituted time
        # that builds normally.  The metadata (.symbols / ._time_symbol) stays
        # populated so callers can introspect.
        if isinstance(self.time, Symbol):
            return

        time_val = float(self.time.real) if isinstance(self.time, complex) else float(self.time)
        # Radians convention: a symbol value s emits U = exp(+i s c P) per term.
        # commuting_pauli_set_exp emits exp(-i/2 · Σ angle_k · P_k), so
        # angle = -2 s c · time / steps.
        per_step_factor = -2.0 * time_val / self.steps

        # A generator whose terms all vanish (zero coefficient, or an imaginary
        # one under imaginary=False) emits no rotation, so its symbol gates
        # nothing: publishing it would let parameter_map take a value the
        # circuit ignores.  symbol_qop_pairs keeps the full pairing — set_time()
        # rebuilds the ansatz from it.
        live = {sym for triples in self._terms_by_generator() for _, sym, _ in triples}
        dead = tuple(s for s in self.symbols if str(s) not in live)
        if dead:
            import warnings

            warnings.warn(
                f"{self.name}: the generator(s) for "
                f"{', '.join(str(s) for s in dead)} emit no rotation (every "
                "coefficient vanishes); dropping them from .symbols.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.symbols = tuple(s for s in self.symbols if str(s) in live)

        sequences = self._suzuki_recurse(self._sequences_grouped(), self.order)

        for _ in range(self.steps):
            for seq in sequences:
                self.commuting_pauli_set_exp(
                    [_pauli_str(p) for p, _, _ in seq],
                    [qx.Param.linear(per_step_factor * c, s) for _, s, c in seq],
                )

    # ── Sequence generation ────────────────────────────────────────────────

    def _terms_by_generator(self) -> List[List[Tuple[tuple, str, float]]]:
        """One ``(paulis, sym_name, c)`` list per (symbol, qop) pair.

        ``c`` is the real (or imaginary, per ``self.imaginary``) Pauli
        coefficient — pre-multiplication by ``per_step_factor`` is deferred to
        gate emission so symmetrisation can scale ``c`` cleanly.
        """
        out: List[List[Tuple[tuple, str, float]]] = []
        for symbol, qop in self.symbol_qop_pairs:
            sym_name = str(symbol)
            per_generator: List[Tuple[tuple, str, float]] = []
            for term, coeff in qop.terms.items():
                if isinstance(coeff, complex):
                    c = coeff.imag if self.imaginary else coeff.real
                else:
                    c = float(coeff)
                if c == 0.0 or not term:
                    # Zero coeff or identity-only term: no symbol-gated
                    # rotation to emit.
                    continue
                paulis = _pauli_tuple_from_term(term, self.n_qubits)
                per_generator.append((paulis, sym_name, float(c)))
            if per_generator:
                out.append(per_generator)
        return out

    def _sequences_grouped(self) -> List[List[Tuple[tuple, str, float]]]:
        """Commuting partition via the injected strategy, applied *within* each
        generator.

        The ansatz contract is the ordered product ``∏_k exp(θ_k G_k)``, so the
        generator order is meaningful and grouping must not permute across it —
        two generators generally have anticommuting Paulis, and merging them
        silently redefines the ansatz.  Within a generator the Pauli order is an
        arbitrary artefact of the operator, so grouping there is free.
        """
        out: List[List[Tuple[tuple, str, float]]] = []
        for triples in self._terms_by_generator():
            dicts = [_pauli_tuple_to_dict(p) for p, _, _ in triples]
            for grp in self.grouping.group(dicts, self.n_qubits):
                out.append([triples[i] for i in grp])
        return out

    @staticmethod
    def _symmetrize_sequence(flat):
        """Second-order symmetrisation: halved pre-groups, full middle, halved
        reversed post-groups.  Halving is applied to the per-term coefficient
        ``c``; ``per_step_factor`` is multiplied later at gate-emission time."""
        if not flat:
            return []
        pre = [[(p, s, c / 2) for p, s, c in seq] for seq in flat[:-1]]
        mid = [flat[-1]]
        post = list(reversed(pre))
        return pre + mid + post

    @staticmethod
    def _suzuki_recurse(flat, order: int):
        """Suzuki 5-term recursive composition for order ``p``.

        Mirrors ``TrotterBlock._make_sequences``' recursion structure, but
        threads the per-chunk time-scale factor into the symbol-bound Pauli
        coefficient ``c`` (instead of into a concrete ``time * reduction``
        substitution).  ``per_step_factor`` is multiplied in later at
        gate-emission time, so the recursion stays purely on coefficient
        arithmetic.  Order ``p`` is decomposed as ``S_p(t) = S_{p-2}(u·t)² ·
        S_{p-2}((1-4u)·t) · S_{p-2}(u·t)²`` with ``u = 1/(4 - 4^(1/(p-1)))``.
        Validated for ``order ∈ {4, 6}`` in
        ``tests/test_blocks/test_primitives/test_trotter_ansatz.py``.
        """
        if order == 1:
            return flat
        if order == 2:
            return TrotterAnsatzBlock._symmetrize_sequence(flat)
        if order % 2 != 0:
            raise ValueError(
                "Construction of Trotter circuits for odd orders greater than one is not defined."
            )
        reduction = 1.0 / (4.0 - (4.0 ** (1.0 / (order - 1))))
        sub = TrotterAnsatzBlock._suzuki_recurse(flat, order - 2)

        def scale_c(seqs, factor):
            return [[(p, s, c * factor) for p, s, c in seq] for seq in seqs]

        outer = scale_c(sub, reduction)
        inner = scale_c(sub, 1.0 - 4.0 * reduction)
        return outer + outer + inner + outer + outer

    def replace_symbols(self, new_parameters: Dict[Symbol, Symbol]) -> "SimpleBlock":  # type: ignore
        new_object = super().replace_symbols(new_parameters)
        if new_object.symbols and hasattr(new_object, "symbol_qop_pairs"):
            new_object.symbol_qop_pairs = [
                (new_parameters.get(s, s), qop) for s, qop in new_object.symbol_qop_pairs
            ]
        return new_object
