"""Problems and parameterized-circuit templates, shared by every stack.

A template op is a plain tuple; parameterized angles are
``("p", index, coefficient)`` meaning ``angle = coefficient * params[index]``.
Each adapter maps that onto its SDK's parameter object (sympy symbol, qiskit
Parameter, parametric gate slot); `bind` produces the concrete canonical op
list the numpy oracle executes.  Everything is seeded/deterministic.
"""

from benchmarks.statevector.inputs import _Rng, molecular_terms

SEED = 20260825
HEA_LAYERS = 3


# --- ansatz templates -------------------------------------------------------


def hea_template(n: int) -> tuple[list, int]:
    """Hardware-efficient ansatz: Ry layers + a CZ ring; one param per Ry.

    The entangler must be CZ, not CX: CZ is diagonal, so at zero angles the
    circuit only phases computational basis states and the prepared reference
    determinant keeps its occupation (E(0) = E_ref exactly).  A CX ring flips
    occupations and turns the reference into a different determinant — which
    is why chemistry HEAs (qarp's `HEABlock` included) entangle with CZ.
    CZ(a, b) = H(b) · CX(a, b) · H(b), staying inside the canonical gate set.
    """
    ops: list = []
    k = 0
    for _ in range(HEA_LAYERS):
        for q in range(n):
            ops.append(("ry", q, ("p", k, 1.0)))
            k += 1
        for q in range(n):
            a, b = q, (q + 1) % n
            ops.append(("h", b))
            ops.append(("cx", a, b))
            ops.append(("h", b))
    for q in range(n):
        ops.append(("ry", q, ("p", k, 1.0)))
        k += 1
    return ops, k


def qaoa_template(n: int, p_layers: int = 2) -> tuple[list, int]:
    """QAOA for MaxCut: params ordered [gamma_1, beta_1, gamma_2, beta_2, ...]."""
    edges = maxcut_edges(n)
    ops: list = []
    for q in range(n):
        ops.append(("h", q))
    for layer in range(p_layers):
        gamma, beta = 2 * layer, 2 * layer + 1
        for a, b in edges:
            # exp(+i gamma/2 Z_a Z_b) == rzz(-gamma), decomposed as cx rz cx.
            ops.append(("cx", a, b))
            ops.append(("rz", b, ("p", gamma, -1.0)))
            ops.append(("cx", a, b))
        for q in range(n):
            ops.append(("rx", q, ("p", beta, 2.0)))
    return ops, 2 * p_layers


def bind(template: list, params) -> list:
    """Template -> concrete canonical ops (what the numpy oracle executes)."""
    out = []
    for op in template:
        if len(op) == 3 and isinstance(op[2], tuple) and op[2][0] == "p":
            _, index, coefficient = op[2]
            out.append((op[0], op[1], coefficient * float(params[index])))
        else:
            out.append(op)
    return out


def initial_params(n_params: int, spread: float = 0.25) -> list:
    """Small deterministic start near (not at) zero — identical everywhere."""
    rng = _Rng(SEED + n_params)
    return [rng.uniform(-spread, spread) for _ in range(n_params)]


# --- problems ---------------------------------------------------------------


def maxcut_edges(n: int) -> list[tuple[int, int]]:
    """Deterministic 3-regular graph (n even): ring + antipodal chords."""
    if n % 2:
        raise ValueError("maxcut graph needs even n")
    edges = [(q, (q + 1) % n) for q in range(n)]
    edges += [(q, q + n // 2) for q in range(n // 2)]
    return sorted(tuple(sorted(e)) for e in edges)


def maxcut_terms(n: int) -> list:
    """Minimizing this energy maximizes the cut: E = sum (Z_i Z_j - 1) / 2."""
    edges = maxcut_edges(n)
    terms: list = [((), complex(-0.5 * len(edges)))]
    for a, b in edges:
        terms.append((((a, "Z"), (b, "Z")), complex(0.5)))
    return terms


def problem(family: str, size: int) -> dict:
    """Everything a stack needs: template, terms, x0, budget, references."""
    from benchmarks.algorithms import spec

    if family == "vqe":
        import math

        terms, payload = molecular_terms(size)
        template, n_params = hea_template(size)
        # HF reference state first (occupied spin-orbitals are the low JW
        # indices): without it the ansatz starts in the vacuum sector and a
        # budgeted optimizer never finds the electrons.  rx(pi) = -iX; the
        # global phase cancels in every expectation value.
        occupied = payload["spatial"]["n_electrons"]
        template = [("rx", q, math.pi) for q in range(occupied)] + template
        # Tight start: at many parameters a wide random start scrambles the HF
        # reference and COBYLA converges in a basin far above HF (measured:
        # LiH landed 1.7 Ha above HF at spread 0.25).  0.02 keeps the energy
        # surface anchored at HF so the optimizer descends below it.
        spread = 0.02
        rhobeg = 0.05
        reference = {"fci_energy": payload["fci_energy"], "hf_energy": payload["hf_energy"]}
    elif family == "qaoa":
        terms = maxcut_terms(size)
        template, n_params = qaoa_template(size)
        spread = 0.25
        rhobeg = 0.4
        reference = {"max_cut": brute_force_maxcut(size)}
    else:
        raise ValueError(f"unknown family {family}")
    return {
        "n_qubits": size,
        "template": template,
        "n_params": n_params,
        "terms": terms,
        "x0": initial_params(n_params, spread),
        "budget": spec.eval_budget(n_params),
        "rhobeg": rhobeg,
        "reference": reference,
    }


def brute_force_maxcut(n: int) -> int:
    """Exact optimum over all 2^n cuts — the §18 anchor for the QAOA rows."""
    edges = maxcut_edges(n)
    best = 0
    for assignment in range(1 << n):
        cut = sum(1 for a, b in edges if ((assignment >> a) ^ (assignment >> b)) & 1)
        best = max(best, cut)
    return best
