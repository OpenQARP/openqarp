"""One timing kernel per workload family, shared by every stack.

Each factory takes a stack adapter module (_qx / _of / _qk / _pl) and returns
(DESCRIPTION, bench).  The measurement protocol lives here once; what differs
between stacks is the adapter idiom.  The modules under workloads/ are thin
shims over FACTORIES, one per (family, stack), discovered by filename.
"""

from benchmarks.common import peak_mem_mib, timed
from benchmarks.operators import checks, inputs

SEED_A = inputs.SEED + 11
SEED_B = inputs.SEED + 12


def construct_string(stack):
    description = "QubitOperator(term_string, coeff) constructor loop"

    def bench(size):
        prepared = stack.prepare_terms(inputs.pauli_term_specs(size))
        singles, t_run = timed(lambda: [stack.construct_one(item) for item in prepared])

        def all_entries():
            for op in singles:
                yield from stack.entries(op)

        return {"t_run": t_run, "check": checks.summarize(all_entries())}

    return description, bench


def accumulate(stack):
    description = "H += term accumulation with ~50% key collisions"

    def bench(size):
        prepared = stack.prepare_terms(inputs.colliding_term_specs(size))
        op, t_run = timed(lambda: stack.accumulate(prepared, inputs.N_QUBITS))
        return {"t_run": t_run, "check": checks.summarize(stack.entries(op))}

    return description, bench


def op_product(stack):
    description = "A·B for T-term random operators (n=20, weight ≤ 4)"

    def bench(size):
        a, t_a = timed(
            lambda: stack.qubit_op(inputs.pauli_term_specs(size, seed=SEED_A), inputs.N_QUBITS)
        )
        b, t_b = timed(
            lambda: stack.qubit_op(inputs.pauli_term_specs(size, seed=SEED_B), inputs.N_QUBITS)
        )
        out, t_run = timed(lambda: stack.product(a, b))
        return {"t_build": t_a + t_b, "t_run": t_run, "check": checks.summarize(stack.entries(out))}

    return description, bench


def commutator(stack):
    description = "A·B − B·A for T-term random operators (n=20, weight ≤ 4)"

    def bench(size):
        a, t_a = timed(
            lambda: stack.qubit_op(inputs.pauli_term_specs(size, seed=SEED_A), inputs.N_QUBITS)
        )
        b, t_b = timed(
            lambda: stack.qubit_op(inputs.pauli_term_specs(size, seed=SEED_B), inputs.N_QUBITS)
        )
        out, t_run = timed(lambda: stack.commutator(a, b))
        return {"t_build": t_a + t_b, "t_run": t_run, "check": checks.summarize(stack.entries(out))}

    return description, bench


def hermitian_conjugated(stack):
    description = "hermitian_conjugated of a T-term operator (×5)"

    def bench(size):
        op, t_build = timed(lambda: stack.qubit_op(inputs.pauli_term_specs(size), inputs.N_QUBITS))

        def kernel():
            for _ in range(5):
                out = stack.hc(op)
            return out

        out, t_run = timed(kernel)
        return {"t_build": t_build, "t_run": t_run, "check": checks.summarize(stack.entries(out))}

    return description, bench


def jw_molecular(stack):
    description = "Jordan-Wigner of a random 1+2-body FermionOperator (n modes)"

    def bench(size):
        fop, t_build = timed(lambda: stack.fermion_op(inputs.ladder_term_specs(size)))
        qop, t_run = timed(lambda: stack.jordan_wigner(fop))
        return {"t_build": t_build, "t_run": t_run, "check": checks.summarize(stack.entries(qop))}

    return description, bench


def bk_molecular(stack):
    description = "Bravyi-Kitaev of the same random FermionOperator family (n modes)"

    def bench(size):
        fop, t_build = timed(lambda: stack.fermion_op(inputs.ladder_term_specs(size)))
        qop, t_run = timed(lambda: stack.bravyi_kitaev(fop, size))
        return {"t_build": t_build, "t_run": t_run, "check": checks.summarize(stack.entries(qop))}

    return description, bench


def parity_encode(stack):
    description = "parity mapping of the same random FermionOperator family (n modes)"

    def bench(size):
        fop, t_build = timed(lambda: stack.fermion_op(inputs.ladder_term_specs(size)))
        qop, t_run = timed(lambda: stack.parity_transform(fop, size))
        return {"t_build": t_build, "t_run": t_run, "check": checks.summarize(stack.entries(qop))}

    return description, bench


def sparse_construct(stack):
    description = "sparse-matrix realization of a 1000-term operator (n qubits)"

    def bench(size):
        specs = inputs.pauli_term_specs(1000, n_qubits=size)
        op, t_build = timed(lambda: stack.qubit_op(specs, size))
        matrix, t_run = timed(lambda: stack.to_sparse(op, size))
        return {"t_build": t_build, "t_run": t_run, "check": checks.summarize_sparse(matrix)}

    return description, bench


def symbolic_algebra(stack):
    description = "H·H and H−H† with symbolic coefficients"

    def bench(size):
        h, t_build = timed(lambda: stack.symbolic_op(inputs.symbolic_term_specs(size)))

        def kernel():
            return stack.product(h, h), h - stack.hc(h)

        (h2, diff), t_run = timed(kernel)
        check = {f"h2_{k}": v for k, v in checks.summarize(stack.substituted_entries(h2)).items()}
        check.update(
            {f"d_{k}": v for k, v in checks.summarize(stack.substituted_entries(diff)).items()}
        )
        return {"t_build": t_build, "t_run": t_run, "check": check}

    return description, bench


def memory_hold(stack):
    description = "build + hold a T-term operator (peak RSS)"

    def bench(size):
        op, t_run = timed(lambda: stack.hold(inputs.pauli_term_spec_stream(size), 24))
        # Captured before the check pass so summarize() cannot inflate the
        # row's own measurement.
        meta = {"mem_peak_mib": peak_mem_mib()}
        return {"t_run": t_run, "check": checks.summarize(stack.entries(op)), "meta": meta}

    return description, bench


FACTORIES = {
    "construct_string": construct_string,
    "accumulate": accumulate,
    "op_product": op_product,
    "commutator": commutator,
    "hermitian_conjugated": hermitian_conjugated,
    "jw_molecular": jw_molecular,
    "bk_molecular": bk_molecular,
    "parity_encode": parity_encode,
    "sparse_construct": sparse_construct,
    "symbolic_algebra": symbolic_algebra,
    "memory_hold": memory_hold,
}
