"""count_resources: exact counts, pseudo-op exclusion, None-vs-0 semantics."""

import qarpx as qx
from qarp.blocks import SimpleBlock
from qarp.resources import Provenance, Stage, count_resources
from qarp.resources._counting import _PSEUDO

LOGICAL = Provenance(stage=Stage.LOGICAL)


def _flat(block: SimpleBlock) -> list:
    block.build()
    return block.flatten()


def test_clifford_t_circuit_exact_counts():
    b = SimpleBlock(3)
    b.h(0)
    b.cx(0, 1)
    b.t(2)
    b.measure([(0, 0), (1, 1), (2, 2)])
    v = count_resources(_flat(b), provenance=LOGICAL)
    assert v.n_qubits == 3
    assert v.depth == 3
    assert v.n_gates == 3
    assert v.n_1q == 2
    assert v.n_2q == 1
    assert v.n_measurements == 3
    assert v.n_resets == 0
    assert v.t_count == 1  # no parametric unitary: T is countable
    assert v.swap_count is None  # not routed — None, not 0
    assert v.op_histogram == {"H": 1, "CX": 1, "T": 1, "Measure": 3}
    assert v.t_count_modeled is None and v.extras == {}


def test_parametric_gate_blocks_counted_t():
    b = SimpleBlock(1)
    b.t(0)
    b.rz(0, 0.3)
    v = count_resources(_flat(b), provenance=LOGICAL)
    assert v.t_count is None  # an Rz remains: a T count would be a lie
    assert v.op_histogram["T"] == 1  # the raw histogram still shows it


def test_wide_mcz_blocks_counted_t():
    # Width >= 3 MCZ has no T-free exact form (an exact Clifford+T CCZ costs
    # 7 T, Amy et al. 2013), so t_count = 0 flagged exact would be a lie.
    b = SimpleBlock(4)
    b.t(0)
    b.mcz(0, 1, 2, 3)
    v = count_resources(_flat(b), provenance=LOGICAL)
    assert v.t_count is None
    assert v.n_gates == 2  # counted fields stay stream-exact
    assert v.op_histogram["MCZ"] == 1


def test_width_2_mcz_keeps_exact_t_count():
    b = SimpleBlock(2)
    b.t(0)
    b.mcz(0, 1)  # = CZ: Clifford, T-free
    v = count_resources(_flat(b), provenance=LOGICAL)
    assert v.t_count == 1
    assert v.n_2q == 1


def test_pseudo_ops_excluded_from_gates_and_histogram():
    b = SimpleBlock(2)
    b.h(0)
    b.gphase(0.5)
    cmds = list(_flat(b))
    cmds.append(qx.Command(qx.GateType.Barrier, 0))
    v = count_resources(cmds, provenance=LOGICAL)
    assert v.n_gates == 1
    assert "GPhase" not in v.op_histogram and "Barrier" not in v.op_histogram
    # GPhase is resource-free: its parameter must not block t_count
    assert v.t_count == 0


def test_user_authored_swap_is_2q_but_not_swap_count():
    b = SimpleBlock(2)
    b.swap(0, 1)
    v = count_resources(_flat(b), provenance=LOGICAL)
    assert v.n_2q == 1
    assert v.op_histogram == {"SWAP": 1}
    assert v.swap_count is None  # swap_count means router overhead only


def test_routed_stage_reports_swap_count_zero_when_absent():
    b = SimpleBlock(2)
    b.cx(0, 1)
    v = count_resources(_flat(b), provenance=Provenance(stage=Stage.ROUTED))
    assert v.swap_count == 0  # at ROUTED, zero is a real answer


def test_n_qubits_override_covers_idle_qubits():
    b = SimpleBlock(3)
    b.h(0)  # qubits 1, 2 idle: the DAG infers width 1
    cmds = _flat(b)
    assert count_resources(cmds, provenance=LOGICAL).n_qubits == 1
    assert count_resources(cmds, provenance=LOGICAL, n_qubits=3).n_qubits == 3


def test_accepts_circuit_dag_input():
    b = SimpleBlock(2)
    b.h(0)
    b.cx(0, 1)
    cmds = _flat(b)
    from_dag = count_resources(qx.CircuitDAG.from_commands(cmds), provenance=LOGICAL)
    assert from_dag == count_resources(cmds, provenance=LOGICAL)


def test_pseudo_set_derives_from_gate_is_physical():
    """counting's exclusion set is derived from qx.gate_is_physical — the
    same classification behind Block.n_nqb_gates — so the two surfaces
    cannot drift apart."""
    assert _PSEUDO == {"Barrier", "GPhase", "BranchBegin", "BranchElse", "BranchEnd"}


def test_counting_agrees_with_block_accessors():
    """estimate/count and the Block gate-count accessors are two views of one
    classification: they must agree on a mixed circuit."""
    from qarp.blocks import SimpleBlock

    b = SimpleBlock(3)
    b.h(0)
    b.t(1)
    b.cx(0, 1)
    b.ccx(0, 1, 2)
    b.measure([(q, q) for q in range(3)])
    b.build()

    vec = count_resources(b.flatten(), provenance=Provenance(stage=Stage.LOGICAL), n_qubits=3)
    assert vec.n_1q == b.n_1q_gates()
    assert vec.n_2q == b.n_2q_gates()
    assert vec.n_3q_plus == b.n_nqb_gates(3)  # the CCX
    assert vec.n_gates == b.n_gates()
    assert b.n_gates() == b.n_1q_gates() + b.n_2q_gates() + b.n_nqb_gates(3)
    assert vec.op_histogram.get("T", 0) == b.n_gates_of_type(qx.GateType.T)


def test_arity_buckets_partition_n_gates_for_wide_gates():
    """A width-6 MCZ is neither 1q nor 2q; before ``n_3q_plus`` it counted
    toward ``n_gates`` and vanished from the arity breakdown entirely.

    Oracle is the C++ classifier ``qx.n_nqb_gates`` (``Block.n_nqb_gates``),
    which buckets by each command's own qubit list — independent of the
    Python counting loop under test.
    """
    b = SimpleBlock(6)
    b.h(0)
    b.cx(0, 1)
    b.ccx(0, 1, 2)
    b.mcz(0, 1, 2, 3, 4, 5)
    b.build()

    vec = count_resources(_flat(b), provenance=LOGICAL)
    assert vec.n_1q == b.n_1q_gates() == 1
    assert vec.n_2q == b.n_2q_gates() == 1
    assert vec.n_3q_plus == b.n_nqb_gates(3) + b.n_nqb_gates(6) == 2
    assert vec.n_1q + vec.n_2q + vec.n_3q_plus == vec.n_gates == 4
