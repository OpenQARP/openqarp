"""The ``ResourceModeler`` seam: ``estimate(modeler=)`` plumbing, provenance
tagging and the ``Engine.resource_modeler()`` hook.

No modeler ships in the tree, so the seam is driven by ``CountRzModeler``
(tests/conftest.py).  Oracles are structural — which stage carries the
modeled field, which name provenance records, which engines override the
hook — never the stub's arithmetic.
"""

import qarpx as qx
from qarp.blocks import SimpleBlock
from qarp.engines import CudaqEngine, Engine, QarpEngine
from qarp.resources import Stage, estimate
from tests.conftest import CountRzModeler, _StubEngine


def _block() -> SimpleBlock:
    b = SimpleBlock(2)
    b.h(0)
    b.cx(0, 1)
    b.rz(1, 0.3)
    b.rz(0, 1.1)
    return b


def test_modeler_fills_the_final_stage_only():
    rep = estimate(_block(), gateset=qx.clifford_t_rz_gateset(), modeler=CountRzModeler())
    final = rep.final
    assert final.provenance.stage is Stage.OPTIMIZED
    # the modeler saw the final-stage stream: its Rz count is the histogram's
    assert final.t_count_modeled == float(final.op_histogram.get("Rz", 0))
    assert final.extras["n_rz"] == final.t_count_modeled
    # earlier stages stay counted-only (§19: counted and modeled never mix)
    assert rep[Stage.LOGICAL].t_count_modeled is None
    assert rep[Stage.LOGICAL].provenance.modeler is None


def test_provenance_records_the_modeler_name():
    rep = estimate(_block(), gateset=qx.clifford_t_rz_gateset(), modeler=CountRzModeler())
    assert rep.final.provenance.modeler == "count_rz:t=1"
    # counted field untouched by modeling
    assert rep.final.t_count is None  # Rz survives the Clifford+T+Rz rebase


def test_no_modeler_means_no_modeled_fields():
    rep = estimate(_block(), gateset=qx.clifford_t_rz_gateset())
    assert rep.final.t_count_modeled is None
    assert rep.final.provenance.modeler is None


def test_resource_modeler_hook_is_unimplemented_in_tree():
    """§14: the hook survives with no implementer — base ``None`` is the
    only in-tree return.  CudaqEngine is checked at class level (not
    constructible on CPU builds)."""
    assert QarpEngine().resource_modeler() is None
    assert _StubEngine().resource_modeler() is None
    assert CudaqEngine.resource_modeler is Engine.resource_modeler
