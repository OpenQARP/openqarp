Resource estimation
===================

:mod:`qarp.resources` answers "what does this circuit cost?" with numbers that
say where in the compilation pipeline they were taken.  A
:class:`~qarp.resources.ResourceVector` is one snapshot — width, depth, gate
counts by arity, measurements, resets, T count, SWAP count, a per-gate
histogram — stamped with a :class:`~qarp.resources.Provenance` naming the
stage, gate set, optimisation level, router and device that produced it.  A
:class:`~qarp.resources.ResourceReport` is the ordered set of those snapshots
for one circuit.

The module never runs a circuit.  It mirrors the staging an engine performs
(rebase → optimise → route → rebase, see :doc:`devices` and :doc:`engines`) and
counts at each boundary, so a number is always attributable to a configured
pipeline rather than to "the circuit" in the abstract.  The rules the numbers
obey are the contract's §19; this page shows the API.

Counting a block
----------------

:func:`~qarp.resources.estimate` is the entry point.  With no gate set and no
device it takes the ``LOGICAL`` snapshot only — the circuit exactly as built:

.. code-block:: python

    from qarp.blocks import SimpleBlock
    from qarp.resources import Stage, estimate

    def workload(n=4):
        b = SimpleBlock(n, name="workload")
        b.h(0)
        for k in range(1, n):
            b.cx(0, k)
        b.ccx(0, 1, 2)
        b.rz(3, 0.4)
        b.gphase(0.2)                     # a global phase is not a physical gate
        b.reset(1)                        # nor is a reset
        b.measure([(q, q) for q in range(n)])
        return b.build()

    block = workload()
    report = estimate(block)
    v = report[Stage.LOGICAL]

    print("width / depth:", v.n_qubits, v.depth)
    print("gates by arity:", v.n_1q, v.n_2q, v.n_3q_plus, "of", v.n_gates)
    print("measurements / resets:", v.n_measurements, v.n_resets)
    print("histogram:", dict(sorted(v.op_histogram.items())))

Two invariants worth internalising, both from §19:

* **The arity buckets partition the gate count.**  ``n_1q + n_2q + n_3q_plus
  == n_gates`` at every stage, and bucketing follows each command's own qubit
  list — the ``CCX`` above lands in ``n_3q_plus``, so a controlled block's
  wide gates are never invisible to a consumer reading the breakdown.
* **``None`` is never ``0``.**  A field is ``None`` when the quantity is not
  expressible for that snapshot: nothing has been routed, so
  ``swap_count is None``; a symbolic or non-Clifford ``Rz`` is present, so
  ``t_count is None``.  Unknowns are declared, never imputed — which is what
  makes ``swap_count == 0`` on a routed stage a real answer.

.. code-block:: python

    assert v.n_1q + v.n_2q + v.n_3q_plus == v.n_gates
    assert v.swap_count is None           # not routed
    assert v.t_count is None              # an Rz(0.4) has no exact T count

What counts as a gate is one classification, ``qx.gate_is_physical``, shared
with the block accessors ``n_gates()`` / ``n_1q_gates()`` / ``n_2q_gates()`` /
``n_nqb_gates(k)`` / ``depth()`` — so the report and the block agree by
construction.  ``Measure`` and ``Reset`` are not gates but are reported
separately; ``Barrier``, ``GPhase`` and branch markers are pseudo-ops and
appear nowhere.

.. code-block:: python

    assert v.n_gates == block.n_gates()
    assert v.n_3q_plus == block.n_nqb_gates(3)
    assert v.depth == block.depth()

Stages
------

Give ``estimate`` a gate set and a device and it snapshots every pipeline
boundary.  ``OPTIMIZED`` is taken after the rebase and O1 optimisation,
``ROUTED`` after routing but *before* the final rebase — the only stage at
which the router's SWAPs exist as ``SWAP`` gates, so ``swap_count`` is
countable there — and ``TARGET`` after the final rebase into the device gate
set:

.. code-block:: python

    import qarpx as qx
    from qarp.blocks import SimpleBlock
    from qarp.devices import Device, get_nearest_neighbour_architecture
    from qarp.resources import Stage, estimate

    def fanout(n=5):
        b = SimpleBlock(n)
        b.h(0)
        for k in range(1, n):
            b.cx(0, k)                    # CX(0, k) fan-out forces SWAPs on a line
        b.rz(3, 1e-3)
        b.rz(4, 0.3)
        b.measure([(q, q) for q in range(n)])
        return b.build()

    line = Device(5, architecture=get_nearest_neighbour_architecture(1, 5))
    report = estimate(
        fanout(),
        gateset=qx.clifford_t_rz_gateset(),
        device=line,
        device_label="line-5",
    )

    show = lambda x: "—" if x is None else x
    print(f"{'stage':<10}{'width':>6}{'depth':>6}{'gates':>6}{'2q':>5}{'swap':>6}{'T':>5}")
    for stage, v in report.items():
        print(f"{stage.value:<10}{v.n_qubits:>6}{v.depth:>6}{v.n_gates:>6}"
              f"{v.n_2q:>5}{show(v.swap_count):>6}{show(v.t_count):>5}")

``report.final`` is the last stage present; ``report.stages`` lists them in
pipeline order; ``report[Stage.ROUTED]`` addresses one.  ``swap_count`` counts
the ``SWAP`` gates in the routed stream — router-inserted plus any the user
wrote that survive the rebase — not router overhead alone.  The ``Rz`` gates
survive into ``TARGET`` (the Clifford+T+Rz gate set keeps them), so ``t_count``
stays ``None`` on every stage here; making it countable is the job of
synthesis below.

The pipeline ``estimate`` walks is the *configured* one.  It may legitimately
differ from a particular engine's execution pipeline — an engine need not fuse
at all — so pick ``opt_level`` to match the pipeline you are claiming to
describe.

Modelers
--------

Counted fields come from the command stream and nothing else.  Anything that
requires a cost model — a T budget for a synthesis you have not run, a
per-gate error, a wall-clock estimate — is a *modeled* field, produced by a
:class:`~qarp.resources.ResourceModeler` and stamped with
``provenance.modeler``.  The two never mix: a modeler cannot rewrite a counted
field, and when the stream contains something opaque (a ``Custom`` gate) the
modeler must null its outputs rather than undercount.

No modeler ships in-tree; the protocol is the extension seam.  A conforming
object exposes ``name`` and ``model(commands, vector) -> ResourceVector``, and
:meth:`~qarp.resources.ResourceVector.with_model` is how it returns the
annotated copy:

.. code-block:: python

    import qarpx as qx
    from qarp.resources import estimate

    class OneTPerRz:
        """Toy: price every Rz as one T.  A real modeler prices a synthesis budget."""

        name = "one_t_per_rz"

        def model(self, commands, vector):
            n_rz = sum(c.gate == qx.GateType.Rz for c in commands)
            opaque = any(c.gate == qx.GateType.Custom for c in commands)
            return vector.with_model(
                modeler=self.name,
                t_count_modeled=None if opaque else float(n_rz),
                extras={"n_rz": float(n_rz)},
            )

    report = estimate(
        fanout(),
        gateset=qx.clifford_t_rz_gateset(),
        device=line,
        device_label="line-5",
        modeler=OneTPerRz(),
    )
    final = report.final
    print("stage:          ", final.provenance.stage.value)
    print("modeler:        ", final.provenance.modeler)
    print("t_count:        ", final.t_count, "  (counted: Rz gates remain, so None)")
    print("t_count_modeled:", final.t_count_modeled)
    print("extras:         ", final.extras)

    assert final.t_count is None                     # counted stays honest …
    assert final.t_count_modeled == final.extras["n_rz"]   # … the model fills the gap

Clifford+T synthesis
--------------------

Every transpiler pass is unitary-exact (§16), so approximate ``Rz`` →
Clifford+T synthesis is not a pass: it is a declared-``ε`` stage that lives
outside the transpiler.  Pass ``synthesis_epsilon`` and ``estimate`` appends a
``SYNTHESIZED`` snapshot in which every ``Rz`` has been replaced by a
phase-exact ``H``/``S``/``T``/``X`` sequence within ``ε`` per rotation (Ross &
Selinger's gridsynth, via the optional ``pygridsynth`` dependency —
``pip install "openqarp[cliffordt]"``).  There ``t_count`` is exactly countable:

.. code-block:: python

    import math
    import qarpx as qx
    from qarp.resources import Stage, estimate

    report = estimate(
        fanout(),
        gateset=qx.clifford_t_rz_gateset(),
        device=line,
        device_label="line-5",
        synthesis_epsilon=1e-10,
    )
    syn = report[Stage.SYNTHESIZED]
    print("synthesis:", syn.provenance.synthesis)
    print("t_count:  ", syn.t_count)
    print("Rz left:  ", syn.op_histogram.get("Rz", 0))

    # Ross & Selinger: ~3·log2(1/ε) T per non-Clifford rotation.
    print("expected per rotation:", round(3 * math.log2(1e10)))
    assert syn.t_count is not None
    assert "Rz" not in syn.op_histogram

Angles are folded modulo ``π/4`` *exactly* before synthesis, so an ``Rz`` whose
angle is a multiple of ``π/4`` becomes a one-gate ``S``/``T`` form and never
reaches gridsynth — multi-controlled lowerings emit those by the hundred.
``ε`` is a per-rotation knob, not a global budget.

The same replacement is available standalone on a Clifford+T+Rz-format stream
through :func:`~qarp.resources.synthesize_clifford_t`:

.. code-block:: python

    import qarpx as qx
    from qarp.blocks import SimpleBlock
    from qarp.resources import synthesize_clifford_t

    b = SimpleBlock(1)
    b.rz(0, 0.3)
    b.build()

    synthesized = synthesize_clifford_t(b.flatten(), epsilon=1e-6)
    print(len(synthesized), "Clifford+T gates for one Rz at ε = 1e-6")
    assert all(c.gate != qx.GateType.Rz for c in synthesized)

Serialising a report
--------------------

``to_dict()`` on a vector or a report is a stable wire format for external
resource-estimation tooling, versioned by
:data:`~qarp.resources.SCHEMA_VERSION`; :meth:`~qarp.resources.ResourceVector.from_dict`
reads it back.  Changes are additive-with-a-bump: the number never moves
without a migration.

.. code-block:: python

    import json
    from qarp.resources import SCHEMA_VERSION, ResourceVector

    d = final.to_dict()
    print("schema", SCHEMA_VERSION, "keys:", sorted(d)[:6], "…")
    assert ResourceVector.from_dict(d) == final

    print(json.dumps(d, indent=2)[:400], "…")

Counting a raw command stream
-----------------------------

``estimate`` is a facade over :class:`~qarp.resources.ResourceEstimator`,
which owns the pipeline; the bare counter underneath is
:func:`~qarp.resources.count_resources`, for when you already hold a command
list or a ``CircuitDAG`` and only want one snapshot.  It needs the provenance
spelled out, because a command list carries no memory of what produced it:

.. code-block:: python

    from qarp.resources import Provenance, Stage, count_resources

    v = count_resources(block.flatten(),
                        provenance=Provenance(stage=Stage.LOGICAL),
                        n_qubits=block.n_qubits)
    assert v == estimate(block)[Stage.LOGICAL]

The notebooks ``examples/resources/mwe_counting.ipynb`` (the block accessors
and how the books balance between physical gates and non-gate commands) and
``examples/resources/mwe_resource_estimation.ipynb`` (the full staged report,
a modeler, and synthesis) are the runnable companions to this page.
