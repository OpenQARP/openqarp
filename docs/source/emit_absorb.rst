Emitters & Absorbers
=====================

Any OpenQARP block can be **emitted** to four Python SDKs (Qiskit, Qulacs,
PyTKET, PennyLane) plus OpenQASM 3 and OpenQASM 2, and **absorbed** back from any of them. This is the
circuit-interchange boundary: use it to hand a OpenQARP-built circuit to another SDK's tooling,
or to bring a circuit built elsewhere into OpenQARP's block/engine pipeline.

The block-level methods are the ergonomic entry point, symmetric by construction:

.. code-block:: python

    import math
    from qarp.blocks import SimpleBlock

    block = SimpleBlock(2, name="bell")
    block.h(0)
    block.cx(0, 1)
    block.rz(1, math.pi / 4)
    block.build()

    qc = block.to_qiskit()          # -> qiskit.QuantumCircuit
    back = SimpleBlock.from_qiskit(qc)   # -> SimpleBlock

    print("round trip ==:", back == block)

``to_qasm3()`` / ``SimpleBlock.from_qasm3(...)`` are the OpenQASM 3 pair and
``to_qasm2()`` / ``SimpleBlock.from_qasm2(...)`` the OpenQASM 2 one; neither needs an SDK
installed (the emitters/absorbers are built into qarpx). The dialect is always in the
method name — the two languages carry different things, so the call site says which one
it got, and matches the ``can_emit_to("qasm2")`` / ``can_emit_to("qasm3")`` pre-flight
that guards it. ``to_qir()`` emits QIR (LLVM IR) the same way. The four SDK pairs
``to_qiskit``/``from_qiskit``, ``to_qulacs``/``from_qulacs``, ``to_pytket``/``from_pytket``,
``to_pennylane``/``from_pennylane`` need the corresponding package installed; each is a
thin wrapper around the matching class in :mod:`qarp.emit` / :mod:`qarp.absorb`:

.. code-block:: python

    from qarp.emit import QiskitEmitter, QulacsEmitter, PytketEmitter, PennylaneEmitter
    from qarp.emit import QASM3Emitter, QASM2Emitter, QIREmitter
    from qarp.absorb import QiskitAbsorber, QulacsAbsorber, PytketAbsorber, PennylaneAbsorber, QASM3Absorber, QASM2Absorber

The rest of this page uses these classes directly, since several examples emit the same
block to multiple SDKs side by side.

Checking a round trip: ``Block.__eq__``
------------------------------------------

``block1 == block2`` compares qubit count and the flattened command sequence (gate types,
qubits, cbits, classical conditions, rotation angles or symbols) within a small numerical
tolerance (default ``1e-9``). It is the tool every example on this page uses to check that
what comes back out of a round trip is *the same circuit*, not just an equivalent-looking one.

Parametric gates
-------------------

Rotation angles survive the round trip even when the target SDK uses a different angle
convention internally (Qulacs negates rotation angles; PyTKET uses half-turns), because ``==``
compares the logical parameter value, not the SDK's internal representation:

.. code-block:: python

    param_block = SimpleBlock(2)
    param_block.crx(0, 1, math.pi / 4).rzz(0, 1, math.pi / 6)
    param_block.build()

    for name, emitter, absorber in [
        ("Qiskit", QiskitEmitter(), QiskitAbsorber()),
        ("Qulacs", QulacsEmitter(), QulacsAbsorber()),
        ("PyTKET", PytketEmitter(), PytketAbsorber()),
        ("PennyLane", PennylaneEmitter(), PennylaneAbsorber()),
    ]:
        native = emitter.emit(param_block.flatten(), param_block.n_qubits)
        absorbed = absorber.absorb(native)
        print(f"{name:10s} absorbed == param_block: {absorbed == param_block}")

Symbolic parameters
-----------------------

Qiskit and PyTKET support symbolic (unbound) parameters natively; the round trip preserves
the symbol *name* through ``==``, not just a numeric value:

.. code-block:: python

    import qarpx as qx

    symbolic_block = SimpleBlock(1)
    symbolic_block.rx(0, qx.Param.symbol("theta"))
    symbolic_block.build()

    qc = QiskitEmitter().emit(symbolic_block.flatten(), 1)
    print("Qiskit  absorbed == symbolic_block:", QiskitAbsorber().absorb(qc) == symbolic_block)

    circ = PytketEmitter().emit(symbolic_block.flatten(), 1)
    print("PyTKET  absorbed == symbolic_block:", PytketAbsorber().absorb(circ) == symbolic_block)

Across the SDK boundary a symbolic parameter travels as a *linear form* ``c*x + d`` in one
symbol — that is what Qiskit's ``ParameterExpression`` arithmetic and PyTKET's half-turn
``sympy`` expressions can carry losslessly.  Both directions probe the expression at three
points and **refuse** anything else (``Param.symbol("t") * Param.symbol("t")``, a two-symbol
sum) with :class:`~qarp.errors.CapabilityError`; ``can_emit_to`` reports the same without the
SDK.  OpenQASM 3 writes the expression verbatim and has no such limit.

Qulacs and PennyLane have no symbolic-parameter concept. Emitting a block with an unbound
symbol to either **raises** rather than silently dropping or freezing the symbol:

.. code-block:: python

    from qarp.errors import CapabilityError

    try:
        QulacsEmitter().emit(symbolic_block.flatten(), 1)
    except CapabilityError as e:
        print("Qulacs raises as expected:", e, "on command", e.command)

Emit/absorb rejections follow the same contract :doc:`errors` documents for engines: a
circuit the target cannot represent raises :class:`~qarp.errors.CapabilityError`, carrying
the offending command on its ``command`` attribute; a missing SDK raises ``ImportError``
with a ``pip install`` hint; only parse and internal failures stay ``RuntimeError``. The
check runs inside ``emit()`` via the emitter's ``validate()``, so it fires before the SDK
is even imported. Use :meth:`~qarp.blocks.SimpleBlock.can_emit_to` to ask the same question
without raising.

Mid-circuit measurement
---------------------------

Measure-then-reuse on the same qubit survives the round trip, in order, with the correct
classical-bit mapping:

.. code-block:: python

    mcm_block = SimpleBlock(1)
    mcm_block.x(0)
    mcm_block.measure(0, 0)
    mcm_block.x(0)
    mcm_block.measure(0, 1)
    mcm_block.build()

    for name, emitter, absorber in [
        ("Qiskit", QiskitEmitter(), QiskitAbsorber()),
        ("Qulacs", QulacsEmitter(), QulacsAbsorber()),
        ("PyTKET", PytketEmitter(), PytketAbsorber()),
    ]:
        native = emitter.emit(mcm_block.flatten(), mcm_block.n_qubits)
        absorbed = absorber.absorb(native)
        print(f"{name:10s} absorbed == mcm_block: {absorbed == mcm_block}")

PennyLane has no classical-register concept, so it requires ``cbit == qubit`` on every
measurement, so the equivalent circuit for PennyLane measures the same qubit into its own
index both times:

.. code-block:: python

    mcm_block_pl = SimpleBlock(1)
    mcm_block_pl.x(0)
    mcm_block_pl.measure(0, 0)
    mcm_block_pl.x(0)
    mcm_block_pl.measure(0, 0)
    mcm_block_pl.build()

    tape = PennylaneEmitter().emit(mcm_block_pl.flatten(), mcm_block_pl.n_qubits)
    absorbed_pl = PennylaneAbsorber().absorb(tape)
    print("PennyLane  absorbed == mcm_block_pl:", absorbed_pl == mcm_block_pl)

Classical conditionals and barriers
--------------------------------------

A :class:`~qarp.blocks.ConditionalBlock` crosses to Qiskit as an ``IfElseOp`` (single-bit
condition; a wider ``ClassicalRegister`` comparison raises
:class:`~qarp.errors.CapabilityError` on the way back) and to PyTKET as per-command
``Conditional`` ops — PyTKET has no block-level branch, so the *else* arm is written as a
second run on the complemented bit and the absorber folds it back.  Either way the absorbed
block compares ``==`` to the original, else arm and ``GPhase`` inside the branch included:

.. code-block:: python

    from qarp.blocks import CompositeBlock, ConditionalBlock

    meas = SimpleBlock(2)
    meas.h(0).measure(0, 0)
    meas.build()
    flip = SimpleBlock(2)
    flip.x(1)
    flip.build()
    cond = ConditionalBlock(cbits=[0], values=[True], then_body=flip)
    cond.target_cbits = [0]  # reads the cbit `meas` wrote (§8)
    feedforward = CompositeBlock([meas, cond], 2)
    feedforward.build()

    qc = QiskitEmitter().emit(feedforward.flatten(), 2)
    print("Qiskit  absorbed == feedforward:", QiskitAbsorber().absorb(qc) == feedforward)
    circ = PytketEmitter().emit(feedforward.flatten(), 2)
    print("PyTKET  absorbed == feedforward:", PytketAbsorber().absorb(circ) == feedforward)

Barriers are kept, not dropped, on both SDKs — a ``Barrier`` command comes back as a
``Barrier`` on the same qubits, so a scheduling fence placed before emission survives the
round trip.  Qiskit's ``circuit.global_phase`` (float or ``ParameterExpression``) and its
legacy ``u1``/``u2``/``u3``/``r`` gates absorb exactly, so a circuit that went through
``qiskit.transpile`` — which folds phases into ``global_phase`` — equals its
``qiskit.quantum_info.Operator`` in qarp's own LSB layout.

Nested ``CompositeBlock``\\ s
---------------------------------

``Block.__eq__`` compares *flattened* command sequences, so a nested ``CompositeBlock``
compares equal to the flat ``SimpleBlock`` an absorber reconstructs, since the comparison
doesn't care that one side has sub-block structure and the other doesn't:

.. code-block:: python

    from qarp.blocks import CompositeBlock

    sub1 = SimpleBlock(2)
    sub1.h(0).cx(0, 1)
    sub1.build()

    sub2 = SimpleBlock(2)
    sub2.rz(1, math.pi / 4).cz(0, 1)
    sub2.build()

    comp = CompositeBlock([sub1, sub2], 2)
    comp.build()

    qc = QiskitEmitter().emit(comp.flatten(), comp.n_qubits)
    absorbed = QiskitAbsorber().absorb(qc)
    print("absorbed == comp:", absorbed == comp)

QASM3 round trips and tolerance
------------------------------------

For circuits with rotation angles, a target format's exported numeric precision can matter.
If a round trip through a given backend doesn't compare equal at the default ``1e-9``
tolerance, use ``qx.commands_equal(a, b, atol=...)`` (or ``Block.equals(other, atol=...)``)
directly with a wider tolerance rather than concluding the round trip is broken:

.. code-block:: python

    qasm_block = SimpleBlock(2, name="qasm_demo")
    qasm_block.h(0).cx(0, 1).rz(1, math.pi / 4)
    qasm_block.build()

    qasm_str = QASM3Emitter().emit(qasm_block.flatten(), qasm_block.n_qubits, qasm_block.name)
    absorbed_qasm = QASM3Absorber().absorb(qasm_str)

    print("bare ==:                  ", absorbed_qasm == qasm_block)
    print("commands_equal(atol=1e-4):", qx.commands_equal(absorbed_qasm.flatten(), qasm_block.flatten(), atol=1e-4))

OpenQASM 2: a narrower language
------------------------------------

OpenQASM 2 is what most hardware submission endpoints, older simulators and published
circuit files speak, so ``to_qasm2()`` exists alongside ``to_qasm3()``. It is a genuinely
smaller language, and the emitter says so rather than quietly dropping what it cannot
write — symbolic parameters, ``GPhase`` and ``MCZ`` each raise
:class:`~qarp.errors.CapabilityError`:

.. code-block:: python

    from qarp.errors import CapabilityError

    q2_block = SimpleBlock(2, name="qasm2_demo")
    q2_block.h(0).cx(0, 1).rzz(0, 1, math.pi / 4)
    q2_block.build()

    print(q2_block.to_qasm2())
    print("can_emit_to:", q2_block.can_emit_to("qasm2"))

    phase_block = SimpleBlock(1, name="phased")
    phase_block.h(0).gphase(0.25)
    phase_block.build()
    try:
        phase_block.to_qasm2()
    except CapabilityError as exc:
        print("rejected:", exc)

The spec's ``qelib1.inc`` is only 23 gates, so anything outside it — ``swap``, ``rzz``,
``ecr``, ``cu`` and the rest — is written with its own ``gate`` definition in the program's
prelude, each body phase-exact. ``from_qasm2`` deliberately accepts more than ``to_qasm2``
writes (multiple registers, whole-register ``measure q -> c;``, the qiskit-extended
``qelib1.inc`` names), so files OpenQARP did not produce parse too.

What ``==`` catches
-----------------------

A different rotation angle, a different symbol name, or a different qubit count all
correctly compare unequal, so ``==`` is a real equality check, not a shape/type check:

.. code-block:: python

    a = SimpleBlock(2)
    a.h(0).cx(0, 1).rz(1, math.pi / 4)
    a.build()

    b_diff_angle = SimpleBlock(2)
    b_diff_angle.h(0).cx(0, 1).rz(1, math.pi / 3)
    b_diff_angle.build()

    print("different rotation angle:", a == b_diff_angle)
    assert not (a == b_diff_angle)

See ``examples/emit_absorb/mwe_emitters_absorbers.ipynb`` for the complete
worked notebook this page is drawn from.
