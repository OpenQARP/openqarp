Gradients
=========

Every engine differentiates the primitives it has built through one call,
:meth:`~qarp.engines.Engine.run_gradient`, and every variational algorithm
accepts the same method names through its ``gradient=`` argument.  This page
is the contract: which methods exist, what each returns, what it costs, and
what it refuses.

``examples/engines/mwe_gradients.ipynb`` is this page as an executable
notebook: every method checked against a hand-derived analytic gradient, the
options, the return contract, and each refusal.

.. code-block:: python

    import numpy as np
    import qarpx as qx
    from qarp.algorithms import StateVector
    from qarp.blocks import SimpleBlock
    from qarp.engines import QarpEngine

    ansatz = SimpleBlock(2)
    ansatz.ry(0, qx.Param.symbol("a"))
    ansatz.rx(1, qx.Param.symbol("b"))
    ansatz.cx(0, 1)
    ansatz.build()

    hamiltonian = qx.QubitOperator("Z0") + qx.QubitOperator("X1", 0.5)
    engine = QarpEngine()
    engine.build([StateVector(ket=ansatz, operator=hamiltonian)])

    point = ansatz.parameter_map([0.4, -0.7])
    grad = engine.run_gradient(point)[0]                       # engine default
    shift = engine.run_gradient(point, method="parameter-shift")[0]
    fd = engine.run_gradient(point, method="finite-diff", options={"fd_eps": 1e-6})[0]
    assert np.allclose(grad, shift, atol=1e-10)

Methods
-------

The registry is a set of plain strings.  The descriptive names are
PennyLane's, so a user of either library reads the other's code; the policy
entry is called ``"default"`` rather than ``"best"`` because it is a
per-engine choice that can change between releases, not a method.

.. list-table::
   :header-rows: 1
   :widths: 18 34 24 24

   * - ``method``
     - What it computes
     - Engines
     - Cost per gradient
   * - ``"default"``
     - The engine's policy: ``QarpEngine`` picks ``"adjoint"`` for an
       eligible primitive and ``"parameter-shift"`` otherwise; ``CudaqEngine``
       picks ``"parameter-shift"``.
     - all
     - as resolved
   * - ``"adjoint"``
     - Exact reverse-mode backpropagation in C++ for ``StateVector``
       expectation values over a ``QubitOperator`` and ``StateVector``
       overlaps.
     - ``QarpEngine``
     - about two statevector simulations per differentiated circuit,
       independent of the number of parameters (an overlap whose bra also
       carries a symbol is swept on both sides)
   * - ``"parameter-shift"``
     - Exact analytic shift rules derived from each gate's generator
       spectrum, one shift per *occurrence* of a symbol, all evaluated in one
       batched sweep.
     - all
     - 2 evaluations per occurrence (4 for ``CRx``/``CRy``/``CRz``), batched
   * - ``"finite-diff"``
     - Forward (``fd_order=1``) or central (``fd_order=2``, default)
       differences, step ``fd_eps`` (default ``1e-5``).
     - all
     - ``n + 1`` or ``2n`` evaluations, batched
   * - ``"spsa"``
     - Simultaneous-perturbation estimate over ``num_spsa`` Rademacher
       directions of size ``spsa_c0``; ``spsa_seed`` pins the draw (default:
       the engine's own seeded stream, a fixed stream when the engine is
       unseeded, so independent processes draw the same directions).
     - all
     - ``2 · num_spsa`` evaluations, batched

Two names are reserved for later releases and refused with
:class:`~qarp.errors.CapabilityError` today: ``"hadamard"`` and
``"metric-tensor"``.  An unknown string is a ``ValueError``; a method the
engine does not declare in its ``gradient_methods`` set is a
:class:`~qarp.errors.CapabilityError` naming an engine that has it.

The finite-difference and SPSA methods reproduce the formulas of the
optimizer-side helpers :func:`~qarp.optimizers.compute_fd_gradients` and
:func:`~qarp.optimizers.compute_spsa_gradients` bit for bit under the same
seed.  The difference is where the loop runs: the optimizer helpers call the
objective once per point, the engine methods evaluate every point in a single
batched sweep.

Return contract
---------------

``run_gradient`` returns one array per built primitive, in build order.
Each array has one column per entry of ``params``, in the **insertion order**
of that mapping — for every method, so a caller that builds the mapping from
``block.symbols`` (or ``block.parameter_map``) gets the canonical order back.

The dtype is decided by the primitive's *target*, never by the numeric type
of one evaluation: ``float64`` for expectation values over a
``QubitOperator`` and for overlaps; ``complex128`` for a transition
amplitude, for an expectation value over a circuit-valued (``qx.Block``)
operator, and for a Hadamard test that returns both parts.  Real and
imaginary parts are differentiated independently.

What is differentiated depends on the target:

* **Expectation value** and **transition amplitude**: the value ``run()``
  returns.
* **Overlap on a** ``StateVector``: ``|⟨bra|ket⟩|²`` — although ``run()``
  returns the complex amplitude ``⟨bra|ket⟩``.  This is the quantity the VQD
  family's deflation penalties need, and it holds on every method.
* **Overlap on any other primitive**: the value ``run()`` returns — the
  probability ``|⟨bra|ket⟩|²`` for ``SWAPTest`` and ``MirrorTest``, the real
  (and imaginary) part of the amplitude for a ``HadamardTest``.  ``VQD``
  applies the chain rule itself for the amplitude-returning case.

Every primitive declares a ``gradient_kind`` that says what the shift rules
may assume about ``run()`` as a function of each compiled circuit's state,
separately: ``"expectation"`` (bilinear — every sampled estimator),
``"amplitude"`` (linear in one circuit's amplitudes), ``"squared_overlap"``,
or ``"none"``.  A ``"none"`` primitive (``Sampler``, the classical-shadow
protocols whose median-of-means estimator is not linear, the projected VQE
objective which is a ratio) refuses ``"parameter-shift"`` and keeps
``"finite-diff"``.

Shift rules
-----------

For a gate ``exp(−i·angle·G)`` the value is a trigonometric polynomial in
the angle whose frequencies are the differences of the eigenvalues of ``G``
(expectation values, bilinear in the state) or the eigenvalues themselves
(amplitudes, linear in the state).  The general rule for equidistant
frequencies is that of Wierichs, Izaac, Wang and Lin, *Quantum* **6**, 677
(2022); the table instantiates it in angle units, and every entry is scaled
by the affine coefficient ``c`` of ``angle = c·x + b`` (shift ``s/c`` in
symbol units, coefficient ``a·c``).

.. list-table::
   :header-rows: 1
   :widths: 22 16 31 31

   * - Gate
     - Spectrum
     - Expectation value
     - Amplitude
   * - ``Rx Ry Rz RXX RYY RZZ``
     - ``{±½}``
     - two-term, ``±π/2``, coefficient ``½``
     - two-term, ``±π``, coefficient ``¼``
   * - ``P CP``
     - ``{0, −1}``
     - two-term, ``±π/2``, ``½``
     - two-term, ``±π/2``, ``½``
   * - ``CRx CRy CRz``
     - ``{0, ±½}``
     - four-term, ``±π/2`` with ``(√2+1)/(4√2)`` and ``±3π/2`` with
       ``−(√2−1)/(4√2)``
     - two-term, ``±π``, ``¼``
   * - ``GPhase``
     - ``{−1}``
     - no contribution
     - two-term, ``±π/2``, ``½``
   * - ``U CU``
     - —
     - rewritten into the rows above before differentiating
     - rewritten

A symbol that appears in several gates — a UCC or Trotter ansatz, or a
compound angle such as ``(a + b)/2`` produced by the optimizer's rotation
merge — is shifted one occurrence at a time and the contributions summed, so
shared symbols and mixed coefficients are exact.  The one shape refused is an
angle that is not affine in its symbols (``t·t``); ``"finite-diff"`` still
applies.  A symbolic ``U``/``CU`` has no single generator, so both analytic
paths first rewrite it into ``GPhase·Rz·Ry·Rz`` (the ``P``/``CX`` ladder for
``CU``) and differentiate the compound ``(φ±λ)/2`` phases through the chain
rule; the circuit ``run()`` executes is untouched.  Any other symbolic gate
without a rule is refused, never silently returned as zero (which the adjoint
used to do for ``U``).

Engines
-------

**QarpEngine** provides every method.  ``"default"`` resolves to the adjoint
for an eligible ``StateVector`` primitive and to the batched shift otherwise;
a noise model that is enabled removes the adjoint (amplitudes are undefined
under noise) and the sampled shift takes over.

**CudaqEngine** never constructs a CPU simulator behind the caller's back:
``"adjoint"`` is refused (a GPU adjoint is post-release work; use
``QarpEngine`` for it) and ``"default"`` is the batched shift.  Only a
``StateVector`` expectation value over a ``QubitOperator`` stays on the
device for every shifted point (``batch_expectation``); overlap, Block-operator
and transition targets pull the host statevector per point.  For a small
register with many parameters the two-sweep CPU adjoint of ``QarpEngine`` is
faster than ``2·P`` device evaluations.

Shot noise.  Each shifted point is one more parameter set of the batched
sweep and draws its *own* random tape (set ``i`` is seeded with
``seed + i``), so the difference of two finite-shot evaluations carries the
variance of both; a fixed engine seed makes the whole gradient reproducible
and nothing more.

Variational algorithms
----------------------

``VQE``, ``VQD``, ``SSVQE``, ``QAOA``, ``PCE`` and the ADAPT family accept
``gradient=True`` (the engine's ``"default"``), ``gradient=False`` (a
gradient-free optimizer) or a method name:

.. code-block:: python

    from qarp import EXACT
    from qarp.algorithms import VQE, PauliAveraging

    vqe = VQE(operator=hamiltonian, ket=ansatz, gradient="parameter-shift",
              primitive=PauliAveraging(n_shots=EXACT), initial_parameters=[0.4, -0.7])
    vqe.build()

A sampled primitive with a gradient used to be refused at construction; it
now differentiates through the batched shift.

Using qarp inside jax or torch
------------------------------

qarp does not depend on any autodiff framework, and its gradient is a
Jacobian: one row per primitive, one column per parameter.  That is the
whole contract a reverse-mode framework needs, so plugging an engine into
jax or torch is a few lines the caller owns.  The backward rule is the
cotangent times the Jacobian.

.. docs-lint: skip jax is not a qarp dependency; the recipe is illustrative and exercised by tests/test_integrations/test_gradient_jax_recipe.py

.. code-block:: python

    import jax
    import jax.numpy as jnp
    import numpy as np

    def make_energy(engine, block):
        """A jax-differentiable scalar E(x) backed by a built engine."""

        @jax.custom_vjp
        def energy(x):
            return jnp.asarray(_value(x))

        def _value(x):
            return float(np.real(engine.run(block.parameter_map(np.asarray(x)))[0]))

        def fwd(x):
            return jnp.asarray(_value(x)), x

        def bwd(x, ct):
            jac = np.real(engine.run_gradient(block.parameter_map(np.asarray(x)))[0])
            return (ct * jnp.asarray(jac),)

        energy.defvjp(fwd, bwd)
        return energy

    energy = make_energy(engine, ansatz)
    x = jnp.array([0.4, -0.7])
    jax.grad(energy)(x)            # == engine.run_gradient(...)[0]
    jax.grad(lambda x: energy(x) ** 2)(x)   # chain rule for free

For an overlap primitive, square the forward value inside ``_value`` so the
pair matches the differentiated objective (``|⟨bra|ket⟩|²``).  The torch
equivalent is a ``torch.autograd.Function`` whose ``backward`` returns
``grad_output * jacobian``:

.. docs-lint: skip torch is not a qarp dependency; the recipe is illustrative

.. code-block:: python

    import numpy as np
    import torch

    class Energy(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            ctx.save_for_backward(x)
            value = engine.run(ansatz.parameter_map(x.detach().numpy()))[0]
            return torch.tensor(float(np.real(value)), dtype=x.dtype)

        @staticmethod
        def backward(ctx, grad_output):
            (x,) = ctx.saved_tensors
            jac = np.real(engine.run_gradient(ansatz.parameter_map(x.detach().numpy()))[0])
            return grad_output * torch.as_tensor(jac, dtype=x.dtype)

Post-release roadmap
--------------------

Listed so the names are stable: a GPU adjoint on ``CudaqEngine``
(``"adjoint"`` there), Hadamard-test gradients (``"hadamard"``), the
Fubini–Study metric tensor and natural-gradient optimizer
(``"metric-tensor"``), Hessians, shot-aware shift rules, and a fused
backward pass for many observables on one circuit.  Each is a new method
name, a new keyword with a default, or an error that stops being raised.
