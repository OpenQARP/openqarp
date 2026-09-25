Post-selection
==============

:class:`~qarp.PostSelection` conditions results *after the fact*: it never touches circuits
or engines. It applies equally to readout distributions (sampled or ``qarp.EXACT``) and to
statevectors, and always reports the **success rate**: the probability mass that survived
the condition.

Conventions (qarpx LSB throughout, see :doc:`endianness`): distribution keys are LSB-first
tuples with qubit ``q`` at position ``q``; for statevectors, bit ``q`` of the amplitude index
is qubit ``q``.  :meth:`~qarp.PostSelection.apply` takes a :class:`~qarp.SamplingDistribution`
or a plain ``{bits-tuple: probability}`` dict whose keys share one width, and its result's
``distribution`` is a :class:`~qarp.SamplingDistribution`.

Two kinds of condition
-----------------------

``PostSelection`` is constructed one of two ways, and the choice changes the **shape** of the
output. This is the footgun to know about before using either:

- **Fixed-bit**: ``PostSelection({qubit: bit, ...})``. Collapses the selected qubits to a
  definite basis state, so they carry no more information and are **removed** from the
  output: distribution keys shrink to the surviving qubits (ascending order), statevectors
  compress from ``2**n`` to ``2**(n - k)`` amplitudes.
- **Sector**: :meth:`~qarp.PostSelection.hamming_weight` / :meth:`~qarp.PostSelection.parity`.
  Projects onto a *subspace* (e.g. a fixed particle number under Jordan-Wigner) in which the
  selected qubits generally stay entangled with the rest of the register. There is nothing to
  drop, so the output keeps the **full register width**.

Mixing the two up silently produces the wrong-shaped output rather than an error, since both
are valid outputs, just not the one you meant.

Fixed bits on a readout distribution
-------------------------------------

.. code-block:: python

    from qarp import PostSelection

    # A 7-qubit readout distribution.
    readout = {
        (1, 0, 1, 0, 0, 0, 1): 0.55,
        (1, 1, 0, 1, 0, 0, 1): 0.30,
        (0, 0, 1, 0, 0, 0, 0): 0.15,
    }

    # Keep outcomes where qubits 3, 4, 5 read 0; those qubits then drop out.
    ps = PostSelection({3: 0, 4: 0, 5: 0})
    out = ps.apply(readout)

    print("conditioned distribution:", out.distribution)
    print("success rate:", out.success_rate)

``apply`` returns a :class:`~qarp.PostSelected`, a ``(distribution,
success_rate)`` pair. The conditioned distribution is renormalised (probabilities sum to 1
over the surviving keys); ``success_rate`` is the probability mass that satisfied the
condition before renormalisation. Zero surviving mass returns ``PostSelected({}, 0.0)``
rather than raising, so a condition with vanishing support does not break a parameter sweep.

The same spec applies to any :class:`~qarp.algorithms.Sampler` output, whether it came from
finite shots or the exact Born distribution (``n_shots=qarp.EXACT``):

.. code-block:: python

    import qarp
    from qarp.algorithms import Sampler
    from qarp.blocks import SimpleBlock
    from qarp.engines import QarpEngine

    bell = SimpleBlock(2, name="bell")
    bell.h(0)
    bell.cx(0, 1)
    bell.build()

    ps = PostSelection({0: 0})

    for n_shots in (4000, qarp.EXACT):
        sampler = Sampler(ket=bell, n_shots=n_shots)
        engine = QarpEngine(seed=7)
        engine.build([sampler])
        out = ps.apply(engine.run()[0])
        print(f"n_shots={n_shots}:  distribution={out.distribution}  success={out.success_rate:.4f}")

On a Bell pair, conditioning qubit 0 on 0 forces qubit 1 to 0 in both cases; the sampled
success rate carries shot noise around the exact value of 0.5.

Statevectors
------------

:meth:`~qarp.PostSelection.apply_statevector` projects a statevector onto the condition and
renormalises, returning ``(conditional_state, success_probability)`` with
``success = ‖P|ψ⟩‖²``. For a fixed-bit condition the selected qubits factor out of the state
entirely, so the returned state is the (renormalised) state of the surviving qubits only:

.. code-block:: python

    import numpy as np
    from qarp.blocks import SimpleBlock

    ghz = SimpleBlock(3, name="ghz")
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(1, 2)
    ghz.build()

    sv = ghz.statevector()

    conditional, p = PostSelection({0: 0}).apply_statevector(sv, 3)
    print("success probability:", p)
    print("conditional state of qubits 1, 2:", np.round(conditional, 6))

Sector post-selection
----------------------

:meth:`~qarp.PostSelection.hamming_weight` / :meth:`~qarp.PostSelection.parity` select a
*symmetry sector*, e.g. a fixed particle number under Jordan-Wigner, rather than fixed
qubit values. A sector is a subspace, not a basis state: the selected qubits stay entangled
inside it, so (unlike the fixed-bit case) the output keeps the **full register width**.

.. code-block:: python

    # A state with particle-number leakage: mostly N=2, some N=1 amplitude.
    sv = np.zeros(16, dtype=complex)
    sv[0b0011] = np.sqrt(0.45)  # qubits 0, 1 occupied
    sv[0b0101] = np.sqrt(0.45)  # qubits 0, 2 occupied
    sv[0b0001] = np.sqrt(0.10)  # N=1 leakage

    ps_n2 = PostSelection.hamming_weight(range(4), k=2)

    projected, p = ps_n2.apply_statevector(sv, 4)
    print("N=2 sector weight:", round(p, 4))
    print("projected state norm:", round(float(np.linalg.norm(projected)), 4))

    born = {tuple(int(i >> q) & 1 for q in range(4)): abs(a) ** 2
            for i, a in enumerate(sv) if abs(a) > 0}
    out = ps_n2.apply(born)
    print("conditioned distribution:", out.distribution)

Note that the last ``apply`` call above returns keys covering all 4 qubits. A sector
condition constrains a *joint* property of the register, here the total weight over
``range(4)``, rather than pinning individual qubits to values, so there is nothing to
factor out and no qubit is ever removed. This holds however few qubits the spec names:
``hamming_weight([0, 1], k=2)`` on the same input also returns full-width keys.

:meth:`~qarp.PostSelection.parity` follows the same shape rule: ``PostSelection.parity([0,
1], even=True)`` keeps the even-bit-sum sector of qubits 0 and 1, full register width
preserved.

Reusable specs and parameter sweeps
-------------------------------------

A ``PostSelection`` instance is immutable and hashable (fixed-bit specs compare by their
conditions; sector specs by predicate identity), so the same spec can be applied across a
sweep without rebuilding it each time. :meth:`~qarp.PostSelection.success_rate` is a
shortcut for ``apply(...).success_rate`` when only the rate is needed:

.. code-block:: python

    import numpy as np

    ps = PostSelection({1: 0})

    for theta in np.linspace(0, np.pi, 5):
        ket = SimpleBlock(2)
        ket.ry(0, theta)
        ket.cx(0, 1)
        ket.build()
        sampler = Sampler(ket=ket, n_shots=qarp.EXACT)
        engine = QarpEngine()
        engine.build([sampler])
        rate = ps.success_rate(engine.run()[0])
        print(f"{theta / np.pi:.2f}  success={rate:.4f}  cos^2(theta/2)={np.cos(theta / 2) ** 2:.4f}")

By construction, ``success_rate`` here equals :math:`\cos^2(\theta/2)`, including the
vanishing-support point at :math:`\theta = \pi`, which reports rate ``0.0`` instead of
raising.

Shot-noise caveat
-------------------

On sampled input the success rate carries statistical error (:math:`\sigma \approx
\sqrt{p(1-p)/N}`), and the conditioned distribution rests on an effective
``N * success_rate`` shots, so quote error bars accordingly. With ``n_shots=qarp.EXACT`` input,
both the success rate and the conditioned distribution are exact.
