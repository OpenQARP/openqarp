=========================================
Open Quantum Application Research Package
=========================================

|ci| |coverage| |pypi| |docs| |python| |license| |doi|

.. |ci| image:: https://github.com/OpenQARP/openqarp/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/OpenQARP/openqarp/actions/workflows/ci.yml
   :alt: CI

.. |coverage| image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fopenqarp.github.io%2Fopenqarp%2Fbadges%2Fcoverage.json
   :target: https://github.com/OpenQARP/openqarp/actions/workflows/ci.yml
   :alt: Coverage

.. |pypi| image:: https://img.shields.io/pypi/v/openqarp
   :target: https://pypi.org/project/openqarp/
   :alt: PyPI

.. |docs| image:: https://img.shields.io/badge/Docs-GitHub%20Pages-blue
   :target: https://openqarp.github.io/openqarp/
   :alt: Documentation

.. |python| image:: https://img.shields.io/badge/python-3.11+-blue.svg
   :target: https://pypi.org/project/openqarp/
   :alt: Python 3.11+

.. |license| image:: https://img.shields.io/badge/License-Apache_2.0-blue.svg
   :target: https://github.com/OpenQARP/openqarp/blob/develop/LICENSE
   :alt: License: Apache 2.0

.. |doi| image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22755228-blue
   :target: https://doi.org/10.5281/zenodo.22755228
   :alt: DOI

----

OpenQARP is a Python package for quantum computing research.  You describe a
problem — a molecule, a spin model, an optimisation instance — build a circuit
for it, and run that circuit on a fast simulator or compile it for real
hardware.  It is aimed at researchers: the physics is in the foreground, and
the C++ engine underneath is something you should never have to think about.

----

Install
=======

.. code-block:: bash

   pip install openqarp

Wheels ship for Linux (x86_64, arm64), macOS (Apple silicon, Intel) and
Windows on Python 3.11–3.14.  Anywhere else ``pip`` compiles the C++ backend
from source, which takes a few minutes.  For GPUs, editable installs, or a
build on a machine with little memory, see the `Installation Guide`_.  The
distribution is named ``openqarp``; the package you import is ``qarp``.

----

Getting started
===============

Three snippets.  Paste them into a Jupyter notebook in order — each one builds
on the last.

**1. Build a circuit and look at it.**  Circuits are *blocks*: you say how many
qubits, add gates, then ``build()``.

.. code-block:: python

   from qarp.blocks import SimpleBlock

   bell = SimpleBlock(2, name="bell")
   bell.h(0)
   bell.cx(0, 1)
   bell.measure([(q, q) for q in range(2)])   # measure qubit q into bit q
   bell.build()

   bell.plot()                                # draws the circuit diagram

**2. Run it and get the results.**  A *primitive* says what you want out of the
circuit — here, the distribution of measured bitstrings.  The *engine* runs it.

.. code-block:: python

   from qarp.algorithms import Sampler
   from qarp.engines import QarpEngine

   sampler = Sampler(ket=bell, n_shots=4000)

   engine = QarpEngine(seed=42)               # seed makes shots reproducible
   engine.build([sampler])
   results = engine.run()

   print(results[0])
   # {(0, 0): 0.48875, (1, 1): 0.51125}   ← a Bell state, up to shot noise

Keys are tuples of bits indexed by qubit: position ``q`` is qubit ``q``.  OpenQARP
is least-significant-bit-first everywhere.  Values are probabilities, not raw
counts.

**3. Measure an observable.**  Swap the primitive to get an expectation value
instead — exactly, or sampled with shots, from the same circuit.

.. code-block:: python

   from qarp.operators import QubitOperator
   from qarp.algorithms import PauliAveraging, StateVector

   circuit = SimpleBlock(2, name="bell")
   circuit.h(0)
   circuit.cx(0, 1)
   circuit.build()                            # no measurements this time

   H = QubitOperator("Z0 Z1") + 0.5 * QubitOperator("X0 X1")

   exact = StateVector(ket=circuit, operator=H)
   shots = PauliAveraging(ket=circuit, operator=H, n_shots=4000)

   engine = QarpEngine(seed=42)
   engine.build([exact, shots])
   results = engine.run()

   print(results[0], results[1])
   # (1.4999999999999998+0j) 1.5        ← both agree with ⟨H⟩ = 1.5 by hand

That is the whole mental model: **block** (what the circuit does) →
**primitive** (what you want out of it) → **engine** (run it).  Everything
else — VQE, QPE, circuit cutting, noisy simulation — is those three pieces with
more interesting parts plugged in.

**Next:** the six ``tutorial_00`` … ``tutorial_05`` notebooks in
`examples/ <https://github.com/OpenQARP/openqarp/tree/main/examples>`_
take about an hour end to end, or jump straight to the `OpenQARP Tutorial`_.

----

What's in OpenQARP
==================

.. list-table::
   :widths: 24 76
   :header-rows: 0

   * - **Algorithms**
     - `VQE`_, `VQD`_, `SSVQE`_, `ADAPT-VQE`_ / `ADAPT-VQD`_, `QAOA`_, `QPE`_,
       `DOS-QPE`_, `PCE`_ — ready to run, or assembled from primitives.
   * - **Circuits**
     - Composable blocks: hardware-efficient and QAOA ansatzes, Trotterised
       evolution, and custom blocks of your own.
   * - **Chemistry & physics**
     - Electronic structure from plain numpy integral tensors, with
       Jordan-Wigner, Bravyi-Kitaev and parity mappings.
   * - **Simulation**
     - Exact state vectors, shot-based sampling, noisy simulation with
       configurable channels, mid-circuit measurement, and circuit cutting for problems
       too big for the qubits you have.
   * - **Hardware**
     - Compile to a device's gate set and qubit layout, with noise and routing
       models — and plot any of it with Matplotlib.

----

Documentation
=============

Full documentation is on `GitHub Pages`_.

+-------------------------+------------------------------------------------------------+
| Resource                | Description                                                |
+=========================+============================================================+
| `OpenQARP Tutorial`_    | Guided path: blocks, primitives, engines, VQE loops        |
+-------------------------+------------------------------------------------------------+
| `OpenQARP Philosophy`_  | Design principles behind OpenQARP                          |
+-------------------------+------------------------------------------------------------+
| `Installation Guide`_   | Build options, GPU, LAPACK, developer setup                |
+-------------------------+------------------------------------------------------------+
| `Examples Directory`_   | ~50 runnable notebooks by API area, plus worked use cases  |
+-------------------------+------------------------------------------------------------+

----

Benchmarks
==========

OpenQARP is benchmarked against qiskit, pennylane, pytket, qulacs and cirq on
operator algebra, simulation, sampling, compilation and end-to-end algorithm
runs.  Tables are in ``benchmarks/tables/``; every timing row carries a
correctness check against an independent reference, and rows whose checks
disagree are not published.  The harness that regenerates them is
`benchmarks/README.md
<https://github.com/OpenQARP/openqarp/blob/main/benchmarks/README.md>`_.

----

Citing
======

If OpenQARP is useful in your research, please cite it.  A paper describing
the framework is in preparation; until it appears, cite the software itself
(``CITATION.cff`` carries the same entry in machine-readable form):

.. code-block:: bibtex

   @misc{openqarp2026,
     author = {Scali, Stefano and Soloviev, Vicente P. and M{\'a}rquez Romero, Antonio and
               Coyle, Brian and Buonaiuto, Giuseppe and Paine, Annie and Fetherolf, Jonathan H. and
               Diez Garc{\'i}a, Marcos and Krompiec, Michal and Kirsopp, Josh},
     title  = {{OpenQARP: Open Quantum Application Research Package}},
     year   = {2026},
     doi    = {10.5281/zenodo.22755228},
     note   = {\url{https://github.com/OpenQARP/openqarp}}
   }

----

Contributing & support
======================

Contributions are welcome — please read the `Contributing Guidelines`_ first
and the `Code of Conduct`_, which applies to every project space.  For bugs,
questions, or feature ideas, open an issue in the `Issue Tracker`_.

OpenQARP is developed and maintained by the Fujitsu Research of Europe team, and
released under the Apache License 2.0 — see ``LICENSE``.  The copyright notice
is in ``NOTICE``, which the Apache License requires you to carry forward if you
redistribute OpenQARP or a derivative work; the licenses of the bundled
third-party components are in ``LICENSES_bundled.txt``.

.. _GitHub Pages: https://openqarp.github.io/openqarp/
.. _OpenQARP Philosophy: https://openqarp.github.io/openqarp/source/qarp_philosophy.html
.. _OpenQARP Tutorial: https://openqarp.github.io/openqarp/source/tutorial.html
.. _Installation Guide: https://openqarp.github.io/openqarp/source/installation.html
.. _Examples Directory: https://github.com/OpenQARP/openqarp/tree/main/examples
.. _Contributing Guidelines: https://github.com/OpenQARP/openqarp/blob/main/CONTRIBUTING.md
.. _Code of Conduct: https://github.com/OpenQARP/openqarp/blob/main/CODE_OF_CONDUCT.md
.. _Issue Tracker: https://github.com/OpenQARP/openqarp/issues


.. _ADAPT-VQE: https://arxiv.org/abs/1812.11173
.. _ADAPT-VQD: https://arxiv.org/abs/2105.10275
.. _QPE: https://www.cambridge.org/highereducation/books/quantum-computation-and-quantum-information/01E10196D0A682A6AEFFEA52D53BE9AE#overview
.. _DOS-QPE: https://arxiv.org/abs/2510.14744
.. _PCE: https://www.nature.com/articles/s41467-024-55346-z.pdf
.. _QAOA: https://arxiv.org/pdf/1411.4028
.. _SSVQE: https://journals.aps.org/prresearch/pdf/10.1103/PhysRevResearch.1.033062
.. _VQD: https://arxiv.org/abs/1805.08138
.. _VQE: https://www.nature.com/articles/ncomms5213.pdf
