=============
Configuration
=============

This section describes OpenQARP's configuration system and type definitions, and
states what OpenQARP does under an MPI launcher.

- **Global configuration**: Centralized settings for seeds, parallelization, and circuit cutting
- **Type safety**: Clear type definitions for improved code quality
- **MPI**: detection only — see `MPI`_ below

----

Configuration System
====================

OpenQARP provides a global configuration object that controls various settings throughout the package.

The Config Object
-----------------

The configuration is accessed via the ``config`` object imported from the main package:

.. code-block:: python

   from qarp import config

This object is a singleton instance of the internal ``__ConfigObject__`` class and provides the 
following configurable properties:

Random Seed
~~~~~~~~~~~

Control random number generation for reproducibility:

.. code-block:: python

   # Set a seed for reproducible results
   config.seed = 42
   
   # Clear the seed
   config.seed = None
   
   # Get current seed
   current_seed = config.seed

Setting the seed affects:

- Python's built-in ``random`` module
- NumPy's random number generator
- SciPy (which uses NumPy's RNG internally)
- NetworkX random functions (which also use NumPy)

Maximum Number of Cuts
~~~~~~~~~~~~~~~~~~~~~~

For circuit cutting operations, limit the maximum number of cuts:

.. code-block:: python

   # Set maximum cuts (1-6, default is 6)
   config.max_number_of_cuts = 4
   
   # Get current value
   max_cuts = config.max_number_of_cuts

.. warning::

   Setting this value too high can generate an exponential number of circuits (6^n), 
   significantly increasing computational cost.

Implementation Details
~~~~~~~~~~~~~~~~~~~~~~

The configuration object is defined in ``qarp/_config.py`` and provides:

- Property-based getters and setters with validation
- Automatic propagation of settings to dependent libraries
- Runtime validation of configuration values

----

Type Definitions
================

OpenQARP defines type aliases for improved code clarity and type checking. These are defined 
in ``qarp/_types.py``.

SamplingDictionary
------------------

A type alias for dictionary structures representing sampling results:

.. code-block:: python

   from qarp import SamplingDictionary

   # SamplingDictionary is defined as: dict[tuple[int, ...], float]

   # Example usage
   results: SamplingDictionary = {
       (0, 0, 0): 0.25,
       (0, 0, 1): 0.25,
       (1, 1, 0): 0.25,
       (1, 1, 1): 0.25,
   }

The keys are tuples of integers representing measurement outcomes, and values are floats 
representing probabilities or counts.

----

Threads
=======

Two parallel layers share one knob.  The C++ simulator runs a circuit's
shots as one OpenMP team when the circuit forces per-shot trajectories
(mid-circuit measurement, reset, feed-forward, or a noise model); a
single-statevector evaluation instead lets the csim kernels parallelise
each gate across the amplitudes.  The two never nest: inside a shot worker
the kernel regions serialise, so a worker never spawns a kernel team of its
own (the contention that did cost 1.3 ms per shot on 0.02 ms of work).

``QARP_NUM_THREADS`` sizes both layers and the compilation thread pool.
It is read once, at ``import qarpx``:

.. code-block:: bash

    QARP_NUM_THREADS=4 python my_script.py

Unset, ``OMP_NUM_THREADS`` is used.  With neither set, every layer uses the
physical cores the process may run on (its CPU affinity, so ``taskset`` and
scheduler CPU binding are honoured), capped one below its logical CPUs: a
spinning OpenMP worker on each hardware thread starves the process.  Where
the topology is unreadable (outside Linux) the count is the logical CPUs
minus one.  The kernel layer keeps its own finer overrides (``QULACS_NUM_THREADS``,
``QULACS_PARALLEL_NQUBIT_THRESHOLD``, see :doc:`installation`); a
``QULACS_NUM_THREADS`` you set yourself wins over the forwarded value.
Results are seeded per shot, independent of how the shots are partitioned,
so a seeded run is bit-identical at every thread count.

The Pauli expectation kernel (``expectation`` / ``transition`` /
``batch_expectation``, what ``StateVector`` contracts with) is one OpenMP
region per call, capped by both ``QARP_NUM_THREADS`` and
``OMP_NUM_THREADS``.  Its floating-point reduction order follows the team
size, so its results agree across thread counts to rounding (about 1e-14
relative), not bit for bit.

The simulator releases the GIL while it computes (``run``, ``batch_run``,
``statevector``, ``unitary_matrix``, the expectation kernel and the gradient kernels), so Python
threads can drive it concurrently.  Below the kernel layer's parallel
threshold (16 qubits, see *Gate fusion*) two threads overlap almost fully;
above it each call already occupies every core, so threading buys little
there.

Gate fusion
===========

Every kernel call is a pass over all :math:`2^n` amplitudes, so the cost of
a statevector run is passes × state size, not gates × state size.  Before
dispatching, ``QarpSimulator`` fuses runs of gates into dense blocks of up to
``fusion_max_qubits`` qubits and applies each block in one pass — the
optimisation behind qiskit-aer's ``fusion_enable``.  It is exact (global
phase included) and applies to ``statevector``, ``run`` and ``batch_run``;
``unitary_matrix``, the adjoint gradient and noisy trajectories run gate by
gate.

.. code-block:: python

    import qarpx as qx

    sim = qx.QarpSimulator()
    sim.fusion_max_qubits          # 3: the built-in default
    sim.fusion_max_qubits = 1      # single-qubit fusion only (the 0.1.0 behaviour)
    sim.fusion_max_qubits = 0      # raw per-gate dispatch, for diagnostics
    sim.fusion_min_qubits          # 12: narrower registers keep the single-qubit pass

``QARP_FUSION_MAX_QUBITS`` sets the process-wide default for every simulator
an engine constructs — read once, at ``import qarpx``; values that do not
parse or exceed ``QarpSimulator.MAX_FUSION_QUBITS`` are ignored:

.. code-block:: bash

    QARP_FUSION_MAX_QUBITS=1 python my_script.py   # compare against unfused timings

Wider is not better: a :math:`k`-qubit block costs :math:`2^k` multiplies
per amplitude, so the default is the width measured fastest on this
kernel — at 20 qubits, single thread, brickwork 207 → 40 ms, QFT 462 → 190
ms, a Trotter step 230 → 94 ms (macOS arm64, 2026-09-13; see the
statevector benchmark table) — and the pass only widens a block past two
qubits through a shared qubit.  Below ``fusion_min_qubits`` the fold costs
more than the passes it saves, so small registers keep the single-qubit
pass.

The same import also exports ``QULACS_PARALLEL_NQUBIT_THRESHOLD=16`` to
the kernel layer unless you set it: csim's own per-kernel default of 13
forks an OpenMP team where the fork costs more than the gate, which is
where a 13-qubit circuit used to run 11× its 12-qubit twin.

MPI
===

OpenQARP does **not** implement MPI parallelism.  Earlier versions ran every rank
through the same serial optimisation in lockstep, which cost a full cluster
allocation for a single-process result; that machinery is gone.

What remains is *detection*.  ``qarp.MPIConfig`` reads the launcher's
environment variables (OpenMPI, MPICH/PMI/SLURM, Intel MPI, MVAPICH2) without
importing ``mpi4py`` — so no ``MPI_Init_thread`` ever runs inside OpenQARP — and
every composite algorithm refuses to construct under a multi-rank launcher:

.. code-block:: python

    from qarp import MPIConfig

    MPIConfig.is_mpi_env()   # True under mpirun / srun
    MPIConfig.world_size()   # ranks reported by the environment (1 when unknown)
    MPIConfig.is_disabled()  # QARP_DISABLE_MPI=1 silences detection

.. code-block:: text

    $ mpirun -np 4 python my_vqe.py
    CapabilityError: VQE: MPI parallelism is not implemented — 4 ranks detected.
    Run OpenQARP in a single process, or set QARP_DISABLE_MPI=1 on the one rank that drives it.

Set ``QARP_DISABLE_MPI=1`` when one rank of a larger job drives OpenQARP serially
on purpose.  Parallelism inside a single process (threads, the C++ batch
runners) is unaffected; see :doc:`engines`.
