============
Installation
============

`Getting Started <getting_started.html>`_ shows the two-line install that works
on most machines.  This page is everything else: what the build actually needs,
what to do when it fails, and the options for developers, GPUs and alternative
numerical backends.

----

Requirements
============

``pip install openqarp`` needs none of the below on Linux (x86_64, arm64),
macOS (Apple silicon, Intel) and Windows with Python 3.11–3.14: those
platforms get a prebuilt wheel.  The requirements apply to a **source build**
— any other platform, an editable install, or an optional backend.

* **Python 3.11+**
* a **C++20** compiler — gcc 11+, clang 14+, or Xcode 14+
* **cmake 3.20+**
* **network access and** ``git`` for the first build only

No BLAS or LAPACK is required.  OpenMP is optional and auto-detected — the
kernels multithread when it is present and build serial when it is not (see
*Multithreading (OpenMP)* below; macOS needs Homebrew's ``libomp``).

.. code-block:: bash

   # Debian / Ubuntu / WSL
   sudo apt-get update && sudo apt-get install -y build-essential cmake

   # Fedora / RHEL
   sudo dnf install -y gcc-c++ cmake

   # macOS
   brew install cmake

If your distribution's ``cmake`` is older than 3.20 (Ubuntu 20.04, for
example), install a current one with ``pip install cmake``.

The first build fetches nanobind and SymEngine at configure time (Eigen is
vendored).  Behind
a corporate proxy, export ``https_proxy`` and make sure git can reach
``github.com``.

Installing
==========

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/OpenQARP/openqarp.git
   cd openqarp

   # Checkout your desired branch
   git checkout main      # Stable release
   # git checkout develop # Latest features

   # Create and activate a virtual environment
   python -m venv .venv
   source .venv/bin/activate

   # Regular install — compiles the qarpx C++ backend automatically
   pip install .

``pip install`` drives the CMake build of the qarpx C++ backend through
**scikit-build-core** (configured in ``pyproject.toml``).  There is no separate
build step, and no ``cmake`` invocation for you to run.

Checking it worked
------------------

.. code-block:: bash

   pip show openqarp                 # package info
   pytest                            # Python test suite
   python scripts/run_cpp_tests.py   # qarpx C++ suite (rebuilds, then ctest)

.. tip::

   ``scripts/run_cpp_tests.py`` wraps ``cmake --build`` + ``ctest
   --output-on-failure``.  Useful flags: ``-k REGEX`` (filter), ``--jobs N``,
   ``--no-rebuild``, ``--rerun-failed``, ``--list``.  Anything after ``--`` goes
   straight to ctest (e.g. ``-- -V`` for verbose output).

When the build fails
====================

Out of memory
-------------

By far the most common failure.  The first build compiles SymEngine, and CMake
runs one compile job per core — about 12 jobs on a box with less than 8 GB of
RAM gets ``cc1plus`` killed by the OOM reaper.  Cap the parallelism *before*
installing:

.. code-block:: bash

   export CMAKE_BUILD_PARALLEL_LEVEL=3

Ninja resumes incrementally, so retrying after an OOM costs only the work that
had not finished.

``ModuleNotFoundError: No module named 'scikit_build_core'``
------------------------------------------------------------

You used ``--no-build-isolation`` without installing the build backend first.
See *Auto-rebuild on import* below — that flag needs both of its lines.

``import qarp`` raises even though nothing changed
--------------------------------------------------

Either an editable install opted into ``rebuild`` and lost its toolchain (see
*Auto-rebuild on import*), or you are in a git worktree sharing another
checkout's environment (see *Git worktrees*).

For developers
==============

Editable install
----------------

.. code-block:: bash

   pip install -e ".[full-dev]"   # editable + optional backends + test + dev tooling
   # pip install -e ".[dev]"      # ruff, mypy and pre-commit only — no pytest

This is the recommended default (``rebuild = false`` in ``pyproject.toml``):
Python edits are live, and a C++ edit needs the install re-run.  Nothing else
is required.

``[full-dev]`` is not every extra: ``docs``, ``notebooks``, ``integrations``,
``bench`` and ``cudaq-runtime`` are deliberately left out, each because it
pulls a heavy or platform-specific dependency that the default test run does
not need.  Install those by name when you need them.

Auto-rebuild on import
----------------------

``-C editable.rebuild=true`` makes the next ``import qarpx`` recompile just what
changed after a ``.cpp``/``.h`` edit — the up-to-date check adds about 0.1 s to
every import.  It needs **both** lines, never one:

.. code-block:: bash

   pip install scikit-build-core cmake ninja
   pip install -e ".[full-dev]" --no-build-isolation -C editable.rebuild=true

* The rebuild runs ``cmake --build`` from your environment long after pip has
  exited, so ``cmake`` and ``ninja`` must still be there.  Under build isolation
  pip provisions them into a temporary environment and deletes it, baking the
  dead path into ``build/<wheel-tag>/CMakeCache.txt``
  (``CMAKE_MAKE_PROGRAM``).  ``--no-build-isolation`` makes it use your venv's
  copies, which persist.
* That flag also stops pip honouring ``build-system.requires``, so the build
  backend must be installed by hand first — hence the first line.  Skipping it
  fails immediately, before any compilation.

.. warning::

   Half of this is worse than neither: an editable install that opted into
   ``rebuild`` but whose toolchain pip has since deleted raises on **every**
   ``import qarp``, not just after a C++ edit.

.. note::

   Under a ``rebuild`` install, export ``SKBUILD_EDITABLE_VERBOSE=0`` before
   running notebooks.  The import-time rebuild streams to the Jupyter kernel's
   stdout, which has no ``fileno()``, so every notebook otherwise dies at
   ``import qarp`` with ``UnsupportedOperation``.

Pre-commit hooks
----------------

Install the `pre-commit <https://pre-commit.com/>`_ hooks from
``.pre-commit-config.yaml`` — ruff (lint with autofix, then format),
nbstripout, a 500 KB file-size block, and a forbidden-file-types block
(``.pkl``/``.npy``/``.npz``/``.data``/``.csv``/``.json``/``.log``):

.. code-block:: bash

   pre-commit install              # one-time; runs on every `git commit`
   pre-commit run --all-files      # optional: run across the whole repo

``pre-commit`` ships in the ``[dev]`` extras.  It caches an isolated virtualenv
per hook, so tool versions are pinned in ``.pre-commit-config.yaml`` and are
independent of your active environment.

Building the documentation
--------------------------

The hosted documentation follows the ``develop`` branch. For version-specific docs,
build locally with the ``[docs]`` extra:

.. code-block:: bash

   pip install -e ".[docs]"
   sh build_docs.sh

Git worktrees
-------------

One virtual environment and one editable install **per checkout**, worktrees
included.  A shared venv pins a single checkout, so the others silently test the
wrong branch; qarp fails fast at ``import qarp`` when it detects this (ABI +
source-dir stamps).  Set ``QARP_SKIP_ABI_CHECK=1`` only for deliberate
cross-checkout runs.

Multithreading (OpenMP)
=======================

The simulation kernels multithread through OpenMP when the compiler supports
it — enabled automatically, no flag needed.  Small circuits stay
single-threaded (the kernels parallelize per gate only above a size
threshold), so threading never slows the small cases down; large statevectors
use the pool.  The configure log records the outcome either way::

   -- csim: OpenMP enabled (4.5)          # or: OpenMP not found — kernels build serial

Control the pool with ``QARP_NUM_THREADS`` (sizes the per-gate kernel pool,
the per-shot trajectory team and the compilation pool together; see
:doc:`configuration`) or the standard ``OMP_NUM_THREADS``.  Two
kernel-specific overrides also exist: ``QULACS_NUM_THREADS`` (caps the
kernel pool without affecting other OpenMP users in the process) and
``QULACS_PARALLEL_NQUBIT_THRESHOLD`` (the qubit count below which kernels run
serial; qarp exports 16 unless you set it — see :doc:`configuration`).

Per platform:

* **Linux** — works out of the box with gcc or clang; nothing to install.
* **macOS** — 13.3 (Ventura) or newer: the C++ core uses floating-point
  ``std::to_chars``, which Apple's libc++ only provides from that release,
  and the wheels are built against it.  Apple's clang ships without
  OpenMP, so the build falls back to serial kernels unless you install
  Homebrew's ``libomp`` and point CMake at it, then re-run the install:

  .. code-block:: bash

     brew install libomp
     pip install -e ".[full-dev]" \
       -C cmake.define.OpenMP_ROOT="$(brew --prefix libomp)"

* **Windows** — ``pip install openqarp`` installs a native wheel (Python
  3.11–3.14, x86-64).  Its simulator kernels are serial: MSVC's OpenMP
  runtime is a separate DLL the wheel does not bundle.  For multithreaded
  kernels use WSL, where the Linux instructions apply unchanged.  A source
  build needs Visual Studio's C++ tools; Boost headers are downloaded at
  configure time when no system Boost is found.

To force single-threaded kernels at build time (bit-stable timing baselines,
constrained CI runners), opt out with its own build tree:

.. code-block:: bash

   pip install -e ".[full-dev]" \
     -C cmake.define.QARP_USE_OPENMP=OFF -C build-dir=build-serial

At run time, ``OMP_NUM_THREADS=1`` achieves the same effect without a
separate build.

Optional backends
=================

Always pair a CMake-define override with its own ``build-dir``, so it does not
reconfigure the default build tree.

Names gated on a Python extra — ``qarp.graphs.Hypergraph`` (``[hypergraph]``),
``qarp.operators.VUMPO`` / ``qubit_operator_to_mpo`` and
``qarp.blocks.VUMPOBrickworkBlock`` (``[mps]``),
``qarp.algorithms.SpectrumEstimator`` (``[convex-optim]``) — are resolved on
first access, so importing the packages never loads hypernetx, quimb or
cvxpy.  With the extra missing, the first use raises ``ImportError`` naming
it (``hasattr(pkg, name)`` raises too; test ``name in pkg.__all__`` instead,
which lists only what is installed).

GPU execution (CUDA-Q)
----------------------

To run on GPU through ``qarp.engines.CudaqEngine``, install the CUDA-Q Python
runtime *first*, then build the qarpx backend against it.  Both steps are
required — the runtime alone does not enable GPU execution:

.. code-block:: bash

   pip install cuda-quantum-cu12               # CUDA-Q runtime wheel (CUDA 12.x, bundles cuQuantum)
   pip install scikit-build-core cmake ninja   # prerequisite of --no-build-isolation (see above)
   pip install -e . --no-build-isolation \
     -C build-dir=build-cudaq \
     -C cmake.define.QARP_WITH_CUDAQ=ON \
     -C cmake.define.QARP_CUDAQ_WHEEL_DIR="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')" \
     -C cmake.define.QARP_CUDAQ_NVQIR_BACKEND=nvqir-custatevec-fp64

Use ``-C cmake.define.QARP_CUDAQ_NVQIR_BACKEND=nvqir-qpp`` for a CPU-only CUDA-Q
build (for testing without a GPU).  The build links against the pip wheel — the
CUDA-Q **SDK is not needed**, and cannot be used: its clang/libc++ binaries do
not link into this g++ build.  GPU execution needs an NVIDIA driver supporting
CUDA 12.x.  A CPU-only build raises a clear, actionable error from
``CudaqEngine`` — use ``QarpEngine`` for CPU simulation.

LAPACK cosine-sine decomposition
--------------------------------

Unitary synthesis uses a pure-Eigen cosine-sine decomposition by default, with
no BLAS or LAPACK involved.  To route it through LAPACK's ``zuncsd`` instead —
which gives canonical, bit-reproducible decompositions on degenerate spectra —
install a BLAS (``libopenblas-dev liblapack-dev`` / ``openblas-devel
lapack-devel``; macOS uses the built-in Accelerate) and add:

.. code-block:: bash

   pip install -e ".[full-dev]" \
     -C cmake.define.QARP_USE_LAPACK=ON -C build-dir=build-lapack

Boost headers and the symbolic backend
--------------------------------------

The source build needs **Boost headers** at build time only, for the SymEngine
symbolic backend.  A system install is picked up automatically
(``apt install libboost-dev`` / ``dnf install boost-devel`` /
``brew install boost``); without one, the build **downloads the pinned Boost
headers itself** (~125 MB, once per build tree) — no action needed.

To opt out of the symbolic backend entirely, add
``-C cmake.define.QARP_WITH_SYMENGINE=OFF``, with its own ``build-dir``.

Legacy build path
=================

``scripts/bootstrap_qarpx.py`` predates the pip build (CMake build under
``cpp/libqarpx/build/`` plus a ``.pth`` file in site-packages).  It still works,
but only where qarp was *not* pip-installed: any pip install of qarp ships its
own ``qarpx`` module, which silently shadows the ``.pth``-registered build.  Its
CMake defines mirror the wheel's, so both paths build the same backend —
override them with ``--lapack`` / ``--no-symengine``.

``scripts/bootstrap_qarpx_cudaq.py`` is the CUDA-Q equivalent, for those
environments only.
