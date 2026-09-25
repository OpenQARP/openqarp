cmake_minimum_required(VERSION 3.20)
project(qarpx LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Options ──
option(QARP_WITH_MPI      "Enable MPI support"        OFF)
option(QARP_WITH_CUDAQ    "Enable CUDA-Q / cuQuantum GPU backend (CudaqSimulator)" OFF)
option(QARP_BUILD_TESTS   "Build C++ tests"            ON)
option(QARP_BUILD_PYTHON  "Build Python bindings"     OFF)
option(QARP_USE_LAPACK    "Link LAPACK for the CSD (zuncsd) instead of the pure-Eigen path" OFF)
option(QARP_USE_OPENMP    "Compile the csim kernels with OpenMP (auto-off if unavailable)" ON)
option(QARP_WITH_LAPACKE  "Route Eigen SVD/Schur through LAPACK (EIGEN_USE_LAPACKE)" OFF)
option(QARP_WITH_SYMENGINE "SymEngine symbolic coefficient backend for the operator classes" OFF)
option(QARP_FETCH_BOOST   "Skip the system Boost lookup and download the pinned headers (auto-fallback when no system Boost)" OFF)
option(QARP_ENABLE_COVERAGE "Instrument for gcov/lcov coverage (GCC/Clang only; Debug-ish, do NOT ship)" OFF)

# Floating-point std::to_chars (shortest round-trip parameter text) is
# unavailable in Apple's libc++ below macOS 13.3 — fail here, not 120 TUs in.
if(APPLE AND CMAKE_OSX_DEPLOYMENT_TARGET AND CMAKE_OSX_DEPLOYMENT_TARGET VERSION_LESS 13.3)
    message(FATAL_ERROR "qarpx needs MACOSX_DEPLOYMENT_TARGET >= 13.3 "
        "(got ${CMAKE_OSX_DEPLOYMENT_TARGET}): floating-point std::to_chars is "
        "unavailable in older Apple libc++ deployment targets.")
endif()

# MSVC hides M_PI behind _USE_MATH_DEFINES (synthesis/*.cpp, absorb/*.cpp).
# Empty value on purpose: csim's constant.hpp defines it empty too, and MSVC
# warns on a redefinition with a different value.
if(MSVC)
    add_compile_definitions(_USE_MATH_DEFINES=)
endif()

# ── Coverage instrumentation ──
# Applied globally rather than per-target so the vendored csim and the test
# binaries carry the same flags; gcovr filters third_party/ and tests/ back out
# at report time.  -O0 keeps line attribution honest: at -O2 the optimiser
# merges and reorders lines, so gcov reports coverage against source lines that
# no longer correspond to the emitted code.
if(QARP_ENABLE_COVERAGE)
    if(NOT (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang"))
        message(FATAL_ERROR "QARP_ENABLE_COVERAGE needs GCC or Clang, got ${CMAKE_CXX_COMPILER_ID}")
    endif()
    add_compile_options(--coverage -O0 -g)
    add_link_options(--coverage)
endif()

# ── csim (qulacs simulation kernel, vendored) ──
add_subdirectory(third_party/csim)

# ── nanobind ──
# Block derives from nb::intrusive_base so its C++ lifetime is shared with its
# Python wrapper via one reference count.  Only the header-only intrusive counter
# is needed by the core library (no Python/runtime dependency); fetch nanobind
# here so those headers are available regardless of QARP_BUILD_PYTHON.
include(FetchContent)
FetchContent_Declare(
    nanobind
    GIT_REPOSITORY https://github.com/wjakob/nanobind.git
    GIT_TAG        v3.0.1
)
FetchContent_MakeAvailable(nanobind)

# ── Core library ──
add_library(qarpx
    src/core/param.cpp
    src/core/command.cpp
    src/core/canonical.cpp
    src/core/pauli.cpp
    src/core/intrusive_counter.cpp
    src/operators/packed_pauli.cpp
    src/operators/qubit_operator.cpp
    src/operators/fermion_operator.cpp
    src/operators/transforms.cpp
    src/operators/sparse.cpp
    src/operators/symbolic.cpp
    src/block/block.cpp
    src/block/composite_block.cpp
    src/block/controlled_block.cpp
    src/block/measure_block.cpp
    src/block/reset_block.cpp
    src/block/conditional_block.cpp
    src/device/architecture.cpp
    src/device/noise_model.cpp
    src/device/device.cpp
    src/compilation/router.cpp
    src/compilation/sabre.cpp
    src/compilation/perfect_layout.cpp
    src/compilation/compile_for_device.cpp
    src/dag/circuit_dag.cpp
    src/dag/commutation.cpp
    src/dag/passes.cpp
    src/transpiler/gateset.cpp
    src/transpiler/decompositions.cpp
    src/transpiler/identities.cpp
    src/transpiler/fusion.cpp
    src/transpiler/transpiler.cpp
    src/emit/qir_emitter.cpp
    src/emit/qasm3_emitter.cpp
    src/emit/qasm2_emitter.cpp
    src/absorb/qasm3_absorber.cpp
    src/absorb/qasm2_absorber.cpp
    src/synthesis/uniformly_controlled.cpp
    src/synthesis/state_preparation.cpp
    src/synthesis/diagonal.cpp
    src/synthesis/unitary.cpp
    src/synthesis/pauli_exponential.cpp
    src/parallel/cpu_budget.cpp
    src/parallel/thread_pool.cpp
    src/parallel/mpi_utils.cpp
    src/simulator/dense_kernel.cpp
    src/simulator/fusion.cpp
    src/simulator/pauli_expectation.cpp
    src/simulator/qarp_simulator.cpp
    # Throwing stubs with available()==false unless QARP_WITH_CUDAQ is set
    # (see below); no CUDA dependency in the default build.
    src/simulator/cudaq_simulator.cpp
)

target_include_directories(qarpx PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# nanobind intrusive headers — header-only, no Python/runtime dependency.
# PUBLIC so downstream targets that include block.h also see them.
target_include_directories(qarpx PUBLIC ${nanobind_SOURCE_DIR}/include)

# ── Optional: SymEngine symbolic coefficient backend ──
# v0.14.0 (MIT) with the Boost.Multiprecision integer class: MIT + BSL-1.0
# only — no GMP/MPFR (LGPL) anywhere in shipped wheels.  Boost is a
# build-time, header-only requirement: a system install is preferred; when
# none is found the pinned release headers are downloaded into the build
# tree (-DQARP_FETCH_BOOST=ON forces the download path).
# When OFF the operator classes are numeric-only and symbolic coefficients
# raise at the Python boundary.
if(QARP_WITH_SYMENGINE)
    if(NOT QARP_FETCH_BOOST)
        # CONFIG mode — the FindBoost module is gone in recent CMake (CMP0167).
        find_package(Boost QUIET CONFIG)
    endif()
    if(NOT Boost_FOUND)
        message(STATUS "No system Boost — downloading Boost 1.88.0 headers "
                       "(header-only, build-time only, nothing bundled)")
        FetchContent_Declare(
            boost_headers
            URL https://archives.boost.io/release/1.88.0/source/boost_1_88_0.tar.bz2
            URL_HASH SHA256=46d9d2c06637b219270877c9e16155cbd015b6dc84349af064c088e9b5b12f7b
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(boost_headers)
        # The classic release archive keeps the merged boost/ header tree at
        # its root. Seeding Boost_INCLUDE_DIR satisfies symengine's
        # module-mode find_package(Boost) (its CMP0167 is OLD, min 3.13).
        set(Boost_INCLUDE_DIR ${boost_headers_SOURCE_DIR} CACHE PATH "" FORCE)
        add_library(Boost::headers INTERFACE IMPORTED GLOBAL)
        set_target_properties(Boost::headers PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES ${boost_headers_SOURCE_DIR})
    endif()
    # NB: symengine's cache variable really is the generic `INTEGER_CLASS`.
    # boostmp is the license-critical setting — never let this fall back to
    # gmp (LGPL); the guard below hard-fails if it does.
    set(INTEGER_CLASS "boostmp" CACHE STRING "SymEngine bignum backend" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)        # symengine's generic option names
    set(BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        symengine
        GIT_REPOSITORY https://github.com/symengine/symengine.git
        GIT_TAG        v0.14.0
    )
    FetchContent_MakeAvailable(symengine)
    file(READ ${symengine_BINARY_DIR}/symengine/symengine_config.h _qarp_se_config)
    if(NOT _qarp_se_config MATCHES "SYMENGINE_INTEGER_CLASS SYMENGINE_BOOSTMP")
        message(FATAL_ERROR
            "SymEngine did not configure with the boostmp integer class — it "
            "would link GMP/MPFR (LGPL) into a proprietary wheel. Install "
            "Boost headers (brew/dnf boost, or -DBOOST_ROOT=…) and reconfigure.")
    endif()
    # Boost headers for our own TUs too (symengine's Boost include dirs are
    # private to its target).
    target_link_libraries(qarpx PUBLIC symengine Boost::headers)
    target_include_directories(qarpx PUBLIC
        ${symengine_SOURCE_DIR} ${symengine_BINARY_DIR})
    target_compile_definitions(qarpx PUBLIC QARP_WITH_SYMENGINE)
endif()

# Threads (required for thread pool)
find_package(Threads REQUIRED)
target_link_libraries(qarpx PUBLIC Threads::Threads csim_static)

# OpenMP for qarpx's own shot-level parallelism (qarp_simulator.cpp).  The
# trajectory loop runs as an OpenMP team so csim's per-gate regions nest
# inside it and serialise; a second, foreign thread pool contending with
# csim's OpenMP workers cost 15-70× per shot.  Serial csim ⇒ serial here.
if(QARP_USE_OPENMP)
    find_package(OpenMP QUIET)  # csim's find is scoped to its subdirectory
    if(OpenMP_CXX_FOUND)
        target_link_libraries(qarpx PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "qarpx: OpenMP shot-level parallelism enabled")
    endif()
endif()

# ── Optional: MPI ──
if(QARP_WITH_MPI)
    find_package(MPI REQUIRED)
    target_link_libraries(qarpx PUBLIC MPI::MPI_CXX)
    target_compile_definitions(qarpx PUBLIC QARP_WITH_MPI)
endif()

# ── Optional: CUDA-Q / cuQuantum GPU backend ──
# When ON, link the CUDA-Q runtime and switch cudaq_simulator.cpp to its real
# GPU paths.  Two link modes: a pip wheel (QARP_CUDAQ_WHEEL_DIR) or the full
# SDK (find_package).
if(QARP_WITH_CUDAQ)
    target_compile_definitions(qarpx PUBLIC QARP_WITH_CUDAQ)
    target_link_libraries(qarpx PUBLIC ${CMAKE_DL_LIBS})  # dlopen in cudaq_simulator.cpp
    if(QARP_CUDAQ_WHEEL_DIR)
        # ── Wheel mode ──
        # Link the CUDA-Q runtime from a pip-installed cuda-quantum wheel
        # (no find_package/nvq++).  QARP_CUDAQ_WHEEL_DIR = site-packages root.
        message(STATUS "qarpx: CUDA-Q wheel mode from ${QARP_CUDAQ_WHEEL_DIR}")
        # Enables runtime backend selection/switching (cudaq_simulator.cpp).
        target_compile_definitions(qarpx PRIVATE QARP_CUDAQ_WHEEL_MODE)
        target_include_directories(qarpx PRIVATE ${QARP_CUDAQ_WHEEL_DIR}/include)
        target_link_directories(qarpx PUBLIC ${QARP_CUDAQ_WHEEL_DIR}/lib)
        # GPU backends need cuStateVec/cuTensorNet (pip cuquantum) and cuBLAS
        # (pip nvidia-cublas) at runtime; link dirs also feed the build rpath.
        set(_qarp_cudaq_rpath "${QARP_CUDAQ_WHEEL_DIR}/lib")
        if(EXISTS "${QARP_CUDAQ_WHEEL_DIR}/cuquantum/lib")
            target_link_directories(qarpx PUBLIC "${QARP_CUDAQ_WHEEL_DIR}/cuquantum/lib")
            list(APPEND _qarp_cudaq_rpath "${QARP_CUDAQ_WHEEL_DIR}/cuquantum/lib")
        endif()
        file(GLOB _qarp_nvidia_libdirs "${QARP_CUDAQ_WHEEL_DIR}/nvidia/*/lib")
        if(_qarp_nvidia_libdirs)
            target_link_directories(qarpx PUBLIC ${_qarp_nvidia_libdirs})
            list(APPEND _qarp_cudaq_rpath ${_qarp_nvidia_libdirs})
        endif()
        # Link one NVQIR backend so the runtime libs resolve at import; the
        # backend actually used is dlopen'd per cfg.target at execution.
        set(QARP_CUDAQ_NVQIR_BACKEND "nvqir-qpp" CACHE STRING
            "NVQIR backend in wheel mode: nvqir-qpp (CPU) or nvqir-custatevec-fp64 (GPU)")
        message(STATUS "qarpx: wheel-mode NVQIR backend = ${QARP_CUDAQ_NVQIR_BACKEND}")
        # DT_RPATH (not DT_RUNPATH): must also resolve the backends' indirect
        # deps.  _qarp_cudaq_rpath lands on the module in python/CMakeLists.txt.
        target_link_libraries(qarpx PUBLIC
            -Wl,--disable-new-dtags
            -Wl,--no-as-needed
            cudaq cudaq-builder cudaq-mlir-runtime cudaq-common
            cudaq-em-default cudaq-operator
            cudaq-platform-default nvqir ${QARP_CUDAQ_NVQIR_BACKEND}
            -Wl,--as-needed)
    else()
        # ── SDK mode ──
        # Force library mode: the default find_package(CUDAQ) path runs
        # enable_language(CUDAQ), which can't build library targets.  Library
        # mode compiles the .cpp as ordinary C++ and links the CUDA-Q runtime.
        set(CUDAQ_LIBRARY_MODE ON)
        find_package(CUDAQ REQUIRED)
        # The SDK ships MLIR as static archives referenced by bare name (e.g.
        # -lMLIRIR); put LLVM's lib dir on the link path (explicit override, else
        # the SDK's conventional LLVM_INSTALL_PREFIX).
        if(QARP_CUDAQ_MLIR_LIB_DIR)
            set(_qarp_mlir_libdir "${QARP_CUDAQ_MLIR_LIB_DIR}")
        elseif(DEFINED ENV{LLVM_INSTALL_PREFIX})
            set(_qarp_mlir_libdir "$ENV{LLVM_INSTALL_PREFIX}/lib")
        endif()
        if(_qarp_mlir_libdir AND EXISTS "${_qarp_mlir_libdir}")
            target_link_directories(qarpx PUBLIC "${_qarp_mlir_libdir}")
            message(STATUS "qarpx: MLIR/LLVM lib dir on link path — ${_qarp_mlir_libdir}")
        else()
            message(STATUS "qarpx: no MLIR/LLVM lib dir resolved; if linking fails "
                "with 'cannot find -lMLIRIR', pass -DQARP_CUDAQ_MLIR_LIB_DIR=<llvm-prefix>/lib")
        endif()
        target_link_libraries(qarpx PUBLIC
            cudaq::cudaq cudaq::cudaq-builder cudaq::cudaq-em-default)
    endif()
endif()

# ── LAPACK (optional) ──
# Synthesis (src/synthesis/unitary.cpp) can route the CSD step through LAPACK's
# zuncsd_ instead of the default pure-Eigen path.  OFF by default so the build
# — and above all the PyPI wheels — link no BLAS at all (no OpenBLAS/
# libgfortran/libquadmath to bundle); ON buys canonical degenerate-σ basis
# resolution (bit-reproducible decompositions across toolchains).
# macOS: Accelerate.framework (the source redirects zuncsd_ to its NEWLAPACK
# symbol).  Linux: find_package(LAPACK) (OpenBLAS or reference LAPACK).
if(QARP_USE_LAPACK)
    if(APPLE)
        target_link_libraries(qarpx PUBLIC "-framework Accelerate")
        message(STATUS "qarpx: LAPACK enabled via Apple Accelerate.framework")
    else()
        find_package(LAPACK REQUIRED)
        target_link_libraries(qarpx PUBLIC ${LAPACK_LIBRARIES})
        message(STATUS "qarpx: LAPACK enabled — ${LAPACK_LIBRARIES}")
    endif()
    target_compile_definitions(qarpx PUBLIC QARP_USE_LAPACK)
    if(QARP_WITH_LAPACKE)
        target_compile_definitions(qarpx PUBLIC EIGEN_USE_LAPACKE)
        message(STATUS "qarpx: EIGEN_USE_LAPACKE on (Eigen SVD/Schur via LAPACK)")
    endif()
endif()

# ── Tests ──
if(QARP_BUILD_TESTS)
    enable_testing()

    # Fetch GoogleTest
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.15.2
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests/cpp)
endif()

# ── Python bindings ──
if(QARP_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# ── Install ──
#
# Dev / install-tree only.  Under a scikit-build-core wheel build (SKBUILD is set)
# the static core lib and headers must NOT land in the wheel — the wheel ships
# only the nanobind module (python/CMakeLists.txt, COMPONENT python).  Gating here
# is belt-and-suspenders with `install.components = ["python"]` in pyproject.toml.
if(NOT SKBUILD)
    install(TARGETS qarpx
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
    )
    install(DIRECTORY include/qarpx DESTINATION include)
endif()
