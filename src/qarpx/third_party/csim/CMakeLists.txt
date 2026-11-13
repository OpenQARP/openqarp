cmake_minimum_required(VERSION 3.20)

# ── Eigen (header-only, vendored) ──
# Vendored, not fetched: the wheels ship object code compiled from these MPL-2.0
# headers, so the corresponding source must travel with the repo and the sdist
# (MPL-2.0 §3.2(a)).  See ../eigen/README.md for the subset and its provenance.
cmake_path(SET QARP_EIGEN_DIR NORMALIZE "${CMAKE_CURRENT_SOURCE_DIR}/../eigen")
if(NOT EXISTS "${QARP_EIGEN_DIR}/Eigen/Dense")
    message(FATAL_ERROR "Vendored Eigen headers missing at ${QARP_EIGEN_DIR}")
endif()

# ── csim static library ──
file(GLOB CSIM_SRC "*.cpp")

# Exclude files that depend on cppsim (custom matrix gates — not needed).
list(REMOVE_ITEM CSIM_SRC
    "${CMAKE_CURRENT_SOURCE_DIR}/update_ops_matrix_dense_double.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/update_ops_matrix_dense_multi.cpp"
)

add_library(csim_static STATIC ${CSIM_SRC})

target_include_directories(csim_static PUBLIC
    # Allows #include "csim/foo.hpp" from files that use the csim/ prefix.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    # Allows #include "foo.hpp" from within csim sources.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${QARP_EIGEN_DIR}>
)

target_compile_features(csim_static PUBLIC cxx_std_11)
set_target_properties(csim_static PROPERTIES POSITION_INDEPENDENT_CODE ON)

# PUBLIC so every Eigen user in the build inherits it: compilation fails if a
# non-MPL2-licensed Eigen file is ever included (OSS-release license guard).
target_compile_definitions(csim_static PUBLIC EIGEN_MPL2_ONLY)

# ── OpenMP ──
# Every csim kernel already carries qulacs's `#ifdef _OPENMP` parallel path,
# including OMPutil's adaptive gating (serial below a per-gate dim threshold,
# pool size from QULACS_NUM_THREADS / OMP_NUM_THREADS).  Without this block
# none of it compiles and every kernel is silently serial — measured as flat
# 1→8-thread scaling while Aer reached 6×.  PRIVATE: only csim TUs need the
# flag; the runtime library still propagates to consumers through the static
# library's link interface.  Absent OpenMP (Apple clang without libomp) the
# build stays serial, exactly as before.
if(QARP_USE_OPENMP)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(csim_static PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "csim: OpenMP enabled (${OpenMP_CXX_VERSION})")
    else()
        message(STATUS "csim: OpenMP not found — kernels build serial")
    endif()
else()
    message(STATUS "csim: OpenMP disabled by QARP_USE_OPENMP=OFF")
endif()
