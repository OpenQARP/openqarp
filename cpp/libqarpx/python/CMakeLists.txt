# nanobind itself is fetched by the parent CMakeLists (so the core library can
# use its intrusive headers), which makes the nanobind_add_module helper
# available here.  We still resolve Python's development component in this scope
# so the module + nanobind-static targets pick up the Python.h include path.
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)

nanobind_add_module(qarpx_pymod
    bindings.cpp
    bindings_operators.cpp
    ../src/emit/pennylane_emitter.cpp
    ../src/absorb/pennylane_absorber.cpp
    ../src/emit/pytket_emitter.cpp
    ../src/absorb/pytket_absorber.cpp
    ../src/emit/qulacs_emitter.cpp
    ../src/absorb/qulacs_absorber.cpp
    ../src/emit/qiskit_emitter.cpp
    ../src/absorb/qiskit_absorber.cpp
)
set_target_properties(qarpx_pymod PROPERTIES OUTPUT_NAME qarpx)
target_link_libraries(qarpx_pymod PRIVATE qarpx)

# Export only the module entry point.  The statically linked SymEngine (and
# csim/Eigen) objects otherwise leak ~2000 default-visibility symbols, and
# pytket bundles its own SymEngine: whichever extension loads first gets
# interposed into the other (pytket's `Circuit.phase` read 1.0 instead of
# 0.0 after `import qarpx`, and the absorber then segfaulted).
if(APPLE)
    target_link_options(qarpx_pymod PRIVATE "LINKER:-exported_symbol,_PyInit_qarpx")
elseif(UNIX)
    target_link_options(qarpx_pymod PRIVATE "LINKER:--exclude-libs,ALL")
endif()

# Bake the configuring checkout into the module (CMAKE_SOURCE_DIR is
# cpp/libqarpx via scikit-build's cmake.source-dir) so qarp/_abi.py can
# reject a build imported from a different checkout.
target_compile_definitions(qarpx_pymod PRIVATE
    QARPX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# CUDA-Q wheel mode: rpath to the site-packages runtime lib dirs, so the
# backend dependency chain resolves without LD_LIBRARY_PATH.
if(_qarp_cudaq_rpath)
    set_target_properties(qarpx_pymod PROPERTIES
        BUILD_RPATH   "${_qarp_cudaq_rpath}"
        INSTALL_RPATH "${_qarp_cudaq_rpath}")
endif()

# Install the compiled module to the wheel root (→ `import qarpx`).
# COMPONENT python so scikit-build-core's `install.components = ["python"]` picks
# up *only* this target and not the dev install-tree lib/headers.
install(TARGETS qarpx_pymod DESTINATION . COMPONENT python)
