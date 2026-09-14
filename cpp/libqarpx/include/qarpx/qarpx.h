#pragma once

/// @file qarpx.h
/// Umbrella header for libqarpx — OpenQARP's C++ quantum circuit IR.

// Core types
#include "core/gates.h"
#include "core/param.h"
#include "core/command.h"
#include "core/small_vector.h"

// Block hierarchy
#include "block/block.h"
#include "block/composite_block.h"
#include "block/controlled_block.h"

// Device
#include "device/architecture.h"
#include "device/noise_model.h"
#include "device/device.h"

// Compilation
#include "compilation/router.h"
#include "compilation/sabre.h"
#include "compilation/compile_for_device.h"

// Circuit DAG (wire-dependency graph over flat command streams)
#include "dag/circuit_dag.h"
#include "dag/commutation.h"
#include "dag/passes.h"

// Transpiler
#include "transpiler/gateset.h"
#include "transpiler/transpiler.h"
#include "transpiler/decompositions.h"
#include "transpiler/identities.h"
#include "transpiler/fusion.h"

// Emitters
#include "emit/emitter.h"
#include "emit/qir_emitter.h"
#include "emit/qasm3_emitter.h"
#include "emit/qasm2_emitter.h"

// Absorbers
#include "absorb/absorber.h"
#include "absorb/qasm3_absorber.h"
#include "absorb/qasm2_absorber.h"

// Parallelism
#include "parallel/thread_pool.h"
#include "parallel/mpi_utils.h"

// Simulator + Engine
#include "simulator/sampling_result.h"
#include "simulator/qarp_simulator.h"
#include "simulator/cudaq_simulator.h"
#include "engine/primitive_algorithm.h"
#include "engine/engine.h"
#include "engine/qarp_engine.h"
