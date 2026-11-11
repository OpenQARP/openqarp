#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/complex.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "qarpx/montecarlo/contexts.h"
#include "qarpx/simulator/transition_columns.h"
#include "qarpx/simulator/cudaq_simulator.h"

#include <new>
#include <optional>
#include <stdexcept>

namespace nb = nanobind;
using namespace nb::literals;
namespace mc = qarpx::montecarlo;
namespace sim = qarpx::simulator;

namespace {
struct UInt64 {
    std::uint64_t value{};
    operator std::uint64_t() const { return value; }
};
struct Int64 {
    std::int64_t value{};
    operator std::int64_t() const { return value; }
};
}
namespace nanobind::detail {
template<> struct type_caster<UInt64> {
    NB_TYPE_CASTER(UInt64, const_name("int"))
    bool from_python(handle source, uint8_t, cleanup_list*) noexcept {
        if (!PyLong_Check(source.ptr()) || PyBool_Check(source.ptr())) return false;
        value.value=PyLong_AsUnsignedLongLong(source.ptr());
        if(PyErr_Occurred()) { PyErr_Clear(); return false; }
        return true;
    }
    static handle from_cpp(UInt64 value, rv_policy, cleanup_list*) noexcept {
        return handle(PyLong_FromUnsignedLongLong(value.value));
    }
};
template<> struct type_caster<Int64> {
    NB_TYPE_CASTER(Int64, const_name("int"))
    bool from_python(handle source, uint8_t, cleanup_list*) noexcept {
        if (!PyLong_Check(source.ptr()) || PyBool_Check(source.ptr())) return false;
        value.value=PyLong_AsLongLong(source.ptr());
        if(PyErr_Occurred()) { PyErr_Clear(); return false; }
        return true;
    }
    static handle from_cpp(Int64 value, rv_policy, cleanup_list*) noexcept {
        return handle(PyLong_FromLongLong(value.value));
    }
};
}
namespace {
template <class T>
using Input = nb::ndarray<const T, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

template <class T>
std::vector<T> copy_array(const Input<T>& array) {
    if (array.size() == 0) return {};
    return {array.data(), array.data() + array.size()};
}
template <class T>
std::vector<T> copy_array(const std::optional<Input<T>>& array) {
    return array ? copy_array(*array) : std::vector<T>{};
}

template <class Record, class T>
void array_property(nb::class_<Record>& cls, const char* name,
                    std::vector<T> Record::* member) {
    cls.def_prop_ro(name, [member](nb::handle self) {
        const auto& values = nb::cast<const Record&>(self).*member;
        // Python exposes immutable records: vectors cannot resize beneath a view.
        auto owner = std::make_unique<nb::object>(nb::borrow<nb::object>(self));
        nb::capsule capsule(owner.get(), [](void* ptr) noexcept {
            nb::gil_scoped_acquire guard;
            if (guard.is_valid()) delete static_cast<nb::object*>(ptr);
        });
        owner.release();
        const std::size_t shape[] = {values.size()};
        return nb::ndarray<nb::numpy, const T, nb::ndim<1>>(
            values.data(), 1, shape, capsule);
    }, nb::rv_policy::reference);
}

template <class T>
nb::ndarray<nb::numpy, T> owned_array(std::vector<T>&& values,
                                    std::initializer_list<std::size_t> shape) {
    auto owner = std::make_unique<std::vector<T>>(std::move(values));
    nb::capsule capsule(owner.get(), [](void* ptr) noexcept {
        delete static_cast<std::vector<T>*>(ptr);
    });
    auto* data = owner->data();
    owner.release();
    return nb::ndarray<nb::numpy, T>(data, shape.size(), shape.begin(), capsule);
}

nb::dict diagnostics(const sim::Diagnostics& d) {
    nb::dict out;
    out["column_batches"] = d.column_batches;
    out["columns"] = d.columns;
    out["capacity_growths"] = d.capacity_growths;
    out["handle_creations"] = d.handle_creations;
    out["term_uploads"] = d.term_uploads;
    out["gate_uploads"] = d.gate_uploads;
    out["transferred_bytes"] = d.transferred_bytes;
    out["compact_entries"] = d.compact_entries;
    out["pruned_entries"] = d.pruned_entries;
    out["sector_omitted_entries"] = d.sector_omitted_entries;
    out["sector_target_count"] = d.sector_target_count;
    out["capacity"] = d.capacity;
    out["state_buffer_bytes"] = d.state_buffer_bytes;
    out["workspace_bytes"] = d.workspace_bytes;
    out["allocation_wall_seconds"] = d.allocation_wall_seconds;
    out["basis_wall_seconds"] = d.basis_wall_seconds;
    out["u_wall_seconds"] = d.u_wall_seconds;
    out["hamiltonian_wall_seconds"] = d.hamiltonian_wall_seconds;
    out["u_dagger_wall_seconds"] = d.u_dagger_wall_seconds;
    out["transfer_wall_seconds"] = d.transfer_wall_seconds;
    out["sector_gather_wall_seconds"] = d.sector_gather_wall_seconds;
    out["compaction_wall_seconds"] = d.compaction_wall_seconds;
    out["max_pruned_amplitude"] = d.max_pruned_amplitude;
    return out;
}

nb::dict diagnostics(const mc::Diagnostics& d) {
    nb::dict out;
    out["handle_creations"] = d.handle_creations;
    out["state_buffer_bytes"] = d.state_buffer_bytes;
    out["gate_uploads"] = d.gate_uploads;
    out["term_uploads"] = d.term_uploads;
    out["aggregated_cache_uploads"] = d.aggregated_cache_uploads;
    out["aggregated_cache_appended_rows"] = d.aggregated_cache_appended_rows;
    out["aggregated_cache_appended_edges"] = d.aggregated_cache_appended_edges;
    out["cache_rows"] = d.cache_rows;
    out["aggregated_cached_edges"] = d.aggregated_cached_edges;
    out["aggregated_cache_edge_capacity"] = d.aggregated_cache_edge_capacity;
    out["aggregated_cache_capacity_growths"] = d.aggregated_cache_capacity_growths;
    out["aggregated_population_capacity"] = d.aggregated_population_capacity;
    out["aggregated_population_capacity_growths"] = d.aggregated_population_capacity_growths;
    out["aggregated_cache_bytes"] = d.aggregated_cache_bytes;
    out["aggregated_cache_capacity_bytes"] = d.aggregated_cache_capacity_bytes;
    out["aggregated_population_bytes"] = d.aggregated_population_bytes;
    out["aggregated_cache_uploaded_bytes"] = d.aggregated_cache_uploaded_bytes;
    out["aggregated_cache_metadata_uploaded_bytes"] = d.aggregated_cache_metadata_uploaded_bytes;
    out["aggregated_cache_growth_copy_bytes"] = d.aggregated_cache_growth_copy_bytes;
    out["aggregated_host_transfer_bytes"] = d.aggregated_host_transfer_bytes;
    out["aggregated_steps"] = d.aggregated_steps;
    out["aggregated_transition_edge_visits"] = d.aggregated_transition_edge_visits;
    out["aggregated_binomial_calls"] = d.aggregated_binomial_calls;
    out["aggregated_cache_upload_wall_seconds"] = d.aggregated_cache_upload_wall_seconds;
    out["aggregated_kernel_wall_seconds"] = d.aggregated_kernel_wall_seconds;
    out["transition_requests"] = d.transition_requests;
    out["transition_batches"] = d.transition_batches;
    out["transition_edges"] = d.transition_edges;
    out["transition_proposals"] = d.transition_proposals;
    out["transition_uploaded_bytes"] = d.transition_uploaded_bytes;
    out["transition_upload_wall_seconds"] = d.transition_upload_wall_seconds;
    out["transition_kernel_wall_seconds"] = d.transition_kernel_wall_seconds;
    return out;
}

mc::TransitionBatch batch_from_arrays(
    std::uint64_t request_id, std::uint64_t trajectory_index,
    std::uint64_t step_index, std::uint64_t attempts_per_walker,
    const Input<std::uint64_t>& sources, const Input<std::uint64_t>& targets,
    const Input<std::uint64_t>& multiplicities, const Input<double>& probabilities,
    const Input<mc::Coefficient>& coefficients,
    const Input<std::uint64_t>& diagonal_labels, const Input<double>& diagonal_elements,
    const Input<std::uint64_t>& reference_labels, const Input<double>& reference_elements) {
    if (targets.size() != sources.size() || multiplicities.size() != sources.size() ||
        probabilities.size() != sources.size() || coefficients.size() != sources.size() ||
        diagonal_labels.size() != diagonal_elements.size() ||
        reference_labels.size() != reference_elements.size())
        throw std::invalid_argument("TransitionBatch parallel arrays must have matching lengths");
    return {request_id, trajectory_index, step_index, attempts_per_walker,
            copy_array(sources), copy_array(targets), copy_array(multiplicities),
            copy_array(probabilities), copy_array(coefficients),
            copy_array(diagonal_labels), copy_array(diagonal_elements),
            copy_array(reference_labels), copy_array(reference_elements)};
}
}  // namespace

namespace qarpx::python {
void register_montecarlo_bindings(nb::module_& m) {
    nb::enum_<mc::EvolutionStatus>(m, "EvolutionStatus")
        .value("Completed", mc::EvolutionStatus::Completed)
        .value("NeedsTransitionRows", mc::EvolutionStatus::NeedsTransitionRows)
        .value("ReferencePopulationZero", mc::EvolutionStatus::ReferencePopulationZero)
        .value("PopulationExtinct", mc::EvolutionStatus::PopulationExtinct)
        .value("IntegerOverflow", mc::EvolutionStatus::IntegerOverflow)
        .value("InvalidProbability", mc::EvolutionStatus::InvalidProbability)
        .value("InvalidTransitionBatch", mc::EvolutionStatus::InvalidTransitionBatch)
        .value("CapacityExceeded", mc::EvolutionStatus::CapacityExceeded);

    auto request = nb::class_<mc::TransitionRequest>(m, "TransitionRequest")
        .def("__init__", [](mc::TransitionRequest* self, UInt64 id,
                            UInt64 trajectory, UInt64 step,
                            UInt64 attempts, const Input<std::uint64_t>& sources,
                            const Input<std::int64_t>& populations) {
            if (sources.size() != populations.size())
                throw std::invalid_argument("TransitionRequest sources/populations lengths differ");
            new (self) mc::TransitionRequest{id, trajectory, step, attempts,
                                             copy_array(sources), copy_array(populations), {}};
        }, "request_id"_a, "trajectory_index"_a, "step_index"_a, "attempts_per_walker"_a,
           "sources"_a.noconvert(), "populations"_a.noconvert())
        .def_ro("request_id", &mc::TransitionRequest::request_id)
        .def_ro("trajectory_index", &mc::TransitionRequest::trajectory_index)
        .def_ro("step_index", &mc::TransitionRequest::step_index)
        .def_ro("attempts_per_walker", &mc::TransitionRequest::attempts_per_walker);
    array_property(request, "sources", &mc::TransitionRequest::sources);
    array_property(request, "populations", &mc::TransitionRequest::populations);
    array_property(request, "proposal_attempts", &mc::TransitionRequest::proposal_attempts);

    auto batch = nb::class_<mc::TransitionBatch>(m, "TransitionBatch")
        .def("__init__", [](mc::TransitionBatch* self,
                            UInt64 id, UInt64 trajectory,
                            UInt64 step, UInt64 attempts,
                            const Input<std::uint64_t>& sources, const Input<std::uint64_t>& targets,
                            const Input<std::uint64_t>& multiplicities, const Input<double>& probabilities,
                            const Input<mc::Coefficient>& coefficients,
                            const Input<std::uint64_t>& diagonal_labels, const Input<double>& diagonals,
                            const Input<std::uint64_t>& reference_labels, const Input<double>& references) {
            new (self) mc::TransitionBatch(batch_from_arrays(
                id, trajectory, step, attempts, sources, targets, multiplicities,
                probabilities, coefficients, diagonal_labels, diagonals, reference_labels, references));
        }, "request_id"_a, "trajectory_index"_a, "step_index"_a, "attempts_per_walker"_a,
           "sources"_a.noconvert(), "targets"_a.noconvert(), "multiplicities"_a.noconvert(),
           "proposal_probabilities"_a.noconvert(), "transition_coefficients"_a.noconvert(),
           "diagonal_labels"_a.noconvert(), "diagonal_elements"_a.noconvert(),
           "reference_labels"_a.noconvert(), "reference_elements"_a.noconvert())
        .def_ro("request_id", &mc::TransitionBatch::request_id)
        .def_ro("trajectory_index", &mc::TransitionBatch::trajectory_index)
        .def_ro("step_index", &mc::TransitionBatch::step_index)
        .def_ro("attempts_per_walker", &mc::TransitionBatch::attempts_per_walker);
    array_property(batch, "sources", &mc::TransitionBatch::sources);
    array_property(batch, "targets", &mc::TransitionBatch::targets);
    array_property(batch, "multiplicities", &mc::TransitionBatch::multiplicities);
    array_property(batch, "proposal_probabilities", &mc::TransitionBatch::proposal_probabilities);
    array_property(batch, "transition_coefficients", &mc::TransitionBatch::transition_coefficients);
    array_property(batch, "diagonal_labels", &mc::TransitionBatch::diagonal_labels);
    array_property(batch, "diagonal_elements", &mc::TransitionBatch::diagonal_elements);
    array_property(batch, "reference_labels", &mc::TransitionBatch::reference_labels);
    array_property(batch, "reference_elements", &mc::TransitionBatch::reference_elements);

    auto result = nb::class_<mc::EvolutionResult>(m, "EvolutionResult")
        .def("__init__", [](mc::EvolutionResult* self, mc::EvolutionStatus status,
                            UInt64 steps, const std::optional<Input<std::uint64_t>>& missing,
                            const std::optional<Input<std::uint64_t>>& population,
                            const std::optional<Input<std::int64_t>>& reference,
                            const std::optional<Input<double>>& shift,
                            const std::optional<Input<double>>& energy,
                            const std::optional<Input<double>>& numerator,
                            const std::optional<Input<std::uint64_t>>& spawned,
                            const std::optional<Input<std::uint64_t>>& died,
                            const std::optional<Input<std::uint64_t>>& cloned,
                            const std::optional<Input<std::uint64_t>>& annihilated) {
            new (self) mc::EvolutionResult{status, steps, copy_array(missing),
                copy_array(population), copy_array(reference), copy_array(shift),
                copy_array(energy), copy_array(numerator), copy_array(spawned),
                copy_array(died), copy_array(cloned), copy_array(annihilated)};
        }, "status"_a = mc::EvolutionStatus::Completed, "completed_steps"_a = 0,
           "missing_labels"_a.noconvert() = nb::none(),
           "population_history"_a.noconvert() = nb::none(),
           "reference_population_history"_a.noconvert() = nb::none(),
           "shift_history"_a.noconvert() = nb::none(),
           "projected_energy_history"_a.noconvert() = nb::none(),
           "projected_numerator_history"_a.noconvert() = nb::none(),
           "spawned_history"_a.noconvert() = nb::none(), "died_history"_a.noconvert() = nb::none(),
           "cloned_history"_a.noconvert() = nb::none(), "annihilated_history"_a.noconvert() = nb::none())
        .def_ro("status", &mc::EvolutionResult::status)
        .def_ro("completed_steps", &mc::EvolutionResult::completed_steps);
    array_property(result, "missing_labels", &mc::EvolutionResult::missing_labels);
    array_property(result, "population_history", &mc::EvolutionResult::population_history);
    array_property(result, "reference_population_history", &mc::EvolutionResult::reference_population_history);
    array_property(result, "shift_history", &mc::EvolutionResult::shift_history);
    array_property(result, "projected_energy_history", &mc::EvolutionResult::projected_energy_history);
    array_property(result, "projected_numerator_history", &mc::EvolutionResult::projected_numerator_history);
    array_property(result, "spawned_history", &mc::EvolutionResult::spawned_history);
    array_property(result, "died_history", &mc::EvolutionResult::died_history);
    array_property(result, "cloned_history", &mc::EvolutionResult::cloned_history);
    array_property(result, "annihilated_history", &mc::EvolutionResult::annihilated_history);
    array_property(result, "active_label_history", &mc::EvolutionResult::active_label_history);
    array_property(result, "transition_edge_history", &mc::EvolutionResult::transition_edge_history);

    nb::class_<mc::ResidentEvolutionContext>(m, "ResidentEvolutionContext")
        .def("reset_aggregated_evolution", [](mc::ResidentEvolutionContext& c,
            UInt64 reference, Int64 population, double shift, double energy, double dt,
            double damping, double threshold, UInt64 seed, UInt64 trajectory) {
            nb::gil_scoped_release release;
            c.reset_aggregated_evolution(reference,population,shift,energy,dt,damping,threshold,seed,trajectory);
        }, "reference_label"_a,"initial_population"_a,"initial_energy_shift"_a,
           "initial_projected_energy"_a,"dt"_a,"damping"_a,"threshold"_a,
           "root_seed"_a,"trajectory_index"_a)
        .def("upload_aggregated_transition_rows", [](mc::ResidentEvolutionContext& c,
            const Input<std::uint64_t>& offsets, const Input<std::uint64_t>& sources,
            const Input<std::uint64_t>& targets, const Input<double>& probabilities,
            const Input<std::int8_t>& signs, const Input<double>& diagonals,
            const Input<double>& references) {
            auto o=copy_array(offsets), s=copy_array(sources), t=copy_array(targets);
            auto p=copy_array(probabilities), d=copy_array(diagonals), r=copy_array(references);
            auto sign=copy_array(signs);
            nb::gil_scoped_release release;
            c.upload_aggregated_transition_rows(o,s,t,p,sign,d,r);
        }, "offsets"_a.noconvert(), "sources"_a.noconvert(), "targets"_a.noconvert(),
           "probabilities"_a.noconvert(), "int8_signs"_a.noconvert(),
           "diagonals"_a.noconvert(), "reference_elements"_a.noconvert())
        .def("advance_aggregated_evolution", [](mc::ResidentEvolutionContext& c,UInt64 steps) {
            nb::gil_scoped_release release;
            return c.advance_aggregated_evolution(steps);
        }, "max_steps"_a)
        .def("request_transitions", [](mc::ResidentEvolutionContext& c,UInt64 attempts) {
            nb::gil_scoped_release release;
            return c.request_transitions(attempts);
        }, "attempts_per_walker"_a)
        .def("apply_transition_batch", [](mc::ResidentEvolutionContext& c,
            UInt64 id, UInt64 trajectory, UInt64 step, UInt64 attempts,
            const Input<std::uint64_t>& sources, const Input<std::uint64_t>& targets,
            const Input<std::uint64_t>& multiplicities, const Input<double>& probabilities,
            const Input<mc::Coefficient>& coefficients, const Input<std::uint64_t>& diagonal_labels,
            const Input<double>& diagonals, const Input<std::uint64_t>& reference_labels,
            const Input<double>& references) {
            auto batch=batch_from_arrays(id, trajectory, step, attempts,
                sources, targets, multiplicities, probabilities, coefficients,
                diagonal_labels, diagonals, reference_labels, references);
            nb::gil_scoped_release release;
            return c.apply_transition_batch(batch);
        }, "request_id"_a, "trajectory_index"_a, "step_index"_a, "attempts_per_walker"_a,
           "sources"_a.noconvert(), "targets"_a.noconvert(), "multiplicities"_a.noconvert(),
           "proposal_probabilities"_a.noconvert(), "transition_coefficients"_a.noconvert(),
           "diagonal_labels"_a.noconvert(), "diagonal_elements"_a.noconvert(),
           "reference_labels"_a.noconvert(), "reference_elements"_a.noconvert())
        .def_prop_ro("diagnostics", [](const mc::ResidentEvolutionContext& c) {
            mc::Diagnostics snapshot;
            { nb::gil_scoped_release release; snapshot = c.diagnostics(); }
            return diagnostics(snapshot);
        });

    m.def("montecarlo_available", &mc::available,
          "Whether native GPU Monte Carlo numerical execution is implemented in this build.");
    m.def("make_resident_evolution_context", &mc::make_resident_evolution_context,
          "simulator"_a, "n"_a, nb::call_guard<nb::gil_scoped_release>());
}
void register_transition_column_bindings(nb::module_& m) {
    nb::class_<sim::TransitionColumnProvider>(m, "TransitionColumnProvider")
        .def("columns", [](sim::TransitionColumnProvider& p, const Input<std::uint64_t>& sources) {
            auto labels = copy_array(sources);
            std::vector<sim::Coefficient> values;
            { nb::gil_scoped_release release; values=p.columns(labels); }
            return owned_array(std::move(values), {sources.size(), std::size_t{1} << p.n_qubits()});
        }, "source_labels"_a.noconvert())
        .def("compact_columns", [](sim::TransitionColumnProvider& p, const Input<std::uint64_t>& sources, double tolerance) {
            auto input = copy_array(sources);
            sim::CompactColumns rows;
            { nb::gil_scoped_release release; rows=p.compact_columns(input,tolerance); }
            const auto offsets = rows.offsets.size(), labels = rows.labels.size(), values = rows.values.size();
            return nb::make_tuple(owned_array(std::move(rows.offsets), {offsets}),
                                  owned_array(std::move(rows.labels), {labels}),
                                  owned_array(std::move(rows.values), {values}));
        }, "source_labels"_a.noconvert(), "absolute_tolerance"_a,
           nb::sig("def compact_columns(self, source_labels: numpy.ndarray[dtype=uint64, shape=(*), order='C', device='cpu', writable=False], absolute_tolerance: float) -> tuple[numpy.ndarray[dtype=uint64, shape=(*), order='C', device='cpu'], numpy.ndarray[dtype=uint64, shape=(*), order='C', device='cpu'], numpy.ndarray[dtype=complex128, shape=(*), order='C', device='cpu']]") )
        .def_prop_ro("diagnostics", [](const sim::TransitionColumnProvider& p) {
            sim::Diagnostics snapshot;
            { nb::gil_scoped_release release; snapshot = p.diagnostics(); }
            return diagnostics(snapshot);
        });

    using Observable=std::vector<std::pair<std::vector<std::pair<std::uint32_t,char>>,std::complex<double>>>;
    nb::borrow<nb::class_<CudaqSimulator>>(m.attr("CudaqSimulator"))
        .def("make_transition_column_provider", [](CudaqSimulator& simulator,
            const std::vector<Command>& U,int n,const Observable& observable,UInt64 capacity,
            const std::optional<Input<std::uint64_t>>& sector) {
            auto labels=copy_array(sector);
            nb::gil_scoped_release release;
            return simulator.make_transition_column_provider(U,n,observable,capacity,labels);
        }, "U"_a,"n"_a,"observable"_a,"initial_batch_capacity"_a=1,
           "sector_target_labels"_a.noconvert()=nb::none());
}
}  // namespace qarpx::python
