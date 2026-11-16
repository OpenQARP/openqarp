#include "qarpx/montecarlo/contexts.h"
#include "qarpx/simulator/transition_columns.h"
#include <type_traits>
static_assert(std::is_same_v<qarpx::montecarlo::Label,std::uint64_t>);
static_assert(!std::is_copy_constructible_v<qarpx::simulator::TransitionColumnProvider>);
