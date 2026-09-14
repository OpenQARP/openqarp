#pragma once

#include "command.h"

#include <stdexcept>
#include <string>

namespace qarpx {

/// A command (or its parameters) cannot cross to the requested backend.
/// Translated to qarp.errors.CapabilityError at the Python boundary; the
/// offending command rides along as the exception's `command` attribute.
class capability_error : public std::runtime_error {
public:
    capability_error(const std::string& what, Command cmd)
        : std::runtime_error(what), command(std::move(cmd)) {}
    explicit capability_error(const std::string& what)
        : std::runtime_error(what), command{} {}

    Command command;
};

/// A required SDK module is not importable in this environment.
/// Translated to ImportError at the Python boundary: the failure is about
/// the environment, never about the circuit.
class sdk_missing_error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

}  // namespace qarpx
