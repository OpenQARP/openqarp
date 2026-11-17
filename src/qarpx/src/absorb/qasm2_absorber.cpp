#include "qarpx/absorb/qasm2_absorber.h"

#include "qarpx/core/command.h"

#include <cctype>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace qarpx {

// Internal linkage throughout: qasm3_absorber.cpp declares its own
// TokKind/Token/Lexer/Parser at `qarpx` scope, and two translation units
// sharing those names at external linkage is an ODR violation.
namespace {

// ============================================================================
// Lexer
// ============================================================================

enum class TokKind {
    Eof,
    Ident,       // identifier or keyword
    Int,         // integer literal
    Float,       // floating-point literal
    StringLit,   // "..." string literal (consumed whole, used in header)
    Semicolon,   // ;
    Comma,       // ,
    LParen,      // (
    RParen,      // )
    LBracket,    // [
    RBracket,    // ]
    LBrace,      // {
    RBrace,      // }
    Equals,      // ==
    Assign,      // =
    Arrow,       // ->
    At,          // @   (OpenQASM 3 modifier syntax; only to reject it clearly)
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Caret,       // ^
};

struct Token {
    TokKind     kind = TokKind::Eof;
    std::string text;
    double      fval = 0.0;
    int64_t     ival = 0;
};

class Lexer {
    const std::string& src_;
    size_t pos_ = 0;

public:
    explicit Lexer(const std::string& src) : src_(src) {}

    Token next() {
        skip_whitespace_and_comments();
        if (pos_ >= src_.size()) return {TokKind::Eof, ""};

        char c = src_[pos_];

        if (std::isalpha(static_cast<unsigned char>(c)) || c == '_') {
            size_t start = pos_;
            while (pos_ < src_.size() &&
                   (std::isalnum(static_cast<unsigned char>(src_[pos_])) || src_[pos_] == '_'))
                ++pos_;
            return {TokKind::Ident, src_.substr(start, pos_ - start)};
        }

        if (std::isdigit(static_cast<unsigned char>(c)) ||
            (c == '.' && pos_ + 1 < src_.size() &&
             std::isdigit(static_cast<unsigned char>(src_[pos_ + 1])))) {
            return lex_number();
        }

        // Only appears in the header: include "qelib1.inc";
        if (c == '"') {
            ++pos_;
            while (pos_ < src_.size() && src_[pos_] != '"') ++pos_;
            if (pos_ < src_.size()) ++pos_;
            return {TokKind::StringLit, ""};
        }

        ++pos_;
        switch (c) {
            case ';': return {TokKind::Semicolon, ";"};
            case ',': return {TokKind::Comma,     ","};
            case '(': return {TokKind::LParen,    "("};
            case ')': return {TokKind::RParen,    ")"};
            case '[': return {TokKind::LBracket,  "["};
            case ']': return {TokKind::RBracket,  "]"};
            case '{': return {TokKind::LBrace,    "{"};
            case '}': return {TokKind::RBrace,    "}"};
            case '@': return {TokKind::At,        "@"};
            case '+': return {TokKind::Plus,      "+"};
            case '*': return {TokKind::Star,      "*"};
            case '/': return {TokKind::Slash,     "/"};
            case '^': return {TokKind::Caret,     "^"};
            case '-':
                if (pos_ < src_.size() && src_[pos_] == '>') {
                    ++pos_;
                    return {TokKind::Arrow, "->"};
                }
                return {TokKind::Minus, "-"};
            case '=':
                if (pos_ < src_.size() && src_[pos_] == '=') {
                    ++pos_;
                    return {TokKind::Equals, "=="};
                }
                return {TokKind::Assign, "="};
            default:
                throw std::runtime_error(
                    std::string("QASM2Absorber: unexpected character '") + c + "'");
        }
    }

private:
    void skip_whitespace_and_comments() {
        while (pos_ < src_.size()) {
            if (std::isspace(static_cast<unsigned char>(src_[pos_]))) {
                ++pos_;
            } else if (pos_ + 1 < src_.size() && src_[pos_] == '/' && src_[pos_ + 1] == '/') {
                while (pos_ < src_.size() && src_[pos_] != '\n') ++pos_;
            } else {
                break;
            }
        }
    }

    Token lex_number() {
        size_t start = pos_;
        bool is_float = false;
        while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) ++pos_;
        if (pos_ < src_.size() && src_[pos_] == '.') {
            is_float = true;
            ++pos_;
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) ++pos_;
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) ++pos_;
            while (pos_ < src_.size() && std::isdigit(static_cast<unsigned char>(src_[pos_]))) ++pos_;
        }
        std::string text = src_.substr(start, pos_ - start);
        Token t;
        if (is_float) {
            t.kind = TokKind::Float;
            t.fval = std::stod(text);
        } else {
            t.kind = TokKind::Int;
            t.ival = std::stoll(text);
            t.fval = static_cast<double>(t.ival);
        }
        t.text = text;
        return t;
    }
};

// ============================================================================
// Parser
// ============================================================================

/// A declared register: where its bit 0 sits in qarp's flat index space.
struct RegInfo {
    uint32_t base = 0;
    uint32_t size = 0;
};

/// One argument of a quantum operation: `reg[i]` (indexed) or `reg` (whole).
struct Arg {
    std::string name;
    std::optional<uint32_t> index;  ///< nullopt = the whole register
};

/// Gate names this absorber understands but the emitter never writes, plus
/// the ones it does.  Kept as a set so `gate` definitions can be skipped and
/// their name resolved here instead of by parsing the body.
bool is_known_gate_name(const std::string& s) {
    static const std::set<std::string> kw = {
        // spec qelib1.inc (23)
        "u3", "u2", "u1", "cx", "id", "x", "y", "z", "h", "s", "sdg", "t", "tdg",
        "rx", "ry", "rz", "cz", "cy", "ch", "ccx", "crz", "cu1", "cu3",
        // later qiskit additions
        "u0", "u", "U", "swap", "cswap", "crx", "cry", "rxx", "rzz", "p", "cp",
        "sx", "sxdg",
        // definitions QASM2Emitter writes
        "ryy", "rzx", "ecr", "iswap", "iswapdg", "cu",
        "cs", "csdg", "csx", "csxdg",
    };
    return kw.count(s) > 0;
}

class Parser {
    Lexer lex_;
    Token cur_;
    QASM2Absorber::Result& result_;

    std::map<std::string, RegInfo> qregs_;
    std::map<std::string, RegInfo> cregs_;

public:
    explicit Parser(const std::string& src, QASM2Absorber::Result& out)
        : lex_(src), result_(out)
    {
        advance();
    }

    void parse() {
        skip_header();
        while (cur_.kind != TokKind::Eof) {
            parse_statement();
        }
    }

private:
    // ── Token management ─────────────────────────────────────────────────────

    void advance() { cur_ = lex_.next(); }

    Token consume(TokKind k, const char* expected_desc = nullptr) {
        if (cur_.kind != k) {
            std::string msg = "QASM2Absorber: expected ";
            msg += expected_desc ? expected_desc : "token";
            msg += ", got '";
            msg += cur_.text.empty() ? "(eof)" : cur_.text;
            msg += "'";
            throw std::runtime_error(msg);
        }
        Token t = cur_;
        advance();
        return t;
    }

    bool peek(TokKind k) const { return cur_.kind == k; }
    bool peek_ident(const char* s) const {
        return cur_.kind == TokKind::Ident && cur_.text == s;
    }

    // ── Header ───────────────────────────────────────────────────────────────

    void skip_header() {
        // Only the two header forms are skipped — ``OPENQASM <v>;`` and
        // ``include "…";``.  Skipping to the first *recognized* keyword instead
        // would let an unrecognized program (OpenQASM 3, say) run off the end
        // and parse to an empty command list rather than raising.
        while (cur_.kind == TokKind::Ident &&
               (cur_.text == "OPENQASM" || cur_.text == "include")) {
            while (cur_.kind != TokKind::Semicolon) {
                if (cur_.kind == TokKind::Eof)
                    throw std::runtime_error(
                        "QASM2Absorber: unterminated header statement");
                advance();
            }
            advance();  // ';'
        }
    }

    // ── Statement dispatch ───────────────────────────────────────────────────

    void parse_statement() {
        if (cur_.kind == TokKind::Eof) return;

        if (cur_.kind != TokKind::Ident) {
            throw std::runtime_error(
                "QASM2Absorber: expected a statement, got '" + cur_.text + "'");
        }

        const std::string& kw = cur_.text;

        // OpenQASM 3 syntax reaching a QASM 2 parser is a format mix-up, not a
        // parse error — say which so the caller reaches for from_qasm3().
        if (kw == "input" || kw == "gphase" || kw == "ctrl" || kw == "inv" ||
            kw == "qubit" || kw == "bit") {
            throw std::runtime_error(
                "QASM2Absorber: '" + kw + "' is OpenQASM 3 syntax, not OpenQASM 2 — "
                "use QASM3Absorber (Block.from_qasm3) for that program.");
        }
        if (kw == "opaque") {
            throw std::runtime_error(
                "QASM2Absorber: `opaque` declarations have no qarp representation.");
        }

        if (kw == "qreg")    { parse_reg_decl(true);  return; }
        if (kw == "creg")    { parse_reg_decl(false); return; }
        if (kw == "gate")    { skip_gate_def();       return; }
        if (kw == "if")      { parse_if_stmt();       return; }
        if (kw == "barrier") { parse_barrier();       return; }
        if (kw == "measure" || kw == "reset") { parse_qop(); return; }

        if (is_known_gate_name(kw)) { parse_qop(); return; }

        throw std::runtime_error("QASM2Absorber: unknown statement or gate '" + kw + "'");
    }

    /// A `qop`: a named gate call, a measure, or a reset.  The only thing an
    /// OpenQASM 2 `if` may govern, hence its own entry point.
    void parse_qop() {
        if (peek_ident("measure")) { parse_measure(); return; }
        if (peek_ident("reset"))   { parse_reset();   return; }
        parse_named_gate();
    }

    // ── Declarations ─────────────────────────────────────────────────────────

    void parse_reg_decl(bool quantum) {
        consume(TokKind::Ident);                                  // qreg | creg
        auto name = consume(TokKind::Ident, "register name");
        consume(TokKind::LBracket, "'['");
        auto n = consume(TokKind::Int, "register size");
        consume(TokKind::RBracket, "']'");
        consume(TokKind::Semicolon, "';'");

        auto& table = quantum ? qregs_ : cregs_;
        auto& total = quantum ? result_.n_qubits : result_.n_cbits;
        if (table.count(name.text))
            throw std::runtime_error(
                "QASM2Absorber: register '" + name.text + "' declared twice");

        // Registers concatenate in declaration order into qarp's flat index
        // space; that ordering is what makes the emitter's per-bit `creg c0,
        // c1, …` layout read back as bits 0, 1, ….
        table[name.text] = RegInfo{total, static_cast<uint32_t>(n.ival)};
        total += static_cast<uint32_t>(n.ival);
    }

    void skip_gate_def() {
        // The body is not parsed: the name is resolved from the known-gate
        // table instead, which is what lets the emitter's own prelude (and a
        // foreign file's equivalent) round-trip.
        consume(TokKind::Ident);  // "gate"
        if (cur_.kind != TokKind::Ident)
            throw std::runtime_error("QASM2Absorber: expected a name after `gate`");
        const std::string name = cur_.text;
        if (!is_known_gate_name(name))
            throw std::runtime_error(
                "QASM2Absorber: `gate " + name + "` defines a name with no qarp "
                "equivalent — decompose it before absorbing.");

        while (cur_.kind != TokKind::LBrace) {
            if (cur_.kind == TokKind::Eof)
                throw std::runtime_error("QASM2Absorber: unterminated gate definition");
            advance();
        }
        int depth = 0;
        do {
            if (cur_.kind == TokKind::LBrace) ++depth;
            else if (cur_.kind == TokKind::RBrace) --depth;
            else if (cur_.kind == TokKind::Eof)
                throw std::runtime_error("QASM2Absorber: unterminated gate definition");
            advance();
        } while (depth > 0);
    }

    // ── Arguments ────────────────────────────────────────────────────────────

    Arg parse_arg() {
        auto name = consume(TokKind::Ident, "register name");
        Arg arg{name.text, std::nullopt};
        if (peek(TokKind::LBracket)) {
            advance();
            auto idx = consume(TokKind::Int, "index");
            consume(TokKind::RBracket, "']'");
            arg.index = static_cast<uint32_t>(idx.ival);
        }
        return arg;
    }

    const RegInfo& lookup(const std::map<std::string, RegInfo>& table,
                          const std::string& name, const char* kind) const {
        auto it = table.find(name);
        if (it == table.end())
            throw std::runtime_error(
                "QASM2Absorber: undeclared " + std::string(kind) + " register '" + name + "'");
        return it->second;
    }

    /// Flat index of an indexed argument; whole-register args are an error here.
    uint32_t resolve_bit(const Arg& arg, bool quantum) const {
        const RegInfo& reg = lookup(quantum ? qregs_ : cregs_, arg.name,
                                    quantum ? "quantum" : "classical");
        if (!arg.index)
            throw std::runtime_error(
                "QASM2Absorber: register '" + arg.name +
                "' used where a single bit is required");
        if (*arg.index >= reg.size)
            throw std::runtime_error(
                "QASM2Absorber: index " + std::to_string(*arg.index) +
                " out of range for register '" + arg.name + "'");
        return reg.base + *arg.index;
    }

    /// Every flat index an argument covers — one bit, or a whole register.
    std::vector<uint32_t> expand(const Arg& arg, bool quantum) const {
        const RegInfo& reg = lookup(quantum ? qregs_ : cregs_, arg.name,
                                    quantum ? "quantum" : "classical");
        if (arg.index) return {resolve_bit(arg, quantum)};
        std::vector<uint32_t> all;
        all.reserve(reg.size);
        for (uint32_t i = 0; i < reg.size; ++i) all.push_back(reg.base + i);
        return all;
    }

    // ── Parameter expressions ────────────────────────────────────────────────
    //   expr  = term (('+' | '-') term)*
    //   term  = unary (('*' | '/') unary)*
    //   unary = '-' atom | atom
    //   atom  = 'pi' | INT | FLOAT | '(' expr ')'
    //
    // OpenQASM 2 has no symbolic parameter, so every expression concretizes.

    Param parse_param_expr() { return parse_add_expr(); }

    Param parse_add_expr() {
        Param lhs = parse_mul_expr();
        while (peek(TokKind::Plus) || peek(TokKind::Minus)) {
            bool add = peek(TokKind::Plus);
            advance();
            Param rhs = parse_mul_expr();
            lhs = add ? lhs + rhs : lhs - rhs;
        }
        return lhs;
    }

    Param parse_mul_expr() {
        Param lhs = parse_unary_expr();
        while (peek(TokKind::Star) || peek(TokKind::Slash)) {
            bool mul = peek(TokKind::Star);
            advance();
            Param rhs = parse_unary_expr();
            lhs = mul ? lhs * rhs : lhs / rhs;
        }
        return lhs;
    }

    Param parse_unary_expr() {
        if (peek(TokKind::Minus)) {
            advance();
            return -parse_unary_expr();
        }
        return parse_atom();
    }

    Param parse_atom() {
        if (cur_.kind == TokKind::LParen) {
            advance();
            Param p = parse_param_expr();
            consume(TokKind::RParen, "')'");
            return p;
        }
        if (cur_.kind == TokKind::Float) {
            double v = cur_.fval;
            advance();
            return Param(v);
        }
        if (cur_.kind == TokKind::Int) {
            double v = static_cast<double>(cur_.ival);
            advance();
            return Param(v);
        }
        if (cur_.kind == TokKind::Ident) {
            std::string name = cur_.text;
            advance();
            if (name == "pi") return Param(M_PI);
            throw std::runtime_error(
                "QASM2Absorber: unknown identifier '" + name + "' in a parameter "
                "expression — OpenQASM 2 has no symbolic parameters, only 'pi' "
                "and numeric literals.");
        }
        throw std::runtime_error(
            "QASM2Absorber: unexpected token '" + cur_.text + "' in parameter expression.");
    }

    std::vector<Param> parse_param_list() {
        std::vector<Param> params;
        if (!peek(TokKind::LParen)) return params;
        advance();
        if (!peek(TokKind::RParen)) {
            params.push_back(parse_param_expr());
            while (peek(TokKind::Comma)) {
                advance();
                params.push_back(parse_param_expr());
            }
        }
        consume(TokKind::RParen, "')'");
        return params;
    }

    std::vector<Arg> parse_arg_list() {
        std::vector<Arg> args;
        args.push_back(parse_arg());
        while (peek(TokKind::Comma)) {
            advance();
            args.push_back(parse_arg());
        }
        return args;
    }

    // ── Statements ───────────────────────────────────────────────────────────

    void parse_named_gate() {
        std::string name = cur_.text;
        advance();
        auto params = parse_param_list();
        auto args   = parse_arg_list();
        consume(TokKind::Semicolon, "';'");

        std::vector<uint32_t> qubits;
        qubits.reserve(args.size());
        for (const auto& a : args) qubits.push_back(resolve_bit(a, /*quantum=*/true));

        emit_named_gate(name, params, qubits);
    }

    void parse_measure() {
        consume(TokKind::Ident);                 // "measure"
        Arg q = parse_arg();
        consume(TokKind::Arrow, "'->'");
        Arg c = parse_arg();
        consume(TokKind::Semicolon, "';'");

        auto qbits = expand(q, /*quantum=*/true);
        auto cbits = expand(c, /*quantum=*/false);
        if (qbits.size() != cbits.size())
            throw std::runtime_error(
                "QASM2Absorber: measure operand sizes differ (" +
                std::to_string(qbits.size()) + " vs " + std::to_string(cbits.size()) + ")");

        for (size_t i = 0; i < qbits.size(); ++i) {
            Command cmd(GateType::Measure, qbits[i]);
            cmd.cbits.push_back(cbits[i]);
            push(std::move(cmd));
        }
    }

    void parse_reset() {
        consume(TokKind::Ident);                 // "reset"
        Arg q = parse_arg();
        consume(TokKind::Semicolon, "';'");
        for (uint32_t bit : expand(q, /*quantum=*/true))
            push(Command(GateType::Reset, bit));
    }

    void parse_barrier() {
        consume(TokKind::Ident);                 // "barrier"
        auto args = parse_arg_list();
        consume(TokKind::Semicolon, "';'");

        Command cmd;
        cmd.gate = GateType::Barrier;
        for (const auto& a : args)
            for (uint32_t bit : expand(a, /*quantum=*/true)) cmd.qubits.push_back(bit);
        push(std::move(cmd));
    }

    void parse_if_stmt() {
        // ``if (<creg> == <int>) <qop>;`` — the comparison is against the whole
        // register, so the condition is the AND of that integer's bits.
        consume(TokKind::Ident);                 // "if"
        consume(TokKind::LParen, "'('");
        auto reg_name = consume(TokKind::Ident, "classical register name");
        consume(TokKind::Equals, "'=='");
        auto value = consume(TokKind::Int, "integer");
        consume(TokKind::RParen, "')'");

        const RegInfo& reg = lookup(cregs_, reg_name.text, "classical");
        if (value.ival < 0)
            throw std::runtime_error("QASM2Absorber: negative `if` comparison value");
        const auto uval = static_cast<uint64_t>(value.ival);
        if (reg.size < 64 && uval >= (1ULL << reg.size))
            throw std::runtime_error(
                "QASM2Absorber: `if (" + reg_name.text + " == " + value.text +
                ")` compares against a value wider than the register");

        Command begin;
        begin.gate = GateType::BranchBegin;
        for (uint32_t i = 0; i < reg.size; ++i) {
            begin.condition_bits.push_back(reg.base + i);
            begin.condition_values.push_back(((uval >> i) & 1ULL) != 0);
        }
        push(std::move(begin));

        parse_qop();

        Command end;
        end.gate = GateType::BranchEnd;
        push(std::move(end));
    }

    // ── Gate-name → GateType ─────────────────────────────────────────────────

    void emit_named_gate(const std::string& name,
                         const std::vector<Param>& params,
                         const std::vector<uint32_t>& qubits)
    {
        auto check = [&](size_t nq, size_t np) {
            if (qubits.size() != nq)
                throw std::runtime_error("QASM2Absorber: gate '" + name
                    + "' expects " + std::to_string(nq) + " qubits");
            if (params.size() != np)
                throw std::runtime_error("QASM2Absorber: gate '" + name
                    + "' expects " + std::to_string(np) + " params");
        };
        auto simple = [&](GateType g, size_t nq) {
            check(nq, 0);
            Command cmd;
            cmd.gate = g;
            for (auto q : qubits) cmd.qubits.push_back(q);
            push(std::move(cmd));
        };
        auto one_param = [&](GateType g, size_t nq) {
            check(nq, 1);
            Command cmd;
            cmd.gate = g;
            for (auto q : qubits) cmd.qubits.push_back(q);
            cmd.params.push_back(params[0]);
            push(std::move(cmd));
        };
        auto general = [&](GateType g, size_t nq, size_t np) {
            check(nq, np);
            Command cmd;
            cmd.gate = g;
            for (auto q : qubits) cmd.qubits.push_back(q);
            for (const auto& p : params) cmd.params.push_back(p);
            push(std::move(cmd));
        };

        // `id` is the explicit identity gate (§2.6); `u0` is an idle of
        // `length` cycles with no gate meaning and contributes nothing.
        if (name == "id") return simple(GateType::Id, 1);
        if (name == "u0") { check(1, 1); return; }

        if (name == "x")   return simple(GateType::X,   1);
        if (name == "y")   return simple(GateType::Y,   1);
        if (name == "z")   return simple(GateType::Z,   1);
        if (name == "h")   return simple(GateType::H,   1);
        if (name == "s")   return simple(GateType::S,   1);
        if (name == "sdg") return simple(GateType::Sdg, 1);
        if (name == "t")   return simple(GateType::T,   1);
        if (name == "tdg") return simple(GateType::Tdg, 1);

        if (name == "rx") return one_param(GateType::Rx, 1);
        if (name == "ry") return one_param(GateType::Ry, 1);
        if (name == "rz") return one_param(GateType::Rz, 1);
        // u1 and p are the same matrix as qarp's P — diag(1, e^{iλ}).
        if (name == "u1" || name == "p") return one_param(GateType::P, 1);

        if (name == "u3" || name == "u" || name == "U") return general(GateType::U, 1, 3);
        // qelib1: gate u2(phi,lambda) q { U(pi/2,phi,lambda) q; }
        if (name == "u2") {
            check(1, 2);
            Command cmd;
            cmd.gate = GateType::U;
            cmd.qubits.push_back(qubits[0]);
            cmd.params.push_back(Param(M_PI / 2));
            cmd.params.push_back(params[0]);
            cmd.params.push_back(params[1]);
            push(std::move(cmd));
            return;
        }
        if (name == "sx")   return simple(GateType::SX,   1);
        if (name == "sxdg") return simple(GateType::SXdg, 1);

        if (name == "cx")      return simple(GateType::CX,      2);
        if (name == "cy")      return simple(GateType::CY,      2);
        if (name == "cz")      return simple(GateType::CZ,      2);
        if (name == "swap")    return simple(GateType::SWAP,    2);
        if (name == "ecr")     return simple(GateType::ECR,     2);
        if (name == "iswap")   return simple(GateType::iSWAP,   2);
        if (name == "iswapdg") return simple(GateType::iSWAPdg, 2);
        if (name == "ch")      return simple(GateType::CH,      2);
        if (name == "cs")      return simple(GateType::CS,      2);
        if (name == "csdg")    return simple(GateType::CSdg,    2);
        if (name == "csx")     return simple(GateType::CSX,     2);
        if (name == "csxdg")   return simple(GateType::CSXdg,   2);

        if (name == "crx") return one_param(GateType::CRx, 2);
        if (name == "cry") return one_param(GateType::CRy, 2);
        if (name == "crz") return one_param(GateType::CRz, 2);
        // cu1 and cp are the same matrix as qarp's CP.
        if (name == "cu1" || name == "cp") return one_param(GateType::CP, 2);
        if (name == "rzz") return one_param(GateType::RZZ, 2);
        if (name == "rxx") return one_param(GateType::RXX, 2);
        if (name == "ryy") return one_param(GateType::RYY, 2);
        // rzx has no qarp GateType; it exists only as the `ecr` body's helper.
        if (name == "rzx")
            throw std::runtime_error(
                "QASM2Absorber: `rzx` has no qarp gate — it appears only as the "
                "helper inside an `ecr` definition, whose body is not parsed.");

        if (name == "cu")  return general(GateType::CU, 2, 4);
        // cu3(θ,φ,λ) is cu with no global phase.
        if (name == "cu3") {
            check(2, 3);
            Command cmd;
            cmd.gate = GateType::CU;
            for (auto q : qubits) cmd.qubits.push_back(q);
            for (const auto& p : params) cmd.params.push_back(p);
            cmd.params.push_back(Param(0.0));
            push(std::move(cmd));
            return;
        }
        if (name == "ccx")   return simple(GateType::CCX,   3);
        if (name == "cswap") return simple(GateType::CSWAP, 3);

        throw std::runtime_error("QASM2Absorber: unknown gate '" + name + "'");
    }

    void push(Command cmd) { result_.commands.push_back(std::move(cmd)); }
};

}  // namespace

QASM2Absorber::Result QASM2Absorber::absorb(const std::string& qasm2_source) const {
    Result result;
    Parser parser(qasm2_source, result);
    parser.parse();
    return result;
}

}  // namespace qarpx
