#include "qarpx/absorb/qasm3_absorber.h"

#include "qarpx/core/gates.h"

#include <cassert>
#include <cctype>
#include <cmath>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace qarpx {

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
    At,          // @
    Plus,        // +
    Minus,       // -
    Star,        // *
    Slash,       // /
    Ampersand,   // &  (part of &&)
    AmpAmp,      // &&
    Bang,        // !
};

struct Token {
    TokKind     kind;
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

        // Identifier / keyword
        if (std::isalpha(c) || c == '_') {
            size_t start = pos_;
            while (pos_ < src_.size() && (std::isalnum(src_[pos_]) || src_[pos_] == '_'))
                ++pos_;
            return {TokKind::Ident, src_.substr(start, pos_ - start)};
        }

        // Number: integer or float
        if (std::isdigit(c) || (c == '.' && pos_ + 1 < src_.size() && std::isdigit(src_[pos_+1]))) {
            return lex_number();
        }

        // String literal — only appears in header lines like include "stdgates.inc";
        if (c == '"') {
            ++pos_;  // consume opening quote
            while (pos_ < src_.size() && src_[pos_] != '"') ++pos_;
            if (pos_ < src_.size()) ++pos_;  // consume closing quote
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
            case '-': return {TokKind::Minus,     "-"};
            case '*': return {TokKind::Star,      "*"};
            case '/': return {TokKind::Slash,     "/"};
            case '!': return {TokKind::Bang,      "!"};
            case '=':
                if (pos_ < src_.size() && src_[pos_] == '=') {
                    ++pos_;
                    return {TokKind::Equals, "=="};
                }
                return {TokKind::Assign, "="};
            case '&':
                if (pos_ < src_.size() && src_[pos_] == '&') {
                    ++pos_;
                    return {TokKind::AmpAmp, "&&"};
                }
                return {TokKind::Ampersand, "&"};
            default:
                throw std::runtime_error(
                    std::string("QASM3Absorber: unexpected character '") + c + "'");
        }
    }

    [[nodiscard]] size_t pos() const { return pos_; }

private:
    void skip_whitespace_and_comments() {
        while (pos_ < src_.size()) {
            if (std::isspace(src_[pos_])) {
                ++pos_;
            } else if (pos_ + 1 < src_.size() && src_[pos_] == '/' && src_[pos_+1] == '/') {
                // Line comment
                while (pos_ < src_.size() && src_[pos_] != '\n') ++pos_;
            } else {
                break;
            }
        }
    }

    Token lex_number() {
        size_t start = pos_;
        bool is_float = false;
        while (pos_ < src_.size() && std::isdigit(src_[pos_])) ++pos_;
        if (pos_ < src_.size() && src_[pos_] == '.') {
            is_float = true;
            ++pos_;
            while (pos_ < src_.size() && std::isdigit(src_[pos_])) ++pos_;
        }
        if (pos_ < src_.size() && (src_[pos_] == 'e' || src_[pos_] == 'E')) {
            is_float = true;
            ++pos_;
            if (pos_ < src_.size() && (src_[pos_] == '+' || src_[pos_] == '-')) ++pos_;
            while (pos_ < src_.size() && std::isdigit(src_[pos_])) ++pos_;
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

class Parser {
    Lexer               lex_;
    Token               cur_;
    QASM3Absorber::Result& result_;
    // symbol name → true (just tracks which names are declared as input)
    std::set<std::string> declared_symbols_;

public:
    explicit Parser(const std::string& src, QASM3Absorber::Result& out)
        : lex_(src), result_(out)
    {
        advance();
    }

    void parse() {
        // Skip the OPENQASM and include header lines
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
            std::string msg = "QASM3Absorber: expected ";
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

    // ── Header skip ──────────────────────────────────────────────────────────

    void skip_header() {
        // Only the two header forms are skipped — ``OPENQASM <v>;`` and
        // ``include "…";``.  Skipping to the first *recognized* keyword instead
        // let an unrecognized program (an OpenQASM 2 file, say) run past its
        // register declarations, so the gates that followed were absorbed
        // against ``n_qubits == 0`` and only failed later, at flatten().
        while (cur_.kind == TokKind::Ident &&
               (cur_.text == "OPENQASM" || cur_.text == "include")) {
            while (cur_.kind != TokKind::Semicolon) {
                if (cur_.kind == TokKind::Eof)
                    throw std::runtime_error(
                        "QASM3Absorber: unterminated header statement");
                advance();
            }
            advance();  // ';'
        }
    }

    static bool is_gate_keyword(const std::string& s) {
        static const std::set<std::string> kw = {
            "x","y","z","h","s","sdg","t","tdg","sx","id",
            "rx","ry","rz","p","U","u3","u",
            "cx","cy","cz","swap","ecr","iswap","ch",
            "crx","cry","crz","cp","rzz","rxx","ryy","cu",
            "ccx","cswap","gphase","barrier","reset",
        };
        return kw.count(s) > 0;
    }

    // ── Statement dispatch ────────────────────────────────────────────────────

    void parse_statement() {
        if (cur_.kind == TokKind::Eof) return;

        if (peek_ident("gate"))   { skip_gate_def();     return; }
        if (peek_ident("qubit"))  { parse_qubit_decl();  return; }
        if (peek_ident("bit"))    { parse_bit_decl();    return; }
        if (peek_ident("input"))  { parse_input_decl();  return; }
        if (peek_ident("if"))     { parse_if_stmt();     return; }
        if (peek_ident("barrier")){ parse_barrier();     return; }
        if (peek_ident("reset"))  { parse_reset();       return; }
        if (peek_ident("gphase")) { parse_gphase();      return; }
        if (peek_ident("ctrl"))   { parse_ctrl_gate();   return; }
        if (peek_ident("inv"))    { parse_inv_gate();    return; }

        // Classical assignment: ``c[i] = measure q[j];``
        if (cur_.kind == TokKind::Ident && cur_.text == "c") {
            parse_measure_assign();
            return;
        }

        // Named gate statement
        if (cur_.kind == TokKind::Ident && is_gate_keyword(cur_.text)) {
            parse_named_gate();
            return;
        }

        // `qreg`/`creg`/`opaque` are unambiguously OpenQASM 2, and reaching
        // here with one is a format mix-up rather than a syntax error — say
        // which, so the caller reaches for from_qasm2() instead of debugging.
        if (cur_.kind == TokKind::Ident &&
            (cur_.text == "qreg" || cur_.text == "creg" || cur_.text == "opaque")) {
            throw std::runtime_error(
                "QASM3Absorber: '" + cur_.text + "' is OpenQASM 2 syntax, not "
                "OpenQASM 3 — use QASM2Absorber (SimpleBlock.from_qasm2) for that "
                "program.");
        }

        // Silently skipping an unknown statement used to leave the commands
        // that followed referring to registers never declared.
        throw std::runtime_error(
            "QASM3Absorber: unknown statement or gate '" +
            (cur_.text.empty() ? std::string("(eof)") : cur_.text) + "'");
    }

    /// Skip a ``gate name(params) a, b { ... }`` definition wholesale.  The
    /// absorber maps gate *names* straight to GateTypes, so definitions carry
    /// no information for it — but a semicolon-level skip would stop inside
    /// the body and absorb its tail as real program commands.
    void skip_gate_def() {
        consume(TokKind::Ident);  // "gate"
        while (cur_.kind != TokKind::LBrace) {
            if (cur_.kind == TokKind::Eof)
                throw std::runtime_error("QASM3Absorber: unterminated gate definition");
            advance();
        }
        int depth = 0;
        do {
            if (cur_.kind == TokKind::LBrace) ++depth;
            else if (cur_.kind == TokKind::RBrace) --depth;
            else if (cur_.kind == TokKind::Eof)
                throw std::runtime_error("QASM3Absorber: unterminated gate definition");
            advance();
        } while (depth > 0);
    }

    // ── Register declarations ─────────────────────────────────────────────────

    void parse_qubit_decl() {
        consume(TokKind::Ident);           // "qubit"
        consume(TokKind::LBracket, "'['");
        auto n = consume(TokKind::Int, "integer");
        consume(TokKind::RBracket, "']'");
        consume(TokKind::Ident, "register name"); // "q"
        consume(TokKind::Semicolon, "';'");
        result_.n_qubits = static_cast<uint32_t>(n.ival);
    }

    void parse_bit_decl() {
        consume(TokKind::Ident);           // "bit"
        consume(TokKind::LBracket, "'['");
        auto n = consume(TokKind::Int, "integer");
        consume(TokKind::RBracket, "']'");
        consume(TokKind::Ident, "register name"); // "c"
        consume(TokKind::Semicolon, "';'");
        result_.n_cbits = static_cast<uint32_t>(n.ival);
    }

    void parse_input_decl() {
        // ``input float[64] <name>;``
        consume(TokKind::Ident);           // "input"
        consume(TokKind::Ident, "'float'"); // "float"
        consume(TokKind::LBracket, "'['");
        consume(TokKind::Int, "64");
        consume(TokKind::RBracket, "']'");
        auto name_tok = consume(TokKind::Ident, "symbol name");
        consume(TokKind::Semicolon, "';'");
        declared_symbols_.insert(name_tok.text);
        result_.free_symbols.push_back(name_tok.text);
    }

    // ── Qubit / cbit index parsing ────────────────────────────────────────────

    uint32_t parse_qubit_ref() {
        // ``q[N]``
        consume(TokKind::Ident, "'q'");    // "q"
        consume(TokKind::LBracket, "'['");
        auto idx = consume(TokKind::Int, "qubit index");
        consume(TokKind::RBracket, "']'");
        return static_cast<uint32_t>(idx.ival);
    }

    uint32_t parse_cbit_ref() {
        // ``c[N]``
        consume(TokKind::Ident, "'c'");    // "c"
        consume(TokKind::LBracket, "'['");
        auto idx = consume(TokKind::Int, "cbit index");
        consume(TokKind::RBracket, "']'");
        return static_cast<uint32_t>(idx.ival);
    }

    // ── Param expression parser ───────────────────────────────────────────────
    // Grammar (simplified):
    //   expr     = term (('+' | '-') term)*
    //   term     = factor (('*' | '/') factor)*
    //   factor   = ('-')? atom
    //   atom     = 'pi' | INT | FLOAT | IDENT | '(' expr ')'

    Param parse_param_expr() {
        return parse_add_expr();
    }

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
            return -parse_atom();
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
            if (declared_symbols_.count(name)) return Param::symbol(name);
            throw std::runtime_error(
                "QASM3Absorber: unknown identifier '" + name + "' in parameter expression. "
                "Expected 'pi' or a declared input symbol.");
        }
        throw std::runtime_error(
            "QASM3Absorber: unexpected token '" + cur_.text + "' in parameter expression.");
    }

    // Parse a ``(p0, p1, …)`` parameter list.  Returns empty vector if next
    // token is not '('.
    std::vector<Param> parse_param_list() {
        std::vector<Param> params;
        if (!peek(TokKind::LParen)) return params;
        advance();  // '('
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

    // Parse a comma-separated qubit list ``q[i], q[j], …``.
    std::vector<uint32_t> parse_qubit_list() {
        std::vector<uint32_t> qubits;
        qubits.push_back(parse_qubit_ref());
        while (peek(TokKind::Comma)) {
            advance();
            qubits.push_back(parse_qubit_ref());
        }
        return qubits;
    }

    // ── Named gate (standard gate call) ──────────────────────────────────────

    void parse_named_gate() {
        std::string name = cur_.text;
        advance();

        auto params  = parse_param_list();
        auto qubits  = parse_qubit_list();
        consume(TokKind::Semicolon, "';'");

        emit_named_gate(name, params, qubits);
    }

    void emit_named_gate(const std::string& name,
                         const std::vector<Param>& params,
                         const std::vector<uint32_t>& qubits)
    {
        auto check = [&](size_t nq, size_t np) {
            if (qubits.size() != nq)
                throw std::runtime_error("QASM3Absorber: gate '" + name
                    + "' expects " + std::to_string(nq) + " qubits");
            if (params.size() != np)
                throw std::runtime_error("QASM3Absorber: gate '" + name
                    + "' expects " + std::to_string(np) + " params");
        };

        Command cmd;

        if (name == "x")   { check(1,0); cmd.gate = GateType::X;  cmd.qubits.push_back(qubits[0]); }
        else if (name == "y")   { check(1,0); cmd.gate = GateType::Y;  cmd.qubits.push_back(qubits[0]); }
        else if (name == "z")   { check(1,0); cmd.gate = GateType::Z;  cmd.qubits.push_back(qubits[0]); }
        else if (name == "h")   { check(1,0); cmd.gate = GateType::H;  cmd.qubits.push_back(qubits[0]); }
        else if (name == "s")   { check(1,0); cmd.gate = GateType::S;  cmd.qubits.push_back(qubits[0]); }
        else if (name == "sdg") { check(1,0); cmd.gate = GateType::Sdg; cmd.qubits.push_back(qubits[0]); }
        else if (name == "t")   { check(1,0); cmd.gate = GateType::T;  cmd.qubits.push_back(qubits[0]); }
        else if (name == "tdg") { check(1,0); cmd.gate = GateType::Tdg; cmd.qubits.push_back(qubits[0]); }
        else if (name == "sx")  { check(1,0); cmd.gate = GateType::SX;  cmd.qubits.push_back(qubits[0]); }
        else if (name == "id")  { check(1,0); cmd.gate = GateType::Id;  cmd.qubits.push_back(qubits[0]); }

        else if (name == "rx") { check(1,1); cmd.gate = GateType::Rx; cmd.qubits.push_back(qubits[0]); cmd.params.push_back(params[0]); }
        else if (name == "ry") { check(1,1); cmd.gate = GateType::Ry; cmd.qubits.push_back(qubits[0]); cmd.params.push_back(params[0]); }
        else if (name == "rz") { check(1,1); cmd.gate = GateType::Rz; cmd.qubits.push_back(qubits[0]); cmd.params.push_back(params[0]); }
        else if (name == "p")  { check(1,1); cmd.gate = GateType::P;  cmd.qubits.push_back(qubits[0]); cmd.params.push_back(params[0]); }
        else if (name == "U" || name == "u" || name == "u3") {
            check(1,3);
            cmd.gate = GateType::U;
            cmd.qubits.push_back(qubits[0]);
            for (auto& p : params) cmd.params.push_back(p);
        }

        else if (name == "cx")   { check(2,0); cmd.gate = GateType::CX;   for (auto q : qubits) cmd.qubits.push_back(q); }
        else if (name == "cy")   { check(2,0); cmd.gate = GateType::CY;   for (auto q : qubits) cmd.qubits.push_back(q); }
        else if (name == "cz")   { check(2,0); cmd.gate = GateType::CZ;   for (auto q : qubits) cmd.qubits.push_back(q); }
        else if (name == "swap") { check(2,0); cmd.gate = GateType::SWAP; for (auto q : qubits) cmd.qubits.push_back(q); }
        else if (name == "ecr")  { check(2,0); cmd.gate = GateType::ECR;  for (auto q : qubits) cmd.qubits.push_back(q); }
        else if (name == "iswap"){ check(2,0); cmd.gate = GateType::iSWAP; for (auto q : qubits) cmd.qubits.push_back(q); }
        else if (name == "ch")   { check(2,0); cmd.gate = GateType::CH;   for (auto q : qubits) cmd.qubits.push_back(q); }

        else if (name == "crx") { check(2,1); cmd.gate = GateType::CRx; for (auto q : qubits) cmd.qubits.push_back(q); cmd.params.push_back(params[0]); }
        else if (name == "cry") { check(2,1); cmd.gate = GateType::CRy; for (auto q : qubits) cmd.qubits.push_back(q); cmd.params.push_back(params[0]); }
        else if (name == "crz") { check(2,1); cmd.gate = GateType::CRz; for (auto q : qubits) cmd.qubits.push_back(q); cmd.params.push_back(params[0]); }
        else if (name == "cp")  { check(2,1); cmd.gate = GateType::CP;  for (auto q : qubits) cmd.qubits.push_back(q); cmd.params.push_back(params[0]); }
        else if (name == "rzz") { check(2,1); cmd.gate = GateType::RZZ; for (auto q : qubits) cmd.qubits.push_back(q); cmd.params.push_back(params[0]); }
        else if (name == "rxx") { check(2,1); cmd.gate = GateType::RXX; for (auto q : qubits) cmd.qubits.push_back(q); cmd.params.push_back(params[0]); }
        else if (name == "ryy") { check(2,1); cmd.gate = GateType::RYY; for (auto q : qubits) cmd.qubits.push_back(q); cmd.params.push_back(params[0]); }

        else if (name == "cu") {
            check(2,4);
            cmd.gate = GateType::CU;
            for (auto q : qubits) cmd.qubits.push_back(q);
            for (auto& p : params) cmd.params.push_back(p);
        }

        else if (name == "ccx")  { check(3,0); cmd.gate = GateType::CCX;  for (auto q : qubits) cmd.qubits.push_back(q); }
        else if (name == "cswap"){ check(3,0); cmd.gate = GateType::CSWAP; for (auto q : qubits) cmd.qubits.push_back(q); }

        else {
            throw std::runtime_error("QASM3Absorber: unknown gate '" + name + "'");
        }

        result_.commands.push_back(std::move(cmd));
    }

    // ── ctrl(n) @ z ──────────────────────────────────────────────────────────

    void parse_ctrl_gate() {
        consume(TokKind::Ident);  // "ctrl"
        // ``ctrl @ s|sdg|sx q[c], q[t];`` and ``ctrl @ inv @ sx q[c], q[t];`` —
        // the count-less form the emitter writes for CS/CSdg/CSX/CSXdg (§12.1).
        if (!peek(TokKind::LParen)) {
            consume(TokKind::At, "'@'");
            auto gname = consume(TokKind::Ident, "gate name after ctrl @");
            GateType g;
            if (gname.text == "s")        g = GateType::CS;
            else if (gname.text == "sdg") g = GateType::CSdg;
            else if (gname.text == "sx")  g = GateType::CSX;
            else if (gname.text == "inv") {
                consume(TokKind::At, "'@'");
                auto inner = consume(TokKind::Ident, "gate name after ctrl @ inv @");
                if (inner.text != "sx")
                    throw std::runtime_error(
                        "QASM3Absorber: ctrl @ inv modifier only supported for 'sx' (CSXdg). Got '"
                        + inner.text + "'");
                g = GateType::CSXdg;
            } else {
                throw std::runtime_error(
                    "QASM3Absorber: ctrl @ modifier only supported for 's', 'sdg', 'sx', 'inv @ sx'. Got '"
                    + gname.text + "'");
            }
            auto qubits = parse_qubit_list();
            consume(TokKind::Semicolon, "';'");
            if (qubits.size() != 2)
                throw std::runtime_error("QASM3Absorber: ctrl @ " + gname.text + " expects 2 qubits");
            Command cmd;
            cmd.gate = g;
            for (auto q : qubits) cmd.qubits.push_back(q);
            result_.commands.push_back(std::move(cmd));
            return;
        }
        // ``ctrl(N) @ z q[c0], ..., q[cn-1], q[t];``
        consume(TokKind::LParen, "'('");
        auto n_tok = consume(TokKind::Int, "control count");
        consume(TokKind::RParen, "')'");
        consume(TokKind::At, "'@'");
        auto gname = consume(TokKind::Ident, "gate name after ctrl");
        if (gname.text != "z")
            throw std::runtime_error(
                "QASM3Absorber: ctrl modifier only supported for 'z' (MCZ). Got '" + gname.text + "'");
        auto qubits = parse_qubit_list();
        consume(TokKind::Semicolon, "';'");

        Command cmd;
        cmd.gate = GateType::MCZ;
        for (auto q : qubits) cmd.qubits.push_back(q);
        result_.commands.push_back(std::move(cmd));
    }

    // ── inv @ iswap ──────────────────────────────────────────────────────────

    void parse_inv_gate() {
        // ``inv @ iswap q[a], q[b];``
        consume(TokKind::Ident);   // "inv"
        consume(TokKind::At, "'@'");
        auto gname = consume(TokKind::Ident, "gate name after inv");
        GateType g;
        if (gname.text == "iswap")   g = GateType::iSWAPdg;
        else if (gname.text == "sx") g = GateType::SXdg;
        else
            throw std::runtime_error(
                "QASM3Absorber: inv modifier only supported for 'iswap' and 'sx'. Got '" + gname.text + "'");
        auto qubits = parse_qubit_list();
        consume(TokKind::Semicolon, "';'");

        Command cmd;
        cmd.gate = g;
        for (auto q : qubits) cmd.qubits.push_back(q);
        result_.commands.push_back(std::move(cmd));
    }

    // ── gphase ───────────────────────────────────────────────────────────────

    void parse_gphase() {
        consume(TokKind::Ident);  // "gphase"
        auto params = parse_param_list();
        if (params.size() != 1)
            throw std::runtime_error("QASM3Absorber: gphase expects 1 parameter");
        consume(TokKind::Semicolon, "';'");

        Command cmd;
        cmd.gate = GateType::GPhase;
        cmd.params.push_back(params[0]);
        result_.commands.push_back(std::move(cmd));
    }

    // ── barrier ──────────────────────────────────────────────────────────────

    void parse_barrier() {
        consume(TokKind::Ident);  // "barrier"
        auto qubits = parse_qubit_list();
        consume(TokKind::Semicolon, "';'");

        Command cmd;
        cmd.gate = GateType::Barrier;
        for (auto q : qubits) cmd.qubits.push_back(q);
        result_.commands.push_back(std::move(cmd));
    }

    // ── reset ────────────────────────────────────────────────────────────────

    void parse_reset() {
        consume(TokKind::Ident);  // "reset"
        uint32_t q = parse_qubit_ref();
        consume(TokKind::Semicolon, "';'");

        Command cmd;
        cmd.gate = GateType::Reset;
        cmd.qubits.push_back(q);
        result_.commands.push_back(std::move(cmd));
    }

    // ── c[i] = measure q[j]; ─────────────────────────────────────────────────

    void parse_measure_assign() {
        // ``c[i] = measure q[j];``
        uint32_t cbit = parse_cbit_ref();
        consume(TokKind::Assign, "'='");
        if (cur_.text != "measure")
            throw std::runtime_error("QASM3Absorber: expected 'measure' after 'c[i] ='");
        advance();  // "measure"
        uint32_t qbit = parse_qubit_ref();
        consume(TokKind::Semicolon, "';'");

        Command cmd;
        cmd.gate = GateType::Measure;
        cmd.qubits.push_back(qbit);
        cmd.cbits.push_back(cbit);
        result_.commands.push_back(std::move(cmd));
    }

    // ── if (condition) { body } else { body } ────────────────────────────────

    void parse_if_stmt() {
        consume(TokKind::Ident);   // "if"
        consume(TokKind::LParen, "'('");
        auto [bits, vals] = parse_condition();
        consume(TokKind::RParen, "')'");

        // BranchBegin
        Command begin;
        begin.gate = GateType::BranchBegin;
        for (auto b : bits)  begin.condition_bits.push_back(b);
        for (auto v : vals)  begin.condition_values.push_back(v);
        result_.commands.push_back(begin);

        // Then-body
        parse_body();

        // Optional else
        if (peek_ident("else")) {
            advance();  // "else"
            Command else_cmd;
            else_cmd.gate = GateType::BranchElse;
            result_.commands.push_back(else_cmd);
            parse_body();
        }

        Command end_cmd;
        end_cmd.gate = GateType::BranchEnd;
        result_.commands.push_back(end_cmd);
    }

    // Parse ``{ statements… }``
    void parse_body() {
        consume(TokKind::LBrace, "'{'");
        while (!peek(TokKind::RBrace) && cur_.kind != TokKind::Eof) {
            parse_statement();
        }
        consume(TokKind::RBrace, "'}'");
    }

    // Parse a QASM3 boolean condition:
    //   cond = atom_cond (('&&') atom_cond)*
    //   atom_cond = 'c' '[' INT ']' '==' INT
    std::pair<std::vector<uint32_t>, std::vector<bool>> parse_condition() {
        std::vector<uint32_t> bits;
        std::vector<bool>     vals;
        parse_atom_cond(bits, vals);
        while (peek(TokKind::AmpAmp)) {
            advance();
            parse_atom_cond(bits, vals);
        }
        return {bits, vals};
    }

    void parse_atom_cond(std::vector<uint32_t>& bits, std::vector<bool>& vals) {
        uint32_t cbit = parse_cbit_ref();
        consume(TokKind::Equals, "'=='");
        // The emitter writes ``== true`` / ``== false`` (the form strict
        // external parsers accept for a bit); external programs may compare
        // against 1 / 0 — accept both.
        bool value;
        if (cur_.kind == TokKind::Int) {
            value = consume(TokKind::Int).ival != 0;
        } else if (peek_ident("true") || peek_ident("false")) {
            value = cur_.text == "true";
            advance();
        } else {
            throw std::runtime_error(
                "QASM3Absorber: expected 0/1 or true/false after '==', got '"
                + cur_.text + "'");
        }
        bits.push_back(cbit);
        vals.push_back(value);
    }
};

// ============================================================================
// QASM3Absorber::absorb
// ============================================================================

QASM3Absorber::Result QASM3Absorber::absorb(const std::string& qasm3_source) const {
    Result result;
    Parser parser(qasm3_source, result);
    parser.parse();
    return result;
}

}  // namespace qarpx
