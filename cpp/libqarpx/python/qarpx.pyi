"""OpenQARP C++ Circuit IR — Python bindings"""

from collections.abc import Callable, Mapping, Sequence, Set
import enum
import types
from typing import Annotated, Any, ClassVar, Final, Self, overload

import numpy
from numpy.typing import NDArray
import scipy.sparse
import sympy


def _cpu_budget() -> dict:
    """
    Internal: this process's CPU budget as read now (0 = unknown, or no limit) and the default thread count it gives.
    """

def _openblas_threads_callback_address() -> int:
    """
    Internal: address of the OpenBLAS threading callback that runs BLAS jobs on qarpx's parking pool.
    """

def _blas_callback_invocations() -> int:
    """Internal: calls of the OpenBLAS threading callback so far."""

def _blas_pool_workers() -> int:
    """Internal: workers of the OpenBLAS callback's pool alive now."""

def _configured_thread_count() -> int:
    """Internal: the worker count every qarpx parallel layer uses."""

__abi_version__: int = 15

__source_dir__: str

__with_symengine__: bool

class QubitOperator:
    """
    openfermion-compatible QubitOperator backed by a packed binary-symplectic C++ core (PauliEngine design, arXiv:2601.02233).
    """

    def __init__(self, term: object | None = None, coefficient: object = 1.0) -> None:
        """
        openfermion-compatible constructor: term is None (zero operator), a term string, or a factor tuple; coefficient is any numeric type or a sympy expression (SymEngine backend).
        """

    @property
    def terms(self, /) -> types.MappingProxyType[tuple[tuple[int, str], ...], complex | sympy.Expr]:
        """Term dictionary (read-only view; assign a full dict to replace)."""

    @terms.setter
    def terms(self, value: Mapping[tuple[tuple[int, str], ...], complex], /) -> None: ...

    def coefficient(self, term: object) -> complex | sympy.Expr:
        """Fast single-term coefficient lookup (qarpx extra, not openfermion)."""

    @overload
    def __add__(self, arg: QubitOperator, /) -> QubitOperator: ...

    @overload
    def __add__(self, arg: complex, /) -> QubitOperator: ...

    @overload
    def __add__(self, arg: object, /) -> QubitOperator: ...

    @overload
    def __sub__(self, arg: QubitOperator, /) -> QubitOperator: ...

    @overload
    def __sub__(self, arg: complex, /) -> QubitOperator: ...

    @overload
    def __sub__(self, arg: object, /) -> QubitOperator: ...

    @overload
    def __mul__(self, arg: QubitOperator, /) -> QubitOperator: ...

    @overload
    def __mul__(self, arg: complex, /) -> QubitOperator: ...

    @overload
    def __mul__(self, arg: object, /) -> QubitOperator: ...

    @overload
    def __iadd__(self, arg: QubitOperator, /) -> QubitOperator: ...
    @overload
    def __iadd__(self, arg: complex, /) -> QubitOperator: ...
    @overload
    def __iadd__(self, arg: object, /) -> QubitOperator: ...

    @overload
    def __isub__(self, arg: QubitOperator, /) -> QubitOperator: ...
    @overload
    def __isub__(self, arg: complex, /) -> QubitOperator: ...
    @overload
    def __isub__(self, arg: object, /) -> QubitOperator: ...

    @overload
    def __imul__(self, arg: QubitOperator, /) -> QubitOperator: ...
    @overload
    def __imul__(self, arg: complex, /) -> QubitOperator: ...
    @overload
    def __imul__(self, arg: object, /) -> QubitOperator: ...

    @overload
    def __radd__(self, arg: complex, /) -> QubitOperator: ...

    @overload
    def __radd__(self, arg: object, /) -> QubitOperator: ...

    @overload
    def __rsub__(self, arg: complex, /) -> QubitOperator: ...

    @overload
    def __rsub__(self, arg: object, /) -> QubitOperator: ...

    @overload
    def __rmul__(self, arg: complex, /) -> QubitOperator: ...

    @overload
    def __rmul__(self, arg: object, /) -> QubitOperator: ...

    def __truediv__(self, arg: complex, /) -> QubitOperator: ...

    def __itruediv__(self, arg: complex, /) -> QubitOperator: ...

    def __neg__(self) -> QubitOperator: ...

    def __eq__(self, other: object, /) -> bool: ...

    def __ne__(self, other: object, /) -> bool: ...

    def __pow__(self, exponent: int) -> QubitOperator: ...

    __hash__: ClassVar[None]  # type: ignore[assignment]

    def isclose(self, other: QubitOperator, tol: float = 1e-08) -> bool: ...

    def hermitian_conjugated(self) -> QubitOperator: ...

    def count_qubits(self) -> int: ...

    def compress(self, abs_tol: float = 1e-08) -> None: ...

    def get_operators(self) -> list[QubitOperator]: ...

    def is_symbolic(self) -> bool: ...

    @property
    def constant(self) -> complex: ...

    @constant.setter
    def constant(self, arg: object, /) -> None: ...

    @staticmethod
    def zero() -> QubitOperator: ...

    @staticmethod
    def identity(coefficient: object = 1.0) -> QubitOperator: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __copy__(self) -> QubitOperator: ...

    def __deepcopy__(self, memo: object) -> QubitOperator: ...

    def __getstate__(self) -> list: ...

    def __setstate__(self, arg: list, /) -> None: ...

    def substitute(self, values: object) -> QubitOperator:
        """
        Substitute named symbols with numeric values; demotes to the numeric backend when no free symbols remain (qarpx extra).
        """

    def free_symbols(self) -> list[str]:
        """Sorted names of the free symbols across all coefficients."""

    def sparse_matrix(self, n_qubits: int | None = None) -> scipy.sparse.csc_matrix:
        """
        qarpx-LSB ``scipy.sparse.csc_matrix`` of the operator (qubit q ↔ bit q).

        The only matrix API (conventions §1) — directly contractable with qarpx statevectors / unitaries.  openfermion's MSB layout, for interop only, is :func:`qarp.operators.compat.get_sparse_operator`.  ``n_qubits`` None or negative infers the width from the operator.
        """

class FermionOperator:
    """
    openfermion-compatible FermionOperator backed by a packed ladder-sequence C++ core.
    """

    def __init__(self, term: object | None = None, coefficient: object = 1.0) -> None:
        """
        openfermion-compatible constructor: term is None (zero operator), a term string, or a factor tuple; coefficient is any numeric type or a sympy expression (SymEngine backend).
        """

    @property
    def terms(self, /) -> types.MappingProxyType[tuple[tuple[int, int], ...], complex | sympy.Expr]:
        """Term dictionary (read-only view; assign a full dict to replace)."""

    @terms.setter
    def terms(self, value: Mapping[tuple[tuple[int, int], ...], complex], /) -> None: ...

    def coefficient(self, term: object) -> complex | sympy.Expr:
        """Fast single-term coefficient lookup (qarpx extra, not openfermion)."""

    @overload
    def __add__(self, arg: FermionOperator, /) -> FermionOperator: ...

    @overload
    def __add__(self, arg: complex, /) -> FermionOperator: ...

    @overload
    def __add__(self, arg: object, /) -> FermionOperator: ...

    @overload
    def __sub__(self, arg: FermionOperator, /) -> FermionOperator: ...

    @overload
    def __sub__(self, arg: complex, /) -> FermionOperator: ...

    @overload
    def __sub__(self, arg: object, /) -> FermionOperator: ...

    @overload
    def __mul__(self, arg: FermionOperator, /) -> FermionOperator: ...

    @overload
    def __mul__(self, arg: complex, /) -> FermionOperator: ...

    @overload
    def __mul__(self, arg: object, /) -> FermionOperator: ...

    @overload
    def __iadd__(self, arg: FermionOperator, /) -> FermionOperator: ...
    @overload
    def __iadd__(self, arg: complex, /) -> FermionOperator: ...
    @overload
    def __iadd__(self, arg: object, /) -> FermionOperator: ...

    @overload
    def __isub__(self, arg: FermionOperator, /) -> FermionOperator: ...
    @overload
    def __isub__(self, arg: complex, /) -> FermionOperator: ...
    @overload
    def __isub__(self, arg: object, /) -> FermionOperator: ...

    @overload
    def __imul__(self, arg: FermionOperator, /) -> FermionOperator: ...
    @overload
    def __imul__(self, arg: complex, /) -> FermionOperator: ...
    @overload
    def __imul__(self, arg: object, /) -> FermionOperator: ...

    @overload
    def __radd__(self, arg: complex, /) -> FermionOperator: ...

    @overload
    def __radd__(self, arg: object, /) -> FermionOperator: ...

    @overload
    def __rsub__(self, arg: complex, /) -> FermionOperator: ...

    @overload
    def __rsub__(self, arg: object, /) -> FermionOperator: ...

    @overload
    def __rmul__(self, arg: complex, /) -> FermionOperator: ...

    @overload
    def __rmul__(self, arg: object, /) -> FermionOperator: ...

    def __truediv__(self, arg: complex, /) -> FermionOperator: ...

    def __itruediv__(self, arg: complex, /) -> FermionOperator: ...

    def __neg__(self) -> FermionOperator: ...

    def __eq__(self, other: object, /) -> bool: ...

    def __ne__(self, other: object, /) -> bool: ...

    def __pow__(self, exponent: int) -> FermionOperator: ...

    __hash__: ClassVar[None]  # type: ignore[assignment]

    def isclose(self, other: FermionOperator, tol: float = 1e-08) -> bool: ...

    def hermitian_conjugated(self) -> FermionOperator: ...

    def count_qubits(self) -> int: ...

    def compress(self, abs_tol: float = 1e-08) -> None: ...

    def get_operators(self) -> list[FermionOperator]: ...

    def is_symbolic(self) -> bool: ...

    @property
    def constant(self) -> complex: ...

    @constant.setter
    def constant(self, arg: object, /) -> None: ...

    @staticmethod
    def zero() -> FermionOperator: ...

    @staticmethod
    def identity(coefficient: object = 1.0) -> FermionOperator: ...

    def __str__(self) -> str: ...

    def __repr__(self) -> str: ...

    def __copy__(self) -> FermionOperator: ...

    def __deepcopy__(self, memo: object) -> FermionOperator: ...

    def __getstate__(self) -> list: ...

    def __setstate__(self, arg: list, /) -> None: ...

    def substitute(self, values: object) -> FermionOperator:
        """
        Substitute named symbols with numeric values; demotes to the numeric backend when no free symbols remain (qarpx extra).
        """

    def free_symbols(self) -> list[str]:
        """Sorted names of the free symbols across all coefficients."""

    def sparse_matrix(self, n_qubits: int | None = None) -> scipy.sparse.csc_matrix:
        """
        qarpx-LSB ``scipy.sparse.csc_matrix`` of the Jordan-Wigner image (qubit q ↔ bit q).

        The only matrix API (conventions §1) — directly contractable with qarpx statevectors / unitaries.  openfermion's MSB layout, for interop only, is :func:`qarp.operators.compat.get_sparse_operator`.  ``n_qubits`` None or negative infers the width from the operator.
        """

@overload
def hermitian_conjugated(op: QubitOperator) -> QubitOperator: ...

@overload
def hermitian_conjugated(op: FermionOperator) -> FermionOperator: ...

@overload
def count_qubits(op: QubitOperator) -> int: ...

@overload
def count_qubits(op: FermionOperator) -> int: ...

@overload
def is_hermitian(op: QubitOperator) -> bool: ...

@overload
def is_hermitian(op: FermionOperator) -> bool: ...

@overload
def jordan_wigner(op: FermionOperator) -> QubitOperator: ...

@overload
def jordan_wigner(ops: Sequence[FermionOperator]) -> list[QubitOperator]: ...

@overload
def bravyi_kitaev(op: FermionOperator, n_qubits: object | None = None) -> QubitOperator: ...

@overload
def bravyi_kitaev(ops: Sequence[FermionOperator], n_qubits: object | None = None) -> list[QubitOperator]: ...

@overload
def parity_transform(op: FermionOperator, n_qubits: int) -> QubitOperator: ...

@overload
def parity_transform(ops: Sequence[FermionOperator], n_qubits: int) -> list[QubitOperator]: ...

def qubit_operator_to_observable(op: QubitOperator) -> list[tuple[list[tuple[int, str]], complex]]:
    """
    The qarpx simulator Pauli-sum format: list of (sparse [(qubit, letter)] term, complex coefficient) pairs.
    """

def qubit_operator_coo(op: QubitOperator, n_qubits: object | None = None, msb: bool = False) -> tuple:
    """
    COO triplets (data, rows, cols, dim) of the operator's matrix, one per non-zero cell.  Default is the qarpx LSB ordering (qubit q ↔ bit q); msb=True realizes openfermion's MSB qubit-0 layout for interop.
    """

def fermion_operator_coo(op: FermionOperator, n_qubits: object | None = None, msb: bool = False) -> tuple: ...

class GateType(enum.Enum):
    X = 0

    Y = 1

    Z = 2

    H = 3

    S = 4

    Sdg = 5

    T = 6

    Tdg = 7

    SX = 8

    SXdg = 9

    Id = 10

    Rx = 11

    Ry = 12

    Rz = 13

    P = 14

    U = 15

    CX = 16

    CY = 17

    CZ = 18

    SWAP = 19

    ECR = 20

    iSWAP = 21

    iSWAPdg = 22

    CH = 23

    CS = 24

    CSdg = 25

    CSX = 26

    CSXdg = 27

    CRx = 28

    CRy = 29

    CRz = 30

    CP = 31

    RZZ = 32

    RXX = 33

    RYY = 34

    CU = 35

    CCX = 36

    CSWAP = 37

    MCZ = 38

    GPhase = 39

    Barrier = 40

    Measure = 41

    Reset = 42

    BranchBegin = 43

    BranchElse = 44

    BranchEnd = 45

    Custom = 46

class Pauli(enum.Enum):
    I = 0

    X = 1

    Y = 2

    Z = 3

def parse_pauli_string(s: str) -> list[Pauli]:
    """
    Parse a string like 'IXYZ' into a list[Pauli].  Accepts upper/lower case; throws on empty or invalid characters.
    """

def commutes(a: Sequence[Pauli], b: Sequence[Pauli]) -> bool:
    """Pairwise commutation predicate for multi-qubit Paulis."""

def compute_basis_change_clifford(paulis: Sequence[Sequence[Pauli]], n_qubits: int) -> tuple[list[Command], list[list[bool]], list[bool]]:
    """
    Diagonalise a set of mutually-commuting multi-qubit Paulis to a Z-only basis.  Returns (clifford_commands, z_rows, signs): apply clifford_commands (in order) to the state, then measure every qubit; Pauli r then equals (signs[r] ? -1 : +1) * (tensor of Z over the qubits where z_rows[r][q] is true).  Raises ValueError if the input is not pairwise commuting.
    """

class Param:
    def __init__(self, value: float) -> None: ...

    @staticmethod
    def symbol(name: str) -> Param: ...

    @staticmethod
    def linear(coeff: float, symbol_name: str, offset: float = 0.0) -> Param: ...

    def is_symbolic(self) -> bool: ...

    def is_concrete(self) -> bool: ...

    def value(self) -> float: ...

    def free_symbols(self) -> list[str]: ...

    def substitute(self, values: Mapping[str, float]) -> Param: ...

    def rename_symbols(self, mapping: Mapping[str, str]) -> Param: ...

    def evaluate(self, values: Mapping[str, float]) -> float: ...

    def approx_equal(self, other: Param | float, atol: float = 1e-09) -> bool: ...

    def __deepcopy__(self, memo: object) -> Param: ...

    def __eq__(self, other: object, /) -> bool: ...

    def __ne__(self, other: object, /) -> bool: ...

    def __neg__(self) -> Param: ...

    def __repr__(self) -> str: ...

    def __add__(self, other: Param | float) -> Param: ...

    def __sub__(self, other: Param | float) -> Param: ...

    def __mul__(self, other: Param | float) -> Param: ...

    def __truediv__(self, other: Param | float) -> Param: ...

def symbol(name: str) -> Param:
    """Create a symbolic parameter"""

def affine_coefficients(param: Param | float) -> dict[str, float]:
    """
    {symbol: d(param)/d(symbol)} for every free symbol, verified affine; raises CapabilityError otherwise.
    """

def concrete(value: float) -> Param:
    """Create a concrete parameter"""

class Command:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, gate: GateType, q: int) -> None: ...

    @overload
    def __init__(self, gate: GateType, q0: int, q1: int) -> None: ...

    @overload
    def __init__(self, gate: GateType, q: int, param: Param | float) -> None: ...

    @overload
    def __init__(self, gate: GateType, q0: int, q1: int, param: Param | float) -> None: ...

    @property
    def gate(self) -> GateType: ...

    @gate.setter
    def gate(self, arg: GateType, /) -> None: ...

    def __eq__(self, other: object, /) -> bool: ...

    def __ne__(self, other: object, /) -> bool: ...

    @property
    def qubits(self) -> list[int]: ...

    @property
    def params(self) -> list[Param]: ...

    @property
    def cbits(self) -> list[int]: ...

    @property
    def condition_bits(self) -> list[int]: ...

    @property
    def condition_values(self) -> list[bool]: ...

    def is_parametric(self) -> bool: ...

    def dagger(self) -> Command: ...

    def remap_qubits(self, mapping: Sequence[int]) -> Command: ...

    def substitute(self, values: Mapping[str, float]) -> Command: ...

    def rename_symbols(self, mapping: Mapping[str, str]) -> Command: ...

    def approx_equal(self, other: Command, atol: float = 1e-09) -> bool: ...

    def __repr__(self) -> str: ...

def substitute_all(commands: Sequence[Command], values: Mapping[str, float]) -> list[Command]:
    """
    Return a new command sequence with all symbolic parameters substituted.
    """

def rename_symbol_occurrences(commands: Sequence[Command], symbol: str, prefix: str) -> tuple[list[Command], list[tuple[int, int, str]]]:
    """
    (renamed commands, [(command, param, private_name), ...]): every occurrence of `symbol` gets its own private name for per-occurrence parameter shifts.
    """

def commands_equal(a: Sequence[Command], b: Sequence[Command], atol: float = 1e-09) -> bool:
    """Order-sensitive tolerant equality between two command sequences."""

def cbit_register_width(commands: Sequence[Command]) -> int:
    """
    Width of the classical register a command stream needs: 1 + the largest cbit index referenced by a Measure write or a condition read, or 0 if none is.
    """

def gate_is_physical(gate: GateType) -> bool:
    """
    True iff this GateType applies a physical operation to the register (False for Barrier/Measure/Reset/GPhase/branch markers) — the single classification behind n_nqb_gates and qarp.resources counting.
    """

def n_nqb_gates(commands: Sequence[Command], k: int) -> int:
    """
    Number of k-qubit gates in a command stream (excludes Barrier/Measure/Reset/GPhase/branch markers — see gate_is_physical).
    """

def n_physical_gates(commands: Sequence[Command]) -> int:
    """
    Total physical gate count over all arities (same exclusions as n_nqb_gates) — the resource vector's headline n_gates.
    """

def n_gates_of_type(commands: Sequence[Command], gate: GateType) -> int:
    """Number of commands of the given GateType — unfiltered."""

def uninitialised_condition_cbits(commands: Sequence[Command]) -> list[int]:
    """
    Cbits read as a classical condition before any Measure wrote them. Empty means every condition has a preceding write. Evaluate on a complete flattened program — a sub-block whose measurement lives in an enclosing block will report its condition cbits here.
    """

class Block:
    def build(self) -> Self: ...

    @property
    def is_built(self) -> bool: ...

    def flatten(self) -> list[Command]: ...

    def equals(self, other: Block, atol: float = 1e-09) -> bool:
        """Tolerant 'same circuit' check — see Block::equals in block.h."""

    def __eq__(self, other: object, /) -> bool: ...

    def __ne__(self, other: object, /) -> bool: ...

    def commands(self) -> list[Command]: ...

    def set_commands(self, commands: Sequence[Command]) -> None:
        """
        Replace the local command buffer.  Used by Python __deepcopy__ to clone ad-hoc-populated blocks.
        """

    def set_built(self, built: bool) -> None:
        """
        Force the C++ built_ flag.  Used by Python __deepcopy__ to mirror the source block's built state.
        """

    def _copy_base_state_from(self, src: Block) -> None:
        """
        Clone-internal: compiler-complete copy of all base Block fields (commands, built flag, name, sizes, targets, controls).  Tree edges live in subclasses and are untouched.
        """

    def __deepcopy__(self, memo: dict) -> Self:
        """
        Deep clone for raw C++-created blocks (subclasses inherit; qarp's Python mixin __deepcopy__ shadows this).
        """

    def free_symbols(self) -> list[str]: ...

    def dagger(self) -> Block: ...

    def set_symbols(self, symbol_parameter_map: Mapping[str, float]) -> Block: ...

    def replace_symbols(self, new_parameters: Mapping[str, str]) -> Block: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def n_qubits(self) -> int: ...

    @n_qubits.setter
    def n_qubits(self, arg: int, /) -> None: ...

    @property
    def n_cbits(self) -> int: ...

    @n_cbits.setter
    def n_cbits(self, arg: int, /) -> None: ...

    @property
    def target_qubits(self) -> list[int] | None: ...

    @target_qubits.setter
    def target_qubits(self, arg: Sequence[int] | None, /) -> None: ...

    @property
    def target_cbits(self) -> list[int] | None: ...

    @target_cbits.setter
    def target_cbits(self, arg: Sequence[int] | None, /) -> None: ...

    @overload
    def h(self, q: int) -> Block: ...

    @overload
    def h(self, q: Sequence[int]) -> Block: ...

    @overload
    def x(self, q: int) -> Block: ...

    @overload
    def x(self, q: Sequence[int]) -> Block: ...

    @overload
    def y(self, q: int) -> Block: ...

    @overload
    def y(self, q: Sequence[int]) -> Block: ...

    @overload
    def z(self, q: int) -> Block: ...

    @overload
    def z(self, q: Sequence[int]) -> Block: ...

    @overload
    def s(self, q: int) -> Block: ...

    @overload
    def s(self, q: Sequence[int]) -> Block: ...

    @overload
    def sdg(self, q: int) -> Block: ...

    @overload
    def sdg(self, q: Sequence[int]) -> Block: ...

    @overload
    def t(self, q: int) -> Block: ...

    @overload
    def t(self, q: Sequence[int]) -> Block: ...

    @overload
    def tdg(self, q: int) -> Block: ...

    @overload
    def tdg(self, q: Sequence[int]) -> Block: ...

    @overload
    def sx(self, q: int) -> Block: ...

    @overload
    def sx(self, q: Sequence[int]) -> Block: ...

    @overload
    def sxdg(self, q: int) -> Block: ...

    @overload
    def sxdg(self, q: Sequence[int]) -> Block: ...

    @overload
    def id(self, q: int) -> Block: ...

    @overload
    def id(self, q: Sequence[int]) -> Block: ...

    @overload
    def rx(self, q: int, angle: Param | float) -> Block: ...

    @overload
    def rx(self, q: Sequence[tuple[int, Param | float]]) -> Block: ...

    @overload
    def ry(self, q: int, angle: Param | float) -> Block: ...

    @overload
    def ry(self, q: Sequence[tuple[int, Param | float]]) -> Block: ...

    @overload
    def rz(self, q: int, angle: Param | float) -> Block: ...

    @overload
    def rz(self, q: Sequence[tuple[int, Param | float]]) -> Block: ...

    @overload
    def p(self, q: int, angle: Param | float) -> Block: ...

    @overload
    def p(self, q: Sequence[tuple[int, Param | float]]) -> Block: ...

    @overload
    def cx(self, c: int, t: int) -> Block: ...

    @overload
    def cx(self, c: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def cy(self, c: int, t: int) -> Block: ...

    @overload
    def cy(self, c: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def cz(self, c: int, t: int) -> Block: ...

    @overload
    def cz(self, c: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def ch(self, c: int, t: int) -> Block: ...

    @overload
    def ch(self, c: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def cs(self, c: int, t: int) -> Block: ...

    @overload
    def cs(self, c: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def csdg(self, c: int, t: int) -> Block: ...

    @overload
    def csdg(self, c: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def csx(self, c: int, t: int) -> Block: ...

    @overload
    def csx(self, c: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def csxdg(self, c: int, t: int) -> Block: ...

    @overload
    def csxdg(self, c: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def swap(self, q0: int, q1: int) -> Block: ...

    @overload
    def swap(self, q0: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def ecr(self, q0: int, q1: int) -> Block: ...

    @overload
    def ecr(self, q0: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def iswap(self, q0: int, q1: int) -> Block: ...

    @overload
    def iswap(self, q0: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def iswapdg(self, q0: int, q1: int) -> Block: ...

    @overload
    def iswapdg(self, q0: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def crx(self, c: int, t: int, angle: Param | float) -> Block: ...

    @overload
    def crx(self, c: Sequence[tuple[int, int, Param | float]]) -> Block: ...

    @overload
    def cry(self, c: int, t: int, angle: Param | float) -> Block: ...

    @overload
    def cry(self, c: Sequence[tuple[int, int, Param | float]]) -> Block: ...

    @overload
    def crz(self, c: int, t: int, angle: Param | float) -> Block: ...

    @overload
    def crz(self, c: Sequence[tuple[int, int, Param | float]]) -> Block: ...

    @overload
    def cp(self, c: int, t: int, angle: Param | float) -> Block: ...

    @overload
    def cp(self, c: Sequence[tuple[int, int, Param | float]]) -> Block: ...

    @overload
    def rzz(self, q0: int, q1: int, angle: Param | float) -> Block: ...

    @overload
    def rzz(self, q0: Sequence[tuple[int, int, Param | float]]) -> Block: ...

    @overload
    def rxx(self, q0: int, q1: int, angle: Param | float) -> Block: ...

    @overload
    def rxx(self, q0: Sequence[tuple[int, int, Param | float]]) -> Block: ...

    @overload
    def ryy(self, q0: int, q1: int, angle: Param | float) -> Block: ...

    @overload
    def ryy(self, q0: Sequence[tuple[int, int, Param | float]]) -> Block: ...

    @overload
    def ccx(self, c0: int, c1: int, t: int) -> Block: ...

    @overload
    def ccx(self, c0: Sequence[tuple[int, int, int]]) -> Block: ...

    @overload
    def cswap(self, c: int, q0: int, q1: int) -> Block: ...

    @overload
    def cswap(self, c: Sequence[tuple[int, int, int]]) -> Block: ...

    def mcz(self, qubits: Sequence[int]) -> Block: ...

    def gphase(self, angle: Param | float) -> Block: ...

    def u(self, q: int, theta: Param | float, phi: Param | float, lam: Param | float) -> Block: ...

    def cu(self, c: int, t: int, theta: Param | float, phi: Param | float, lam: Param | float, gamma: Param | float) -> Block: ...

    @overload
    def measure(self, qubit: int, cbit: int) -> Block: ...

    @overload
    def measure(self, qubit: Sequence[tuple[int, int]]) -> Block: ...

    @overload
    def reset(self, qubit: int) -> Block: ...

    @overload
    def reset(self, qubit: Sequence[int]) -> Block: ...

    def state_preparation(self, amplitudes: Sequence[complex]) -> Block:
        """Prepare an arbitrary 2^n-amplitude state from |0…0⟩ (Möttönen)."""

    def diagonal_unitary(self, diagonal_elements: Sequence[complex]) -> Block:
        """
        Implement an arbitrary 2^n × 2^n diagonal unitary (Shende-Bullock-Markov).
        """

    def unitary_synthesis(self, U: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='F')]) -> Block:
        """
        Implement an arbitrary 2^n × 2^n unitary via Quantum Shannon Decomposition.
        """

    @overload
    def pauli_exp(self, pauli: str, angle: Param | float) -> Block:
        """Append exp(-i · angle / 2 · pauli) for a single multi-qubit Pauli."""

    @overload
    def pauli_exp(self, pauli: Sequence[Pauli], angle: Param | float) -> Block: ...

    @overload
    def commuting_pauli_set_exp(self, paulis: Sequence[str], angles: Sequence[Param | float]) -> Block:
        """
        Append Π_i exp(-i · angles[i] / 2 · paulis[i]) for a set of mutually commuting Pauli strings.  Validates lengths and pairwise commutation.
        """

    @overload
    def commuting_pauli_set_exp(self, paulis: Sequence[Sequence[Pauli]], angles: Sequence[Param | float]) -> Block: ...

class SimpleBlock(Block):
    def __init__(self, n_qubits: int, name: str = 'Circuit') -> None: ...

    def add_register(self, name: str, size: int) -> list[int]:
        """
        Allocate next `size` qubit indices for a named register. Returns list of qubit indices [next, ..., next+size-1].
        """

class CompositeBlock(Block):
    def __init__(self, children: Sequence[Block], n_qubits: int = 0, name: str = 'CompositeBlock') -> None: ...

    def add_child(self, child: Block) -> None: ...

    def children(self) -> list[Block]: ...

    def uninitialised_condition_cbits(self) -> list[int]:
        """
        Cbits this block's flattened program reads as a classical condition before any Measure wrote them. Empty means every condition has a preceding write. See qx.uninitialised_condition_cbits for the free-function form.
        """

class ControlledBlock(Block):
    def __init__(self, inner: Block, num_controls: int, ctrl_state: Sequence[bool], name: str = 'Controlled') -> None: ...

    def inner(self) -> Block: ...

    @property
    def num_controls(self) -> int: ...

    @property
    def ctrl_state(self) -> list[bool]: ...

class MeasureBlock(Block):
    def __init__(self, qubit: int, cbit: int, name: str = 'Measure') -> None: ...

    @property
    def qubit_index(self) -> int: ...

    @property
    def cbit_index(self) -> int: ...

class ResetBlock(Block):
    def __init__(self, qubit: int, name: str = 'Reset') -> None: ...

    @property
    def qubit_index(self) -> int: ...

class ConditionalBlock(Block):
    @overload
    def __init__(self, cbits: Sequence[int], values: Sequence[bool], then_body: Block, else_body: Block | None = None, name: str = 'Conditional') -> None: ...

    @overload
    def __init__(self, cbits: Sequence[int], values: Sequence[bool], then_body: None, *, else_body: Block, name: str = 'Conditional') -> None: ...

    @property
    def then_body(self, /) -> Block | None: ...

    @property
    def else_body(self, /) -> Block | None: ...

    @property
    def condition_cbits(self) -> list[int]: ...

    @property
    def condition_values(self) -> list[bool]: ...

def conditional(cbits: Sequence[int], values: Sequence[bool], *, then: object | None = None, else_: object | None = None, n_qubits: int, name: str = 'Conditional') -> ConditionalBlock:
    """
    Build a ConditionalBlock from `then` and/or `else_` callables, each of which receives a fresh SimpleBlock to populate.
    """

class GateSet:
    def __init__(self) -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def allowed(self) -> set[GateType]: ...

    @allowed.setter
    def allowed(self, arg: Set[GateType], /) -> None: ...

    @property
    def rules(self) -> str: ...

    @rules.setter
    def rules(self, arg: str, /) -> None: ...

    def contains(self, gate: GateType) -> bool: ...

def native_gateset() -> GateSet: ...

def qulacs_gateset() -> GateSet: ...

def qulacs_emitter_gateset() -> GateSet: ...

def qiskit_gateset() -> GateSet: ...

def pytket_gateset() -> GateSet: ...

def pennylane_gateset() -> GateSet: ...

def cudaq_gateset() -> GateSet: ...

class EmitterCapabilities:
    def __init__(self) -> None: ...

    @property
    def symbolic_params(self) -> bool: ...

    @symbolic_params.setter
    def symbolic_params(self, arg: bool, /) -> None: ...

    @property
    def multi_symbol_params(self) -> bool: ...

    @multi_symbol_params.setter
    def multi_symbol_params(self, arg: bool, /) -> None: ...

    @property
    def distinct_cbits(self) -> bool: ...

    @distinct_cbits.setter
    def distinct_cbits(self, arg: bool, /) -> None: ...

    @property
    def conditionals(self) -> bool: ...

    @conditionals.setter
    def conditionals(self, arg: bool, /) -> None: ...

    @property
    def multibit_conditions(self) -> bool: ...

    @multibit_conditions.setter
    def multibit_conditions(self, arg: bool, /) -> None: ...

    @property
    def multibit_else(self) -> bool: ...

    @multibit_else.setter
    def multibit_else(self, arg: bool, /) -> None: ...

    @property
    def custom_unitary(self) -> bool: ...

    @custom_unitary.setter
    def custom_unitary(self, arg: bool, /) -> None: ...

    @property
    def cu_global_phase(self) -> bool: ...

    @cu_global_phase.setter
    def cu_global_phase(self, arg: bool, /) -> None: ...

class Incompatibility:
    @property
    def command(self) -> Command: ...

    @property
    def reason(self) -> str: ...

def clifford_t_gateset() -> GateSet: ...

def clifford_t_rz_gateset() -> GateSet: ...

def full_gateset_1q() -> GateSet: ...

def full_gateset_2q() -> GateSet: ...

def full_gateset_1q_2q() -> GateSet: ...

def universal_gateset() -> GateSet: ...

def routable_subset(gate_set: GateSet) -> GateSet: ...

class Architecture:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, n_qubits: int, edges: Sequence[tuple[int, int]], name: str = '') -> None: ...

    @property
    def name(self) -> str: ...

    @name.setter
    def name(self, arg: str, /) -> None: ...

    @property
    def n_qubits(self) -> int: ...

    @n_qubits.setter
    def n_qubits(self, arg: int, /) -> None: ...

    @property
    def edges(self) -> list[tuple[int, int]]:
        """Edge list; assigning rebuilds the adjacency indices."""

    @edges.setter
    def edges(self, arg: Sequence[tuple[int, int]], /) -> None: ...

    def is_connected(self, a: int, b: int) -> bool: ...

    def has_directed_edge(self, a: int, b: int) -> bool: ...

    def neighbours(self, q: int) -> list[int]: ...

    def shortest_path(self, a: int, b: int) -> list[int]: ...

def nearest_neighbour_architecture(xdim: int, ydim: int) -> Architecture: ...

def all_to_all_architecture(n_qubits: int) -> Architecture: ...

class NoiseModel:
    def __init__(self) -> None: ...

    @property
    def enabled(self) -> bool: ...

    @enabled.setter
    def enabled(self, arg: bool, /) -> None: ...

    def has_channel(self, gate: GateType) -> bool: ...

    def has_any_channel(self) -> bool: ...

    def __iadd__(self, arg: NoiseModel, /) -> NoiseModel: ...

    def __add__(self, arg: NoiseModel, /) -> NoiseModel: ...

def make_depolarizing(p: float, gates: Sequence[GateType]) -> NoiseModel: ...

def make_pauli(p_x: float, p_y: float, p_z: float, gates: Sequence[GateType]) -> NoiseModel: ...

def make_bit_flip(p: float, gates: Sequence[GateType]) -> NoiseModel: ...

def make_amplitude_damping(p: float, gates: Sequence[GateType]) -> NoiseModel: ...

class Device:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, n_qubits: int) -> None: ...

    @overload
    def __init__(self, n_qubits: int, architecture: Architecture | None = None, noise_model: NoiseModel | None = None, gate_set: GateSet | None = None, directedness: bool = False) -> None: ...

    @property
    def n_qubits(self) -> int: ...

    @n_qubits.setter
    def n_qubits(self, arg: int, /) -> None: ...

    @property
    def architecture(self) -> Architecture | None: ...

    @architecture.setter
    def architecture(self, arg: Architecture | None, /) -> None: ...

    @property
    def noise_model(self) -> NoiseModel | None: ...

    @noise_model.setter
    def noise_model(self, arg: NoiseModel | None, /) -> None: ...

    @property
    def gate_set(self) -> GateSet | None: ...

    @gate_set.setter
    def gate_set(self, arg: GateSet | None, /) -> None: ...

    @property
    def directedness(self) -> bool: ...

    @directedness.setter
    def directedness(self, arg: bool, /) -> None: ...

    def check_fits(self, needed: int) -> None: ...

class CompiledCircuit:
    @property
    def commands(self) -> list[Command]: ...

    @property
    def initial_logical_to_physical(self) -> list[int]: ...

    @property
    def final_logical_to_physical(self) -> list[int]: ...

class RouterKind(enum.Enum):
    Lite = 0

    Sabre = 1

def compile_for_device(commands: Sequence[Command], n_qubits: int, device: Device, router: RouterKind = RouterKind.Sabre) -> CompiledCircuit:
    """
    Run check_fits → rebase → route → (directed SWAP lowering) → rebase → assert_directions against `device`.
    """

def assert_directions(commands: Sequence[Command], arch: Architecture) -> None:
    """
    Raise CapabilityError if an asymmetric 2-qubit gate sits on a pair that is not a stored directed edge of `arch`.
    """

class CircuitDAG:
    @overload
    @staticmethod
    def from_commands(commands: Sequence[Command]) -> CircuitDAG:
        """Build the wire-dependency DAG; qubit/cbit counts are inferred."""

    @overload
    @staticmethod
    def from_commands(commands: Sequence[Command], n_qubits: int, n_cbits: int) -> CircuitDAG: ...

    def to_commands(self) -> list[Command]:
        """Deterministic linearization (RT-1: exact round-trip)."""

    def depth(self) -> int:
        """Longest dependency path (Barrier/GPhase weigh 0)."""

    def layers(self) -> list[list[int]]:
        """ASAP moments: lists of node ids, dependency-levelled."""

    def front_layer(self) -> list[int]: ...

    def count_ops(self) -> dict[str, int]:
        """Gate-name → count over the emitted stream (region interiors included)."""

    def command(self, node: int) -> Command: ...

    def is_region(self, node: int) -> bool: ...

    def region_commands(self, node: int) -> list[Command]: ...

    def is_removed(self, node: int) -> bool: ...

    def wires(self, node: int) -> list[int]: ...

    @property
    def n_qubits(self) -> int: ...

    @property
    def n_cbits(self) -> int: ...

    @property
    def n_nodes(self) -> int: ...

    @property
    def n_slots(self) -> int: ...

class RoutingOptions:
    def __init__(self) -> None: ...

    @property
    def arch(self) -> Architecture: ...

    @arch.setter
    def arch(self, arg: Architecture, /) -> None: ...

    @property
    def directedness(self) -> bool: ...

    @directedness.setter
    def directedness(self, arg: bool, /) -> None: ...

    @property
    def initial_mapping(self) -> list[int] | None: ...

    @initial_mapping.setter
    def initial_mapping(self, arg: Sequence[int] | None, /) -> None: ...

    @property
    def router(self) -> RouterKind: ...

    @router.setter
    def router(self, arg: RouterKind, /) -> None: ...

class RoutingResult:
    @property
    def commands(self) -> list[Command]: ...

    @property
    def initial_logical_to_physical(self) -> list[int]: ...

    @property
    def final_logical_to_physical(self) -> list[int]: ...

def route(commands: Sequence[Command], options: RoutingOptions) -> RoutingResult:
    """
    Route onto options.arch.  options.router picks Sabre (default: lookahead heuristic plus initial-mapping search) or Lite (greedy shortest-path sweep); both H-conjugate CX for direction flips on directed architectures.
    """

def reindex_sampling_result(physical: SamplingResult, final_logical_to_physical: Sequence[int]) -> SamplingResult:
    """
    Permute a SamplingResult from per-physical-qubit bit ordering back to per-logical-qubit bit ordering.
    """

class OptLevel(enum.Enum):
    O0 = 0

    O1 = 1

    O2 = 2

class Transpiler:
    def __init__(self, gate_set: GateSet) -> None: ...

    def transpile(self, input: Sequence[Command]) -> list[Command]: ...

    def transpile_and_optimize(self, input: Sequence[Command], level: OptLevel = OptLevel.O1) -> list[Command]: ...

    def optimize_in_target(self, input: Sequence[Command], level: OptLevel = OptLevel.O1) -> list[Command]:
        """
        Optimize an in-target sequence without leaving the target: DAG passes, then single-qubit fusion only if the target admits Custom.
        """

    def transpile_parallel(self, blocks: Sequence[Sequence[Command]], n_threads: int = 0) -> list[list[Command]]: ...

    def register_decomposition(self, gate: GateType, fn: Callable[[Command], Sequence[Command]]) -> None:
        """
        Register (or override) the rule lowering `gate`: fn(Command) -> list[Command].
        """

    def install_clifford_t_rz_decompositions(self) -> None:
        """
        Install the Rx/Ry/SWAP → Rz+Clifford decompositions on top of the built-in table.
        """

    def verify_closure(self) -> None:
        """Throw if any GateType cannot be reduced to the target gate set."""

    def unreachable_gates(self) -> list[GateType]:
        """
        Return GateTypes that the rule table cannot reduce into the target, sorted by name.
        """

def eliminate_identities(commands: Sequence[Command]) -> tuple[list[Command], int]:
    """
    Eliminate identity patterns. Returns (optimized_commands, num_eliminated).
    """

def gate_name(gate: GateType) -> str: ...

def gate_num_params(gate: GateType) -> int: ...

def gate_num_qubits(gate: GateType) -> int: ...

class QIREmitter:
    def __init__(self) -> None: ...

    def target_name(self) -> str: ...

    def gate_set(self) -> GateSet: ...

    def capabilities(self) -> EmitterCapabilities: ...

    def emit(self, commands: Sequence[Command], n_qubits: int, function_name: str = 'circuit') -> str: ...

    def emit_to_file(self, commands: Sequence[Command], n_qubits: int, path: str, function_name: str = 'circuit') -> None: ...

    def validate(self, commands: Sequence[Command]) -> Incompatibility | None:
        """
        First Incompatibility in the sequence, or None if the whole sequence can cross.
        """

class QASM3Emitter:
    def __init__(self) -> None: ...

    def target_name(self) -> str: ...

    def gate_set(self) -> GateSet: ...

    def capabilities(self) -> EmitterCapabilities: ...

    def emit(self, commands: Sequence[Command], n_qubits: int, circuit_name: str = 'circuit') -> str: ...

    def emit_to_file(self, commands: Sequence[Command], n_qubits: int, path: str, circuit_name: str = 'circuit') -> None: ...

    def validate(self, commands: Sequence[Command]) -> Incompatibility | None:
        """
        First Incompatibility in the sequence, or None if the whole sequence can cross.
        """

class QASM2Emitter:
    def __init__(self) -> None: ...

    def target_name(self) -> str: ...

    def gate_set(self) -> GateSet: ...

    def capabilities(self) -> EmitterCapabilities: ...

    def emit(self, commands: Sequence[Command], n_qubits: int, circuit_name: str = 'circuit') -> str: ...

    def emit_to_file(self, commands: Sequence[Command], n_qubits: int, path: str, circuit_name: str = 'circuit') -> None: ...

    def validate(self, commands: Sequence[Command]) -> Incompatibility | None:
        """
        First Incompatibility in the sequence, or None if the whole sequence can cross.
        """

class QASM3Absorber:
    def __init__(self) -> None: ...

    def absorb(self, qasm3_source: str) -> tuple[list[Command], int, int, list[str]]:
        """
        Parse an OpenQASM 3.0 string into (commands, n_qubits, n_cbits, free_symbols). Supports the subset produced by QASM3Emitter.
        """

    def source_name(self) -> str: ...

class QASM2Absorber:
    def __init__(self) -> None: ...

    def absorb(self, qasm2_source: str) -> tuple[list[Command], int, int, list[str]]:
        """
        Parse an OpenQASM 2.0 string into (commands, n_qubits, n_cbits, free_symbols). free_symbols is always empty — OpenQASM 2 has no symbolic parameter.
        """

    def source_name(self) -> str: ...

class SamplingResult:
    @property
    def n_qubits(self) -> int: ...

    @property
    def n_shots(self) -> int: ...

    @property
    def n_cbits(self) -> int: ...

    @property
    def counts(self) -> dict[int, int]: ...

    @property
    def cbit_history(self) -> list[list[bool]]: ...

    @property
    def statevector(self) -> list[complex]: ...

    def probability(self, outcome: int) -> float: ...

class Program:
    def __init__(self) -> None: ...

    def add_gates(self, commands: Sequence[Command]) -> None: ...

    def add_permutation(self, qubits: Sequence[int], table: Annotated[NDArray[numpy.int64], dict(shape=(None,), order='C', device='cpu', writable=False)]) -> None:
        """
        |x> -> |table[x]> on qubits (local bit b <-> qubits[b]); table is a 1-D int64 bijection of range(2**len(qubits)).
        """

    def add_dense(self, qubits: Sequence[int], matrix: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='F')]) -> None: ...

    def add_controlled_powers(self, controls: Sequence[int], exponents: Sequence[int], targets: Sequence[int], matrix: Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='F')]) -> None:
        """matrix**exponents[j] on targets wherever control j is |1>."""

    def kinds(self) -> list[str]: ...

    def __len__(self) -> int: ...

    @property
    def min_register_width(self) -> int: ...

    def substituted(self, params: Mapping[str, float]) -> Program: ...

    def without_measurements(self) -> Program: ...

def _permutation_table(commands: Sequence[Command], qubits: Sequence[int], max_rest: int, max_work_log2: int) -> Annotated[NDArray[numpy.int64], dict(shape=(None,))] | None:
    """
    Internal: the basis permutation commands apply to qubits, exact including phase, or None (see program.h permutation_table).
    """

def _local_unitary(commands: Sequence[Command], qubits: Sequence[int]) -> Annotated[NDArray[numpy.complex128], dict(shape=(None, None), order='F')]:
    """Internal: the 2^k x 2^k unitary commands apply to qubits (k <= 12)."""

def _commands_digest(commands: Sequence[Command]) -> int:
    """Internal: 64-bit digest of a command stream (a cache key)."""

def _flatten_digest(block: Block) -> int:
    """
    Internal: _commands_digest of block.flatten(), computed without materialising the commands in Python.
    """

def _local_commands_digest(commands: Sequence[Command], qubits: Sequence[int]) -> int:
    """Internal: digest of commands remapped onto range(len(qubits))."""

def _parts_digest(parts: Sequence[tuple[Sequence[Command], int]]) -> int:
    """
    Internal: _commands_digest of each part's commands repeated count times, in order, without materialising the stream.
    """

class QarpSimulator:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, noise_model: NoiseModel) -> None: ...

    @property
    def noise_model(self) -> NoiseModel: ...

    @noise_model.setter
    def noise_model(self, arg: NoiseModel, /) -> None: ...

    @property
    def fusion_max_qubits(self) -> int:
        """
        Gate fusion before statevector/run dispatch: 0 = none, 1 = consecutive single-qubit gates only, k >= 2 = dense blocks of up to k qubits.  Default from QARP_FUSION_MAX_QUBITS (read once per process) else the built-in default; at most MAX_FUSION_QUBITS.
        """

    @fusion_max_qubits.setter
    def fusion_max_qubits(self, arg: int, /) -> None: ...

    @property
    def fusion_min_qubits(self) -> int:
        """
        Registers narrower than this get single-qubit fusion only (the dense pass costs more than it saves there); 0 = always fuse.
        """

    @fusion_min_qubits.setter
    def fusion_min_qubits(self, arg: int, /) -> None: ...

    MAX_FUSION_QUBITS: Final[int] = ...
    """(arg: object, /) -> int"""

    DEFAULT_FUSION_QUBITS: Final[int] = ...
    """(arg: object, /) -> int"""

    DEFAULT_FUSION_MIN_QUBITS: Final[int] = ...
    """(arg: object, /) -> int"""

    def run(self, commands: Sequence[Command], n_qubits: int, n_shots: int, seed: int | None = None, *, initial_state: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> SamplingResult:
        """
        initial_state seeds the register with LSB-indexed amplitudes (length 2^n_qubits, unit norm within 1e-10) instead of |0...0>.
        """

    def batch_run(self, commands: Sequence[Command], n_qubits: int, n_shots: int, param_sets: Sequence[Mapping[str, float]], seed: int | None = None) -> list[SamplingResult]: ...

    def statevector(self, commands: Sequence[Command], n_qubits: int, *, initial_state: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,))]:
        """
        Full statevector as a zero-copy 1-D complex128 numpy array.  initial_state: same contract as run().
        """

    def program_statevector(self, program: Program, n_qubits: int, *, initial_state: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,))]:
        """
        statevector() for a structured program; initial_state: same contract as run().
        """

    def program_run(self, program: Program, n_qubits: int, n_shots: int, seed: int | None = None, *, initial_state: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> SamplingResult:
        """run()'s terminal sample-once path for a structured program."""

    def expectation(self, commands: Sequence[Command], n_qubits: int, observable: Sequence[tuple[Sequence[tuple[int, str]], complex]], *, initial_state: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> float:
        """
        Re <psi|H|psi> for psi = circuit(initial_state or |0...0>), contracted in C++ without returning the statevector.  Observable format as run_gradient; initial_state: same contract as run().
        """

    def transition(self, bra: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)], ket: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)], n_qubits: int, observable: Sequence[tuple[Sequence[tuple[int, str]], complex]]) -> complex:
        """
        <bra|H|ket> over two host statevectors (contiguous complex128, length 2**n_qubits, LSB-indexed); H need not be Hermitian.
        """

    def batch_expectation(self, commands: Sequence[Command], n_qubits: int, observable: Sequence[tuple[Sequence[tuple[int, str]], complex]], param_sets: Sequence[Mapping[str, float]]) -> list[float]:
        """
        expectation() swept over parameter sets — one <H> per set.  Same signature as CudaqSimulator.batch_expectation.
        """

    def unitary_matrix(self, commands: Sequence[Command], n_qubits: int) -> list[list[complex]]:
        """Compute the 2^n × 2^n unitary matrix. Returns a list-of-rows."""

    def run_gradient(self, commands: Sequence[Command], n_qubits: int, observable: Sequence[tuple[Sequence[tuple[int, str]], complex]], params: Mapping[str, float], param_order: Sequence[str], *, initial_state: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> list[float]:
        """
        Adjoint-backprop gradient of <H> w.r.t. each parameter.  Cost: ~2 statevector simulations regardless of parameter count.  Exact for gates whose angle is a linear function of the supplied parameters (Rx/Ry/Rz/P/GPhase/CRx/CRy/CRz/CP/Rxx/Ryy/Rzz).  initial_state seeds the forward-sweep root (same contract as run()).
        """

    def run_gradient_phi(self, commands: Sequence[Command], n_qubits: int, phi: Sequence[complex], params: Mapping[str, float], param_order: Sequence[str], *, initial_state: Annotated[NDArray[numpy.complex128], dict(shape=(None,), order='C', device='cpu', writable=False)] | None = None) -> list[float]:
        """
        Adjoint-backprop gradient with caller-supplied |φ⟩.  Returns ∂(2·Re ⟨φ|U(θ)|ψ₀⟩)/∂θ_k with ψ₀ = initial_state or |0…0⟩.  Set φ = ⟨bra|ψ_n⟩·|bra⟩ to obtain ∂|⟨bra|ψ⟩|²/∂θ — the natural form for VQD-style overlap penalties.
        """

class PennylaneEmitter:
    def __init__(self) -> None: ...

    def target_name(self) -> str: ...

    def gate_set(self) -> GateSet: ...

    def capabilities(self) -> EmitterCapabilities: ...

    def validate(self, commands: Sequence[Command]) -> Incompatibility | None:
        """
        First Incompatibility in the sequence, or None if the whole sequence can cross.  Needs no SDK installed.
        """

    def emit(self, commands: Sequence[Command], n_qubits: int, circuit_name: str = 'circuit') -> Any:
        """Translate a flat QARPx command sequence into a qml.tape.QuantumScript."""

class PennylaneAbsorber:
    def __init__(self) -> None: ...

    def source_name(self) -> str: ...

    def absorb(self, tape: Any) -> tuple[list[Command], int, int]:
        """Parse a qml.tape.QuantumScript into (commands, n_qubits, n_cbits)."""

class PytketEmitter:
    def __init__(self) -> None: ...

    def target_name(self) -> str: ...

    def gate_set(self) -> GateSet: ...

    def capabilities(self) -> EmitterCapabilities: ...

    def validate(self, commands: Sequence[Command]) -> Incompatibility | None:
        """
        First Incompatibility in the sequence, or None if the whole sequence can cross.  Needs no SDK installed.
        """

    def emit(self, commands: Sequence[Command], n_qubits: int, circuit_name: str = 'circuit') -> Any:
        """Translate a flat QARPx command sequence into a pytket.Circuit."""

class PytketAbsorber:
    def __init__(self) -> None: ...

    def source_name(self) -> str: ...

    def absorb(self, circuit: Any) -> tuple[list[Command], int, int]:
        """Parse a pytket.Circuit into (commands, n_qubits, n_cbits)."""

class QulacsEmitter:
    def __init__(self) -> None: ...

    def target_name(self) -> str: ...

    def gate_set(self) -> GateSet: ...

    def capabilities(self) -> EmitterCapabilities: ...

    def validate(self, commands: Sequence[Command]) -> Incompatibility | None:
        """
        First Incompatibility in the sequence, or None if the whole sequence can cross.  Needs no SDK installed.
        """

    def emit(self, commands: Sequence[Command], n_qubits: int, circuit_name: str = 'circuit') -> Any:
        """Convert a QARPx command sequence to a qulacs.QuantumCircuit."""

class QulacsAbsorber:
    def __init__(self) -> None: ...

    def source_name(self) -> str: ...

    def absorb(self, circuit: Any) -> tuple[list[Command], int, int]:
        """Convert a qulacs.QuantumCircuit to (commands, n_qubits, n_cbits)."""

class QiskitEmitter:
    def __init__(self) -> None: ...

    def target_name(self) -> str: ...

    def gate_set(self) -> GateSet: ...

    def capabilities(self) -> EmitterCapabilities: ...

    def validate(self, commands: Sequence[Command]) -> Incompatibility | None:
        """
        First Incompatibility in the sequence, or None if the whole sequence can cross.  Needs no SDK installed.
        """

    def emit(self, commands: Sequence[Command], n_qubits: int, circuit_name: str = 'circuit') -> Any:
        """Translate a flat QARPx command sequence into a qiskit.QuantumCircuit."""

class QiskitAbsorber:
    def __init__(self) -> None: ...

    def source_name(self) -> str: ...

    def absorb(self, circuit: Any) -> tuple[list[Command], int, int]:
        """Parse a qiskit.QuantumCircuit into (commands, n_qubits, n_cbits)."""

class CudaqBackend(enum.Enum):
    StateVector = 0

    StateVectorMGPU = 1

    TensorNet = 2

    TensorNetMPS = 3

class CudaqPrecision(enum.Enum):
    FP32 = 0

    FP64 = 1

class CudaqConfig:
    def __init__(self) -> None: ...

    @property
    def backend(self) -> CudaqBackend: ...

    @backend.setter
    def backend(self, arg: CudaqBackend, /) -> None: ...

    @property
    def precision(self) -> CudaqPrecision: ...

    @precision.setter
    def precision(self, arg: CudaqPrecision, /) -> None: ...

    @property
    def target(self) -> str: ...

    @target.setter
    def target(self, arg: str, /) -> None: ...

    @property
    def max_bond_dim(self) -> int: ...

    @max_bond_dim.setter
    def max_bond_dim(self, arg: int, /) -> None: ...

class CudaqSimulator:
    @overload
    def __init__(self) -> None: ...

    @overload
    def __init__(self, config: CudaqConfig) -> None: ...

    @staticmethod
    def available() -> bool: ...

    @property
    def config(self) -> CudaqConfig: ...

    @config.setter
    def config(self, arg: CudaqConfig, /) -> None: ...

    def run(self, commands: Sequence[Command], n_qubits: int, n_shots: int, seed: int | None = None) -> SamplingResult: ...

    def batch_run(self, commands: Sequence[Command], n_qubits: int, n_shots: int, param_sets: Sequence[Mapping[str, float]], seed: int | None = None) -> list[SamplingResult]: ...

    def statevector(self, commands: Sequence[Command], n_qubits: int) -> Annotated[NDArray[numpy.complex128], dict(shape=(None,))]:
        """Host statevector as a zero-copy 1-D complex128 numpy array."""

    def batch_expectation(self, commands: Sequence[Command], n_qubits: int, observable: Sequence[tuple[Sequence[tuple[int, str]], complex]], param_sets: Sequence[Mapping[str, float]]) -> list[float]:
        """
        Sampling-free <H> swept over parameter sets (CUDA-Q observe). One expectation value per parameter set.
        """
