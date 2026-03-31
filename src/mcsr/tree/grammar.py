from __future__ import annotations

import numpy as np

from mcsr.tree.atom import Atom, BinaryOperator, Constant, UnaryOperator, Variable


def safe_divide(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return np.where(np.abs(y) > 1e-10, x / y, np.nan)


def safe_log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(x > 1e-10, np.log(x), np.nan)


def safe_sqrt(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.where(x >= 0, np.sqrt(x), np.nan)


def safe_exp(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(np.clip(x, -500.0, 500.0))


def safe_square(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return x * x


def safe_sin(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.sin(x)


def safe_cos(x: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        return np.cos(x)


def safe_add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return x + y


def safe_subtract(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return x - y


def safe_multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return x * y


ADD = BinaryOperator(name="+", func=safe_add)
SUB = BinaryOperator(name="-", func=safe_subtract)
MUL = BinaryOperator(name="*", func=safe_multiply)
DIV = BinaryOperator(name="/", func=safe_divide)

BINARY_OPERATORS: list[Atom] = [ADD, SUB, MUL, DIV]

SIN = UnaryOperator(name="sin", func=safe_sin)
COS = UnaryOperator(name="cos", func=safe_cos)
EXP = UnaryOperator(name="exp", func=safe_exp)
LOG = UnaryOperator(name="log", func=safe_log)
SQRT = UnaryOperator(name="sqrt", func=safe_sqrt)
SQUARE = UnaryOperator(name="square", func=safe_square)

UNARY_OPERATORS: list[Atom] = [SIN, COS, EXP, LOG, SQRT, SQUARE]

CONSTANT_VALUES: list[float] = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
CONSTANTS: list[Atom] = [Constant(name="const", value=v) for v in CONSTANT_VALUES]


def make_variable(index: int) -> Atom:
    return Variable(name=f"x{index}", var_index=index)


class Grammar:
    def __init__(self) -> None:
        self._initialize_base_atoms()
        self._reset_to_base()

    def _initialize_base_atoms(self) -> None:
        self._base_atoms = BINARY_OPERATORS + UNARY_OPERATORS + CONSTANTS
        self._base_terminals = [atom for atom in self._base_atoms if atom.arity == 0]
        self._base_nonterminals = [atom for atom in self._base_atoms if atom.arity > 0]

    def _reset_to_base(self) -> None:
        self.all_atoms = list(self._base_atoms)
        self.terminal_atoms = list(self._base_terminals)
        self.nonterminal_atoms = list(self._base_nonterminals)

    def set_variables(self, num_variables: int) -> None:
        self._reset_to_base()
        variables = [make_variable(i) for i in range(num_variables)]
        self.all_atoms.extend(variables)
        self.terminal_atoms.extend(variables)

    def get_valid_atoms(
        self, remaining_leaves: int, max_atoms: int, current_index: int
    ) -> list[Atom]:
        budget = max_atoms - current_index - remaining_leaves + 1
        return [atom for atom in self.all_atoms if atom.arity <= budget]
