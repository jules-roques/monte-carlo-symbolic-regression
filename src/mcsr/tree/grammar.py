from __future__ import annotations

import numpy as np

from mcsr.tree.atom import Atom, Constant, Variable, UnaryOperator, BinaryOperator

# --- Binary operators (arity=2) ---
ADD = BinaryOperator(name="+", func=lambda x, y: x + y)
SUB = BinaryOperator(name="-", func=lambda x, y: x - y)
MUL = BinaryOperator(name="*", func=lambda x, y: x * y)
DIV = BinaryOperator(
    name="/", func=lambda x, y: np.where(np.abs(y) > 1e-10, x / y, np.nan)
)

BINARY_OPERATORS: list[Atom] = [ADD, SUB, MUL, DIV]

# --- Unary operators (arity=1) ---
SIN = UnaryOperator(name="sin", func=lambda x: np.sin(x))
COS = UnaryOperator(name="cos", func=lambda x: np.cos(x))
EXP = UnaryOperator(name="exp", func=lambda x: np.exp(np.clip(x, -500, 500)))
LOG = UnaryOperator(name="log", func=lambda x: np.where(x > 1e-10, np.log(x), np.nan))
SQRT = UnaryOperator(name="sqrt", func=lambda x: np.where(x >= 0, np.sqrt(x), np.nan))

UNARY_OPERATORS: list[Atom] = [SIN, COS, EXP, LOG, SQRT]

# --- Numeric constants (arity=0, terminals) ---
CONSTANT_VALUES: list[float] = [0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
CONSTANTS: list[Atom] = [Constant(name="const", value=v) for v in CONSTANT_VALUES]


def make_variable(index: int, name: str | None = None) -> Atom:
    if name is None:
        name = f"x{index}"
    return Variable(name=name, var_index=index)


class Grammar:
    def __init__(self, num_variables: int, variable_names: list[str] | None = None) -> None:
        self.num_variables: int = num_variables
        if variable_names is None:
            self.variables: list[Atom] = [make_variable(i) for i in range(num_variables)]
        else:
            self.variables: list[Atom] = [
                make_variable(i, name) for i, name in enumerate(variable_names)
            ]
        self.all_atoms: list[Atom] = (
            BINARY_OPERATORS + UNARY_OPERATORS + CONSTANTS + self.variables
        )
        self.terminal_atoms: list[Atom] = [a for a in self.all_atoms if a.arity == 0]
        self.nonterminal_atoms: list[Atom] = [a for a in self.all_atoms if a.arity > 0]

    def get_valid_atoms(
        self, remaining_leaves: int, max_atoms: int, current_index: int
    ) -> list[Atom]:
        """Return atoms that can legally be placed at the current position.

        An atom is valid if adding it would not make the expression exceed
        max_atoms when all remaining leaves are eventually filled with terminals.
        Specifically: current_index + remaining_leaves + (arity - 1) <= max_atoms,
        which simplifies to: arity <= max_atoms - current_index - remaining_leaves + 1.
        If remaining_leaves == 1 and current_index + 1 < max_atoms, all atoms are
        candidates; if remaining_leaves == 1 and current_index + 1 == max_atoms, only
        terminals are valid.
        """
        budget = max_atoms - current_index - remaining_leaves + 1
        return [a for a in self.all_atoms if a.arity <= budget]
