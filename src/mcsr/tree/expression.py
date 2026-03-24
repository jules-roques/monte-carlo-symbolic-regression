from __future__ import annotations
from typing import Iterator
import numpy as np
import zss  # Requires: pip install zss

from mcsr.tree.atom import Atom


class Expression:
    def __init__(self, atom_sequence: list[Atom]):
        self.atom_sequence = atom_sequence

    def _to_string_recursive(self, iterator: Iterator[Atom]) -> str:
        atom: Atom = next(iterator)
        if atom.arity == 0:
            return repr(atom)
        elif atom.arity == 1:
            child = self._to_string_recursive(iterator)
            return f"{atom.name}({child})"
        elif atom.arity == 2:
            left = self._to_string_recursive(iterator)
            right = self._to_string_recursive(iterator)
            return f"({left} {atom.name} {right})"
        raise ValueError(f"Unsupported arity: {atom.arity}")

    def __str__(self) -> str:
        """Convert a prefix-notation atom sequence to a human-readable infix string."""
        iterator = iter(self.atom_sequence)
        try:
            return self._to_string_recursive(iterator)
        except StopIteration:
            return "<malformed>"

    def _to_zss_tree(self, iterator: Iterator[Atom]) -> zss.Node:
        """Convert prefix atom sequence to a zss-compatible tree node."""
        atom = next(iterator)
        node = zss.Node(atom.name)
        for _ in range(atom.arity):
            node.addkid(self._to_zss_tree(iterator))
        return node

    def evaluate(self, input_data: np.ndarray) -> np.ndarray:
        stack: list[np.ndarray] = []
        for atom in reversed(self.atom_sequence):
            if atom.arity > 0:
                assert len(stack) >= atom.arity
                args = [stack.pop() for _ in range(atom.arity)]
                value = atom.operator(*args)
            else:
                value = atom.operator(input_data)
            stack.append(value)

        assert len(stack) == 1
        result = stack[0]
        return np.where(np.isfinite(result), result, np.nan)

    def distance_to(self, other: Expression) -> float:
        """
        Compute normalized edit distance (NED) between two expressions.
        Returns a float in [0, 1].
        """
        tree_self = self._to_zss_tree(iter(self.atom_sequence))
        tree_other = other._to_zss_tree(iter(other.atom_sequence))

        edit_dist = zss.simple_distance(tree_self, tree_other)
        if isinstance(edit_dist, tuple):
            edit_dist = edit_dist[0]

        max_size = max(len(self.atom_sequence), len(other.atom_sequence))

        return edit_dist / max_size if max_size > 0 else 0.0


import sympy
from mcsr.tree.atom import Constant, Variable
from mcsr.tree.grammar import ADD, SUB, MUL, DIV, SIN, COS, EXP, LOG, SQRT

def sympy_to_expression(
    sy_expr: sympy.Expr, variable_names: list[str] | None = None
) -> Expression:
    """Convert a SymPy expression into an Expression object."""
    atoms = []

    def _traverse(node):
        if hasattr(node, "is_Symbol") and node.is_Symbol:
            # Check if name starts with x followed by an integer
            name_str = str(node.name)
            if name_str.startswith("x") and name_str[1:].isdigit():
                index = int(name_str[1:])
                real_name = (
                    variable_names[index]
                    if variable_names and index < len(variable_names)
                    else name_str
                )
                atoms.append(Variable(name=real_name, var_index=index))
            else:
                # Fallback for symbols not matching x<index> pattern
                atoms.append(Variable(name=name_str, var_index=0))
        elif hasattr(node, "is_Number") and node.is_Number:
            atoms.append(Constant(name="const", value=float(node)))
        elif isinstance(node, sympy.Add):
            args = node.args
            for _ in range(len(args) - 1):
                atoms.append(ADD)
            for arg in args:
                _traverse(arg)
        elif isinstance(node, sympy.Mul):
            args = node.args
            for _ in range(len(args) - 1):
                atoms.append(MUL)
            for arg in args:
                _traverse(arg)
        elif isinstance(node, sympy.Pow):
            base, exp = node.args
            if exp == -1:
                atoms.append(DIV)
                atoms.append(Constant(name="const", value=1.0))
                _traverse(base)
            elif exp == 0.5:
                atoms.append(SQRT)
                _traverse(base)
            else:
                # Unsupported Pow. Using dummy for syntactic completeness.
                atoms.append(MUL)
                atoms.append(Constant(name="const", value=1.0))
                _traverse(base)
        elif isinstance(node, sympy.sin):
            atoms.append(SIN)
            _traverse(node.args[0])
        elif isinstance(node, sympy.cos):
            atoms.append(COS)
            _traverse(node.args[0])
        elif isinstance(node, sympy.exp):
            atoms.append(EXP)
            _traverse(node.args[0])
        elif isinstance(node, sympy.log):
            atoms.append(LOG)
            _traverse(node.args[0])
        else:
            atoms.append(Constant(name="const", value=1.0))

    _traverse(sy_expr)
    return Expression(atoms)
