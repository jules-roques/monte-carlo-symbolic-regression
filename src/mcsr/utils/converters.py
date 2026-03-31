from functools import singledispatch
from typing import Iterator

import sympy

from mcsr.tree.atom import Atom, BinaryOperator, Constant, UnaryOperator, Variable
from mcsr.tree.expression import Expression

STR2SYMPY = {
    "+": sympy.Add,
    "-": lambda a, b: a - b,
    "*": sympy.Mul,
    "/": lambda a, b: a / b,
    "sin": sympy.sin,
    "cos": sympy.cos,
    "exp": sympy.exp,
    "log": sympy.log,
    "sqrt": sympy.sqrt,
    "square": lambda x: x**2,
}


def expression_to_sympy(expr: Expression) -> sympy.Expr:
    """Entry point to convert an Expression to a SymPy expression."""
    iterator = iter(expr.atom_sequence)
    return _to_sympy_recursive(iterator)


def _to_sympy_recursive(iterator: Iterator[Atom]) -> sympy.Expr:
    """
    Recursively consumes the iterator to build a SymPy tree.
    """
    try:
        atom = next(iterator)
    except StopIteration:
        raise ValueError("Malformed expression sequence: unexpected end of atoms.")

    children = [_to_sympy_recursive(iterator) for _ in range(atom.arity)]

    return _to_sympy(atom, children)


@singledispatch
def _to_sympy(atom: Atom, children: list[sympy.Expr]) -> sympy.Expr:
    """Default fallback for unknown types."""
    raise TypeError(f"Cannot convert {type(atom)} to SymPy")


@_to_sympy.register(Variable)
def _(atom: Variable, children: list[sympy.Expr]) -> sympy.Expr:
    return sympy.Symbol(atom.name)


@_to_sympy.register(Constant)
def _(atom: Constant, children: list[sympy.Expr]) -> sympy.Expr:
    return sympy.Float(atom.value)


@_to_sympy.register(BinaryOperator | UnaryOperator)
def _(atom: BinaryOperator | UnaryOperator, children: list[sympy.Expr]) -> sympy.Expr:
    return STR2SYMPY[atom.name](*children)
