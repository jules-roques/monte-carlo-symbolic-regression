from __future__ import annotations

from typing import Iterator

import numpy as np
import zss

from mcsr.tree.atom import Atom


class Expression:
    """
    Represents a mathematical expression as a sequence of atoms in prefix notation.
    """

    def is_valid(self) -> bool:

        if not self.atom_sequence:
            return False

        slots_needed = 1

        for atom in self.atom_sequence:
            if slots_needed <= 0:
                return False
            slots_needed += atom.arity - 1

        return slots_needed == 0

    def __init__(self, atom_sequence: list[Atom]):
        self.atom_sequence = atom_sequence
        if not self.is_valid():
            raise ValueError("Invalid atom sequence")

    def _to_string_recursive(self, iterator: Iterator[Atom]) -> str:
        atom = next(iterator)
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
        return self._to_string_recursive(iterator)

    def _to_zss_tree(self, iterator: Iterator[Atom]) -> zss.Node:
        """Convert prefix atom sequence to a zss-compatible tree node."""
        atom = next(iterator)
        node = zss.Node(atom.name)
        for _ in range(atom.arity):
            node.addkid(self._to_zss_tree(iterator))
        return node

    def compute(self, input_data: np.ndarray) -> np.ndarray:
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
