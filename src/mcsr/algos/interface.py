from abc import ABC, abstractmethod

import numpy as np

from mcsr.tree.expression import Expression
from mcsr.tree.grammar import Grammar


class SRAlgorithm(ABC):
    """Interface for symbolic regression algorithms."""

    def __init__(self, grammar: Grammar, max_atoms: int = 10) -> None:
        self.grammar = grammar
        self.max_atoms = max_atoms

    def fit(self, input_data: np.ndarray, target: np.ndarray) -> Expression:
        """Run the symbolic regression algorithm and return the best expression found."""
        self.grammar.set_variables(input_data.shape[1])
        return self._fit(input_data, target)

    @abstractmethod
    def _fit(self, input_data: np.ndarray, target: np.ndarray) -> Expression:
        pass
