from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from mcsr.tree.grammar import Grammar
from mcsr.tree.expression import Expression


class ResearchAlgoInterface(ABC):
    def __init__(
        self, grammar: Grammar, max_atoms: int = 15, seed: Optional[int] = None
    ):
        self.grammar = grammar
        self.max_atoms = max_atoms
        self.seed = seed

    @abstractmethod
    def fit(
        self, input_data: np.ndarray, target: np.ndarray
    ) -> tuple[Expression, float]:
        """Run the symbolic regression algorithm and return the best expression and its fitness."""
        pass
