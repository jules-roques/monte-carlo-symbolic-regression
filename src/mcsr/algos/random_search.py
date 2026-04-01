import random

import numpy as np

from mcsr.algos.interface import SRAlgorithm
from mcsr.algos.uct import random_playout
from mcsr.tree.expression import Expression
from mcsr.tree.grammar import Grammar


class RandomSearch(SRAlgorithm):
    """Random Search for Symbolic Regression."""

    def __init__(
        self,
        grammar: Grammar,
        max_atoms: int = 8,
        num_iterations: int = 50_000,
    ):
        super().__init__(grammar=grammar, max_atoms=max_atoms)
        self.num_iterations = num_iterations

    def _fit(self, input_data: np.ndarray, target: np.ndarray) -> Expression:
        best_sequence = []
        best_fitness = -float("inf")

        for _ in range(self.num_iterations):
            k = random.randint(1, self.max_atoms)

            sequence, fitness = random_playout(
                partial_sequence=[],
                remaining_leaves=1,
                max_atoms=k,
                grammar=self.grammar,
                input_data=input_data,
                target=target,
            )

            if fitness > best_fitness:
                best_fitness = fitness
                best_sequence = list(sequence)

        return Expression(best_sequence)
