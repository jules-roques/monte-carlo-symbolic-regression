"""Tests for UCT search."""
import numpy as np
import pytest

from mcsr.tree.grammar import Grammar, make_variable, MUL
from mcsr.tree.expression import Expression
from mcsr.algos.uct import UCT, random_playout
from mcsr.utils.fitness import compute_r_squared


class TestRandomPlayout:
    def test_produces_valid_expression(self):
        grammar = Grammar(num_variables=2)
        np.random.seed(42)
        import random
        random.seed(42)

        data = np.random.rand(50, 2)
        target = data[:, 0] + data[:, 1]

        for _ in range(20):
            sequence, fitness = random_playout(
                partial_sequence=[],
                remaining_leaves=1,
                max_atoms=10,
                grammar=grammar,
                input_data=data,
                target=target,
            )
            assert len(sequence) > 0
            remaining = 1
            for atom in sequence:
                remaining += atom.arity - 1
            assert remaining == 0, "Playout did not produce a complete expression"


class TestUCTSearch:
    def test_finds_product(self):
        """UCT should find a decent fit for y = x0 * x1 on simple data."""
        np.random.seed(42)
        data = np.random.uniform(0.1, 5.0, size=(200, 2))
        target = data[:, 0] * data[:, 1]

        grammar = Grammar(num_variables=2)
        uct = UCT(
            grammar=grammar,
            max_atoms=7,
            num_iterations=10000,
            exploration_constant=0.5,
            seed=42,
        )
        best_expression, best_fitness = uct.fit(
            input_data=data,
            target=target,
        )

        predicted = best_expression.evaluate(data)
        r2 = compute_r_squared(predicted, target)
        assert r2 > 0.5, f"Expected R² > 0.5 for y=x0*x1, got {r2:.4f}"

    def test_finds_sum(self):
        """UCT should find a decent fit for y = x0 + x1."""
        np.random.seed(123)
        data = np.random.uniform(0.1, 5.0, size=(200, 2))
        target = data[:, 0] + data[:, 1]

        grammar = Grammar(num_variables=2)
        uct = UCT(
            grammar=grammar,
            max_atoms=5,
            num_iterations=5000,
            exploration_constant=0.5,
            seed=123,
        )
        best_expression, best_fitness = uct.fit(
            input_data=data,
            target=target,
        )

        predicted = best_expression.evaluate(data)
        r2 = compute_r_squared(predicted, target)
        assert r2 > 0.5, f"Expected R² > 0.5 for y=x0+x1, got {r2:.4f}"
