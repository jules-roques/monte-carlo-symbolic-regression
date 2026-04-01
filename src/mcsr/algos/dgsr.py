from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mcsr.algos.interface import SRAlgorithm
from mcsr.tree.expression import Expression
from mcsr.tree.grammar import Grammar
from mcsr.utils.metrics import compute_fitness
from mcsr.utils.mutator import DummyMutator, MutatorInterface


@dataclass
class DGSRNode:
    """Nœud de l'arbre MCTS représentant une expression mathématique complète."""

    expression: Expression
    parent: Optional[DGSRNode] = None
    children: list[DGSRNode] = field(default_factory=list)
    sum_scores: float = 0.0
    visit_count: int = 0
    prior: float = 1.0  # e.g., math.exp(log_prob) issu de p_theta
    is_expanded: bool = False

    @property
    def mean_score(self) -> float:
        return 0.0 if self.visit_count == 0 else self.sum_scores / self.visit_count


def _ucb_score(
    child: DGSRNode, parent_visit_count: int, exploration_constant: float
) -> float:
    if child.visit_count == 0:
        return float("inf")
    # Equation du MCTS : Q(s,a) + c * sqrt(ln(N(s)) / N(s,a))
    return child.mean_score + exploration_constant * math.sqrt(
        math.log(parent_visit_count) / child.visit_count
    )


class DGSR(SRAlgorithm):
    """Deep Generative Symbolic Regression with MCTS (DGSR-MCTS)."""

    def __init__(
        self,
        grammar: Grammar,
        max_atoms: int = 15,
        num_iterations: int = 2000,
        exploration_constant: float = 1.0,
        lambda_complexity: float = 0.01,
        num_mutations_per_expansion: int = 5,
        mutator: Optional[MutatorInterface] = None,
        mutator_path: Optional[str] = None,
    ):
        super().__init__(grammar=grammar, max_atoms=max_atoms)
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant
        self.lambda_complexity = lambda_complexity
        self.num_mutations_per_expansion = num_mutations_per_expansion

        if mutator is not None:
            self.mutator = mutator
        elif mutator_path is not None:
            import torch

            from mcsr.utils.mutator import MutatorNN

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Toujours utiliser 5 variables pour le modèle lui-même (vocabulaire stable)
            from mcsr.tree.grammar import Grammar

            model_grammar = Grammar()
            model_grammar.set_variables(5)
            self.mutator = MutatorNN(model_grammar, max_atoms=max_atoms).to(device)
            print(f"Loading MutatorNN from {mutator_path} ({device})...")
            checkpoint = torch.load(
                mutator_path, map_location=device, weights_only=True
            )
            if "model_state_dict" in checkpoint:
                self.mutator.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.mutator.load_state_dict(checkpoint)
            self.mutator.eval()
        else:
            self.mutator = DummyMutator(grammar, max_atoms)

    def _evaluate(
        self, expr: Expression, input_data: np.ndarray, target: np.ndarray
    ) -> float:
        """Evalue une expression avec MSE/NRMSE moins la pénalité de complexité."""
        try:
            predicted = expr.compute(input_data)
            fitness = compute_fitness(predicted, target)
        except Exception:
            fitness = -1e6

        penalty = self.lambda_complexity * len(expr.atom_sequence)
        score = fitness - penalty
        # Clip pour éviter les valeurs extrêmes qui cassent le gradient
        return max(score, -100.0)

    def _fit(self, input_data: np.ndarray, target: np.ndarray) -> Expression:

        # Initial root expression (ex: f(x) = x0)
        from mcsr.tree.grammar import make_variable

        root_expr = Expression([make_variable(0)])
        root = DGSRNode(expression=root_expr)

        # Evaluate root
        score = self._evaluate(root.expression, input_data, target)
        root.visit_count = 1
        root.sum_scores = score

        best_expr = root.expression
        best_score = score

        for _ in range(self.num_iterations):
            node = root

            # 1. Sélection
            while node.is_expanded and node.children:
                best_ucb = -float("inf")
                best_child = None
                for child in node.children:
                    ucb = _ucb_score(child, node.visit_count, self.exploration_constant)
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child

                if best_child is None:
                    break
                node = best_child

            # 2. Expansion via Mutation Neuronale (p_theta)
            if not node.is_expanded:
                # On passe la grammaire du problème pour l'échantillonnage valide
                mutations = self.mutator.mutate(
                    node.expression,
                    self.num_mutations_per_expansion,
                    grammar=self.grammar,
                )

                # Create children corresponding to mutations
                for expr, log_prob in mutations:
                    child = DGSRNode(
                        expression=expr, parent=node, prior=math.exp(log_prob)
                    )
                    node.children.append(child)
                node.is_expanded = True

                # Choisir le premier enfant non visité pour la simulation
                if node.children:
                    node = node.children[0]

            # 3. Simulation (Evaluation directe, puisque les expressions sont complètes)
            if node.visit_count == 0:
                score = self._evaluate(node.expression, input_data, target)
                if score > best_score:
                    best_score = score
                    best_expr = node.expression
            else:
                score = node.mean_score

            # 4. Backpropagation
            curr = node
            while curr is not None:
                curr.visit_count += 1
                curr.sum_scores += score
                curr = curr.parent

        # 5. Collecte des trajectoires pour l'entraînement (e, e', reward)
        trajectories = []
        stack = [root]
        while stack:
            curr = stack.pop()
            if curr.is_expanded:
                for child in curr.children:
                    if child.visit_count > 0:
                        reward = child.mean_score
                        trajectories.append((curr.expression, child.expression, reward))
                    stack.append(child)

        self.last_trajectories = trajectories
        return best_expr
