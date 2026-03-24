from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mcsr.tree.expression import Expression
from mcsr.tree.grammar import Atom, Grammar
from mcsr.algos.interface import ResearchAlgoInterface
from mcsr.utils.predictor import PredictorInterface, DummyPredictor
from mcsr.utils.fitness import compute_fitness


@dataclass
class PUCTNode:
    atom: Optional[Atom] = None
    children: list[PUCTNode] = field(default_factory=list)
    sum_scores: float = 0.0
    visit_count: int = 0
    prior: float = 1.0
    fully_explored: bool = False

    @property
    def mean_score(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.sum_scores / self.visit_count


@dataclass
class _BestState:
    best_sequence: list[Atom] = field(default_factory=list)
    best_fitness: float = -float("inf")
    min_score: float = float("inf")
    max_score: float = -float("inf")


def _update_best(state: _BestState, sequence: list[Atom], fitness: float) -> None:
    if fitness > state.best_fitness:
        state.best_fitness = fitness
        state.best_sequence = list(sequence)
    if fitness < state.min_score:
        state.min_score = fitness
    if fitness > state.max_score:
        state.max_score = fitness


def _puct_score(
    child: PUCTNode,
    parent_visit_count: int,
    exploration_constant: float,
    min_score: float,
    max_score: float,
) -> float:
    # PUCT Formula: Q + c * P * sqrt(N) / (1 + n)
    exploitation = child.mean_score
    if max_score > min_score:
        exploitation = (exploitation - min_score) / (max_score - min_score)
    elif max_score == min_score and max_score != -float("inf"):
        exploitation = 0.5
    else:
        exploitation = 0.0

    exploration = exploration_constant * child.prior * math.sqrt(parent_visit_count) / (1 + child.visit_count)
    return exploitation + exploration


def puct_search(
    node: PUCTNode,
    partial_sequence: list[Atom],
    remaining_leaves: int,
    max_atoms: int,
    grammar: Grammar,
    input_data: np.ndarray,
    target: np.ndarray,
    exploration_constant: float,
    best_state: _BestState,
    predictor: PredictorInterface
) -> float:
    # --- Expansion Phase ---
    if not node.children and not node.fully_explored:
        valid_atoms = grammar.get_valid_atoms(
            remaining_leaves, max_atoms, len(partial_sequence)
        )
        
        if not valid_atoms:
            node.fully_explored = True
            predicted = Expression(partial_sequence).evaluate(input_data)
            fitness = compute_fitness(predicted, target)
            _update_best(best_state, partial_sequence, fitness)
            return fitness

        # Query predictor
        value, policy = predictor.predict(partial_sequence, valid_atoms, grammar)
        
        for atom in valid_atoms:
            child = PUCTNode(atom=atom, prior=policy.get(atom, 0.0))
            node.children.append(child)
            
        return value

    # --- Selection Phase ---
    best_puct = -float("inf")
    best_child: Optional[PUCTNode] = None

    for child in node.children:
        if child.fully_explored:
            continue
        puct = _puct_score(
            child, node.visit_count, exploration_constant, best_state.min_score, best_state.max_score
        )
        if puct > best_puct:
            best_puct = puct
            best_child = child

    if best_child is None:
        node.fully_explored = True
        return node.mean_score

    best_child_atom = best_child.atom
    assert best_child_atom is not None

    new_sequence = partial_sequence + [best_child_atom]
    new_remaining = remaining_leaves + best_child_atom.arity - 1

    if new_remaining == 0:
        predicted = Expression(new_sequence).evaluate(input_data)
        score = compute_fitness(predicted, target)
        _update_best(best_state, new_sequence, score)
        best_child.fully_explored = True
        best_child.sum_scores += score
        best_child.visit_count += 1
    else:
        score = puct_search(
            best_child,
            new_sequence,
            new_remaining,
            max_atoms,
            grammar,
            input_data,
            target,
            exploration_constant,
            best_state,
            predictor
        )

    # Backpropagate
    node.sum_scores += score
    node.visit_count += 1

    if all(c.fully_explored for c in node.children):
        node.fully_explored = True

    return score


class PUCT(ResearchAlgoInterface):
    """Predictor + Upper Confidence Bound applied to Trees (PUCT) for SR."""

    def __init__(
        self,
        grammar: Grammar,
        max_atoms: int = 15,
        num_iterations: int = 2000,
        exploration_constant: float = 1.0,
        seed: Optional[int] = None,
        predictor: Optional[PredictorInterface] = None,
        checkpoint_path: Optional[str] = None,
    ):
        super().__init__(grammar=grammar, max_atoms=max_atoms, seed=seed)
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant
        
        if predictor is not None:
            self.predictor = predictor
        elif checkpoint_path is not None:
            import torch
            import os
            from mcsr.utils.predictor import PredictorNN
            net = PredictorNN(grammar=grammar)
            if os.path.exists(checkpoint_path):
                net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
                # Set to eval mode is done in predict()
            else:
                print(f"Warning: Checkpoint {checkpoint_path} not found. Using untrained weights.")
            self.predictor = net
        else:
            self.predictor = DummyPredictor()

    def fit(
        self, input_data: np.ndarray, target: np.ndarray
    ) -> tuple[Expression, float]:
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        root = PUCTNode()
        best = _BestState()

        for iteration in range(self.num_iterations):
            if root.fully_explored:
                break

            puct_search(
                node=root,
                partial_sequence=[],
                remaining_leaves=1,
                max_atoms=self.max_atoms,
                grammar=self.grammar,
                input_data=input_data,
                target=target,
                exploration_constant=self.exploration_constant,
                best_state=best,
                predictor=self.predictor
            )

        # Ensure return is a valid AST if possible, or fallback to an empty Expression
        return Expression(best.best_sequence), best.best_fitness
