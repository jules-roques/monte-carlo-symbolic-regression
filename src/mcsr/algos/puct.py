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
    next_child_to_try: int = 0  # tracks expansion progress (like UCT)
    # Ordered list of valid atoms for this node, set during first expansion.
    # With NN priors, highest-prior atoms come first so we try the best moves early.
    _expansion_order: list[Atom] = field(default_factory=list)

    @property
    def mean_score(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.sum_scores / self.visit_count


@dataclass
class _BestState:
    best_sequence: list[Atom] = field(default_factory=list)
    best_fitness: float = -float("inf")


def _update_best(state: _BestState, sequence: list[Atom], fitness: float) -> None:
    if fitness > state.best_fitness:
        state.best_fitness = fitness
        state.best_sequence = list(sequence)


def _ucb_score(
    child: PUCTNode,
    parent_visit_count: int,
    exploration_constant: float,
) -> float:
    """Standard UCB1 formula — identical to UCT for proven exploration guarantees."""
    if child.visit_count == 0:
        return float("inf")
    exploitation = child.mean_score
    exploration = exploration_constant * math.sqrt(
        math.log(parent_visit_count) / child.visit_count
    )
    return exploitation + exploration


def random_playout(
    partial_sequence: list[Atom],
    remaining_leaves: int,
    max_atoms: int,
    grammar: Grammar,
    input_data: np.ndarray,
    target: np.ndarray,
) -> tuple[list[Atom], float]:
    """Complete a partial expression randomly and return the full sequence and its fitness."""
    sequence = list(partial_sequence)
    leaves = remaining_leaves
    current_index = len(sequence)

    while leaves > 0:
        valid_atoms = grammar.get_valid_atoms(leaves, max_atoms, current_index)
        if not valid_atoms:
            valid_atoms = grammar.terminal_atoms
        chosen = random.choice(valid_atoms)
        sequence.append(chosen)
        current_index += 1
        leaves += chosen.arity - 1

    try:
        predicted = Expression(sequence).evaluate(input_data)
        fitness = compute_fitness(predicted, target)
    except Exception:
        fitness = 0.0
    return sequence, fitness


def _update_explored_status(node: PUCTNode) -> None:
    all_tried = node.next_child_to_try >= len(node._expansion_order)
    all_children_explored = all(c.fully_explored for c in node.children)
    if all_tried and all_children_explored:
        node.fully_explored = True


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
    valid_atoms = grammar.get_valid_atoms(
        remaining_leaves, max_atoms, len(partial_sequence)
    )

    # --- Compute expansion order once (using NN priors to sort) ---
    if not node._expansion_order and valid_atoms:
        _value, policy = predictor.predict(partial_sequence, valid_atoms, grammar)
        # Sort valid atoms by NN prior (descending) so best moves are tried first
        node._expansion_order = sorted(valid_atoms, key=lambda a: policy.get(a, 0.0), reverse=True)

    # --- Expansion Phase: try untried atoms one at a time (like UCT) ---
    while node.next_child_to_try < len(node._expansion_order):
        atom = node._expansion_order[node.next_child_to_try]
        node.next_child_to_try += 1

        new_sequence = partial_sequence + [atom]
        new_remaining = remaining_leaves + atom.arity - 1

        completed_sequence, score = random_playout(
            new_sequence, new_remaining, max_atoms, grammar, input_data, target
        )

        child = PUCTNode(atom=atom)
        child.sum_scores = score
        child.visit_count = 1
        if new_remaining == 0:
            child.fully_explored = True
        node.children.append(child)

        node.sum_scores += score
        node.visit_count += 1

        _update_explored_status(node)
        _update_best(best_state, completed_sequence, score)
        return score

    # --- Selection Phase: pick best child via UCB1 (proven formula) ---
    best_ucb = -float("inf")
    best_child: Optional[PUCTNode] = None

    for child in node.children:
        if child.fully_explored:
            continue
        ucb = _ucb_score(child, node.visit_count, exploration_constant)
        if ucb > best_ucb:
            best_ucb = ucb
            best_child = child

    if best_child is None:
        node.fully_explored = True
        return node.mean_score

    best_child_atom = best_child.atom
    assert best_child_atom is not None

    new_sequence = partial_sequence + [best_child_atom]
    new_remaining = remaining_leaves + best_child_atom.arity - 1

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

    if all(c.fully_explored for c in node.children) and node.next_child_to_try >= len(node._expansion_order):
        node.fully_explored = True

    return score


class PUCT(ResearchAlgoInterface):
    """Predictor + Upper Confidence Bound applied to Trees (PUCT) for SR.

    Hybrid approach: uses UCT's proven UCB1 formula for selection, but
    leverages NN priors to order expansion (try most promising atoms first).
    This guarantees at least UCT-level performance while benefiting from
    neural network guidance during the expansion phase.
    """

    def __init__(
        self,
        grammar: Grammar,
        max_atoms: int = 15,
        num_iterations: int = 2000,
        exploration_constant: float = 1.0,
        seed: Optional[int] = None,
        predictor: Optional[PredictorInterface] = None,
        checkpoint_path: Optional[str] = None,
        model_num_variables: Optional[int] = None,
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
            if model_num_variables is not None and model_num_variables != grammar.num_variables:
                model_grammar = Grammar(num_variables=model_num_variables)
            else:
                model_grammar = grammar
            net = PredictorNN(grammar=model_grammar)
            if os.path.exists(checkpoint_path):
                net.load_state_dict(torch.load(checkpoint_path, weights_only=True))
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

        return Expression(best.best_sequence), best.best_fitness
