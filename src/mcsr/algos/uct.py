from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mcsr.algos.interface import SRAlgorithm
from mcsr.tree.expression import Expression
from mcsr.tree.grammar import Atom, Grammar
from mcsr.utils.metrics import compute_fitness


@dataclass
class UCTNode:
    atom: Optional[Atom] = None
    children: list[UCTNode] = field(default_factory=list)
    sum_scores: float = 0.0
    visit_count: int = 0
    fully_explored: bool = False
    next_atom_to_try: int = 0

    @property
    def mean_score(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.sum_scores / self.visit_count


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

    predicted = Expression(sequence).compute(input_data)
    fitness = compute_fitness(predicted, target)
    return sequence, fitness


def _ucb_score(
    child: UCTNode, parent_visit_count: int, exploration_constant: float
) -> float:
    if child.visit_count == 0:
        return float("inf")
    exploitation = child.mean_score
    exploration = exploration_constant * math.sqrt(
        math.log(parent_visit_count) / child.visit_count
    )
    return exploitation + exploration


def uct_search(
    node: UCTNode,
    partial_sequence: list[Atom],
    remaining_leaves: int,
    max_atoms: int,
    grammar: Grammar,
    input_data: np.ndarray,
    target: np.ndarray,
    exploration_constant: float,
    best_state: _BestState,
) -> float:
    """Recursive UCT descent following Algorithm 5 from the paper."""
    valid_atoms = grammar.get_valid_atoms(
        remaining_leaves, max_atoms, len(partial_sequence)
    )

    # --- Expansion phase: try atoms not yet expanded ---
    while node.next_atom_to_try < len(valid_atoms):
        atom = valid_atoms[node.next_atom_to_try]
        node.next_atom_to_try += 1

        new_sequence = partial_sequence + [atom]
        new_remaining = remaining_leaves + atom.arity - 1

        completed_sequence, score = random_playout(
            new_sequence, new_remaining, max_atoms, grammar, input_data, target
        )

        child = UCTNode(atom=atom)
        child.sum_scores = score
        child.visit_count = 1
        if new_remaining == 0:
            child.fully_explored = True
        node.children.append(child)

        node.sum_scores += score
        node.visit_count += 1

        _update_explored_status(node, valid_atoms)
        _update_best(best_state, completed_sequence, score)
        return score

    # --- Selection phase: pick best child via UCB ---
    best_ucb = -float("inf")
    best_child: Optional[UCTNode] = None

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

    score = uct_search(
        best_child,
        new_sequence,
        new_remaining,
        max_atoms,
        grammar,
        input_data,
        target,
        exploration_constant,
        best_state,
    )

    node.sum_scores += score
    node.visit_count += 1

    if all(c.fully_explored for c in node.children) and node.next_atom_to_try >= len(
        valid_atoms
    ):
        node.fully_explored = True

    return score


@dataclass
class _BestState:
    best_sequence: list[Atom] = field(default_factory=list)
    best_fitness: float = -float("inf")


def _update_best(state: _BestState, sequence: list[Atom], fitness: float) -> None:
    if fitness > state.best_fitness:
        state.best_fitness = fitness
        state.best_sequence = list(sequence)


def _update_explored_status(node: UCTNode, valid_atoms: list[Atom]) -> None:
    all_tried = node.next_atom_to_try >= len(valid_atoms)
    all_children_explored = all(c.fully_explored for c in node.children)
    if all_tried and all_children_explored:
        node.fully_explored = True


class UCT(SRAlgorithm):
    """Upper Confidence Bound applied to Trees (UCT) for Symbolic Regression."""

    def __init__(
        self,
        grammar: Grammar,
        max_atoms: int = 15,
        num_iterations: int = 50_000,
        exploration_constant: float = 0.5,
    ):
        super().__init__(grammar=grammar, max_atoms=max_atoms)
        self.num_iterations = num_iterations
        self.exploration_constant = exploration_constant

    def _fit(self, input_data: np.ndarray, target: np.ndarray) -> Expression:

        self.root = UCTNode()
        best = _BestState()

        for iteration in range(self.num_iterations):
            if self.root.fully_explored:
                break

            uct_search(
                node=self.root,
                partial_sequence=[],
                remaining_leaves=1,
                max_atoms=self.max_atoms,
                grammar=self.grammar,
                input_data=input_data,
                target=target,
                exploration_constant=self.exploration_constant,
                best_state=best,
            )

        return Expression(best.best_sequence)
