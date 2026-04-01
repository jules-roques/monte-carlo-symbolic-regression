from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from mcsr.algos.interface import SRAlgorithm
from mcsr.tree.expression import Expression
from mcsr.tree.grammar import Atom, Grammar
from mcsr.utils.metrics import compute_fitness


@dataclass
class _BestState:
    best_sequence: list[Atom] = field(default_factory=list)
    best_fitness: float = -float("inf")


def _update_best(state: _BestState, sequence: list[Atom], fitness: float) -> None:
    if fitness > state.best_fitness:
        state.best_fitness = fitness
        state.best_sequence = list(sequence)


def evaluate_sequence(
    sequence: list[Atom],
    input_data: np.ndarray,
    target: np.ndarray,
) -> float:
    try:
        predicted = Expression(sequence).compute(input_data)
        return compute_fitness(predicted, target)
    except Exception:
        return 0.0


def random_playout(
    partial_sequence: list[Atom],
    remaining_leaves: int,
    max_atoms: int,
    grammar: Grammar,
    input_data: np.ndarray,
    target: np.ndarray,
) -> tuple[list[Atom], float]:
    sequence = list(partial_sequence)
    leaves = remaining_leaves
    current_index = len(sequence)

    while leaves > 0:
        valid_atoms = grammar.get_valid_atoms(leaves, max_atoms, current_index)
        if not valid_atoms:
            valid_atoms = grammar.terminal_atoms
        atom = random.choice(valid_atoms)
        sequence.append(atom)
        current_index += 1
        leaves += atom.arity - 1

    score = evaluate_sequence(sequence, input_data, target)
    return sequence, score


def nested_search(
    level: int,
    partial_sequence: list[Atom],
    remaining_leaves: int,
    max_atoms: int,
    grammar: Grammar,
    input_data: np.ndarray,
    target: np.ndarray,
    best_state: _BestState,
) -> tuple[list[Atom], float]:
    if remaining_leaves == 0:
        score = evaluate_sequence(partial_sequence, input_data, target)
        _update_best(best_state, partial_sequence, score)
        return list(partial_sequence), score

    if level == 0:
        completed_sequence, score = random_playout(
            partial_sequence,
            remaining_leaves,
            max_atoms,
            grammar,
            input_data,
            target,
        )
        _update_best(best_state, completed_sequence, score)
        return completed_sequence, score

    sequence = list(partial_sequence)
    leaves = remaining_leaves

    while leaves > 0:
        valid_atoms = grammar.get_valid_atoms(leaves, max_atoms, len(sequence))
        if not valid_atoms:
            valid_atoms = grammar.terminal_atoms

        best_atom = None
        best_score = -float("inf")
        best_completed_sequence = None

        for atom in valid_atoms:
            new_sequence = sequence + [atom]
            new_leaves = leaves + atom.arity - 1

            completed_sequence, score = nested_search(
                level - 1,
                new_sequence,
                new_leaves,
                max_atoms,
                grammar,
                input_data,
                target,
                best_state,
            )

            if score > best_score:
                best_score = score
                best_atom = atom
                best_completed_sequence = completed_sequence

        assert best_atom is not None
        sequence.append(best_atom)
        leaves += best_atom.arity - 1

        if best_completed_sequence is not None:
            _update_best(best_state, best_completed_sequence, best_score)

    final_score = evaluate_sequence(sequence, input_data, target)
    _update_best(best_state, sequence, final_score)
    return sequence, final_score


class NMCTS(SRAlgorithm):
    """Nested Monte-Carlo Search (NMCS) for Symbolic Regression."""

    def __init__(
        self,
        grammar: Grammar,
        max_atoms: int = 15,
        nesting_level: int = 1,
        num_restarts: int = 5,
        seed: Optional[int] = None,
    ):
        super().__init__(grammar=grammar, max_atoms=max_atoms)
        self.nesting_level = nesting_level
        self.num_restarts = num_restarts

    def _fit(self, input_data: np.ndarray, target: np.ndarray) -> Expression:
        best = _BestState()

        for _ in range(self.num_restarts):
            completed_sequence, score = nested_search(
                level=self.nesting_level,
                partial_sequence=[],
                remaining_leaves=1,
                max_atoms=self.max_atoms,
                grammar=self.grammar,
                input_data=input_data,
                target=target,
                best_state=best,
            )
            _update_best(best, completed_sequence, score)

        return Expression(best.best_sequence)
