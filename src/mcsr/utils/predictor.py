from __future__ import annotations

import random
from typing import Protocol, Dict, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from mcsr.tree.atom import Atom, Constant, Variable
from mcsr.tree.grammar import Grammar


def atom_key(a: Atom) -> str:
    """Return a stable, deterministic key for an Atom that is consistent
    across different Grammar instances.  This replaces the broken id()-based
    mapping that changes every time a new Grammar object is created."""
    if isinstance(a, Constant):
        return f"const_{a.value}"
    if isinstance(a, Variable):
        return f"var_{a.var_index}"
    # Operators have unique names: +, -, *, /, sin, cos, exp, log, sqrt
    return f"op_{a.name}"


class PredictorInterface(Protocol):
    def predict(
        self, sequence: List[Atom], valid_atoms: List[Atom], grammar: Grammar
    ) -> Tuple[float, Dict[Atom, float]]:
        """
        Given a partial sequence and valid next atoms, predict:
        - value: float in [-1, 1] estimating how good the sequence is
        - policy: dict mapping each valid_atom to its prior probability
        """
        ...


class DummyPredictor(PredictorInterface):
    """A dummy predictor that returns random values and uniform policy."""

    def predict(
        self, sequence: List[Atom], valid_atoms: List[Atom], grammar: Grammar
    ) -> Tuple[float, Dict[Atom, float]]:
        # Simulated value in [-1, 1]
        v = random.uniform(-1.0, 1.0)

        # Uniform policy
        if not valid_atoms:
            return v, {}

        prob = 1.0 / len(valid_atoms)
        policy = {a: prob for a in valid_atoms}
        return v, policy


class PredictorNN(nn.Module, PredictorInterface):
    """
    A simple Recurrent Neural Network for predicting Value and Policy over Atom sequences.
    """

    def __init__(
        self, grammar: Grammar, embedding_dim: int = 64, hidden_dim: int = 128
    ):
        super().__init__()
        self.grammar = grammar
        self.hidden_dim = hidden_dim

        # Create a vocabulary from the grammar using stable keys (not id())
        self.atom_to_idx = {atom_key(a): i for i, a in enumerate(grammar.all_atoms)}
        self.vocab_size = len(self.atom_to_idx)

        self.embedding = nn.Embedding(
            self.vocab_size + 1, embedding_dim, padding_idx=self.vocab_size
        )
        self.gru = nn.GRU(
            embedding_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1
        )

        # Policy head outputs logits for each atom in the vocabulary
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, self.vocab_size),
        )
        # Value head outputs a single scalar in [-1, 1] via tanh
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def _seq_to_tensor(self, sequence: List[Atom]) -> torch.Tensor:
        # If sequence is empty, we use a special empty tensor or a dummy token.
        indices = [
            self.atom_to_idx[atom_key(a)]
            for a in sequence
            if atom_key(a) in self.atom_to_idx
        ]
        if not indices:
            indices = [self.vocab_size]  # Pad token as sequence start
        return torch.tensor(indices, dtype=torch.long).unsqueeze(
            0
        )  # Shape: (1, seq_len)

    @torch.no_grad()
    def predict(
        self, sequence: List[Atom], valid_atoms: List[Atom], grammar: Grammar
    ) -> Tuple[float, Dict[Atom, float]]:
        # Set to eval mode for single step prediction
        training = self.training
        self.eval()

        # Forward pass
        x = self._seq_to_tensor(sequence).to(next(self.parameters()).device)
        emb = self.embedding(x)
        out, _ = self.gru(emb)
        last_hidden = out[:, -1, :]  # (1, hidden_dim)

        value = self.value_head(last_hidden).squeeze().item()
        logits = self.policy_head(last_hidden).squeeze(0)  # (vocab_size,)

        # Mask invalid atoms
        valid_indices = []
        valid_atoms_filtered = []
        for a in valid_atoms:
            key = atom_key(a)
            if key in self.atom_to_idx:
                valid_indices.append(self.atom_to_idx[key])
                valid_atoms_filtered.append(a)

        if not valid_indices:
            self.train(training)
            return value, {}

        # Extract logits for valid actions only
        valid_logits = logits[valid_indices]
        probs = F.softmax(valid_logits, dim=0)

        policy = {a: p.item() for a, p in zip(valid_atoms_filtered, probs)}

        self.train(training)
        return value, policy

    def forward(
        self, sequences: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        sequences: (batch_size, max_seq_len)
        lengths: (batch_size,)
        Returns: values, policy_logits
        """
        emb = self.embedding(sequences)
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed_emb)
        out, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True)

        # Get the representation at the last valid timestep for each sequence
        batch_size = sequences.size(0)
        idx = (lengths - 1).view(-1, 1).expand(batch_size, self.hidden_dim).unsqueeze(1)
        last_hidden = out.gather(1, idx).squeeze(1)

        values = self.value_head(last_hidden).squeeze(1)
        policy_logits = self.policy_head(last_hidden)
        return values, policy_logits
