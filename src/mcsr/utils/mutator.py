from typing import Optional, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from mcsr.tree.atom import Atom, Constant, Variable
from mcsr.tree.expression import Expression
from mcsr.tree.grammar import Grammar


def atom_key(a: Atom) -> str:
    if isinstance(a, Constant):
        return f"const_{a.value}"
    if isinstance(a, Variable):
        return f"var_{a.var_index}"
    return f"op_{a.name}"


class MutatorInterface(Protocol):
    def mutate(
        self,
        expression: Expression,
        num_mutations: int,
        grammar: Optional[Grammar] = None,
    ) -> list[tuple[Expression, float]]:
        """
        Génère de nouvelles expressions (mutations) à partir d'une expression mère.
        Retourne une liste de paires (nouvelle_expression, log_probabilité).
        """
        ...


class MutatorNN(nn.Module, MutatorInterface):
    """
    Réseau mutateur Seq2Seq.
    Prend en entrée une expression e existante et génère une expression mutée e'
    conditionnée sur e. Représente p_theta(e' | e).
    """

    def __init__(
        self,
        grammar: Optional[Grammar] = None,
        max_atoms: int = 15,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        # On utilise une grammaire standard à 5 variables pour la cohérence des modèles
        if grammar is None:
            grammar = Grammar()
            grammar.set_variables(5)

        self.grammar = grammar
        self.max_atoms = max_atoms
        self.hidden_dim = hidden_dim

        # Vocabulaire stable basé sur atom_key
        self.atom_to_idx = {atom_key(a): i for i, a in enumerate(grammar.all_atoms)}
        self.idx_to_atom = {i: a for i, a in enumerate(grammar.all_atoms)}
        self.vocab_size = len(self.atom_to_idx)

        self.embedding = nn.Embedding(
            self.vocab_size + 1, embedding_dim, padding_idx=self.vocab_size
        )
        self.encoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.decoder = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, self.vocab_size)

    def _seq_to_tensor(self, sequence: list[Atom]) -> torch.Tensor:
        indices = [self.atom_to_idx.get(atom_key(a), self.vocab_size) for a in sequence]
        if not indices:
            indices = [self.vocab_size]
        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    def forward(
        self, parent_sequences: torch.Tensor, child_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Calcule la log-probabilité log p_theta(e' | e) pour un batch de paires (e, e').
        parent_sequences: (batch_size, seq_len_e)
        child_sequences: (batch_size, seq_len_e')
        Retourne: (batch_size,) log-probs
        """
        device = parent_sequences.device
        batch_size = parent_sequences.size(0)

        # 1. Encodage des parents
        parent_emb = self.embedding(parent_sequences)
        _, hidden = self.encoder(parent_emb)  # hidden: (1, batch_size, hidden_dim)

        # 2. Décodage des enfants (Teacher Forcing)
        # On ajoute un token de départ (padding_idx) au début des séquences enfants
        start_tokens = torch.full(
            (batch_size, 1), self.vocab_size, dtype=torch.long, device=device
        )
        decoder_input = torch.cat([start_tokens, child_sequences[:, :-1]], dim=1)

        dec_emb = self.embedding(decoder_input)
        dec_out, _ = self.decoder(dec_emb, hidden)  # (batch_size, seq_len, hidden_dim)

        logits = self.fc_out(dec_out)  # (batch_size, seq_len, vocab_size)

        # On calcule les log_probs
        log_probs_all = F.log_softmax(logits, dim=-1)

        # On récupère les log_probs des atomes effectivement choisis
        # child_sequences: (batch_size, seq_len) - can contain vocab_size as padding index
        # We need to clamp to stay within log_probs_all (0 to vocab_size-1) before gathering
        capped_child = torch.clamp(child_sequences, 0, self.vocab_size - 1)
        target_log_probs = log_probs_all.gather(2, capped_child.unsqueeze(2)).squeeze(2)

        # Masking des tokens de padding dans l'enfant pour ne pas influencer la perte
        mask = (child_sequences != self.vocab_size).float()
        log_prob = (target_log_probs * mask).sum(dim=1)

        return log_prob

    @torch.no_grad()
    def mutate(
        self,
        expression: Expression,
        num_mutations: int,
        grammar: Optional[Grammar] = None,
    ) -> list[tuple[Expression, float]]:
        training = self.training
        self.eval()

        device = next(self.parameters()).device
        sampling_grammar = grammar or self.grammar

        # 1. Encodage de l'expression courante
        x = self._seq_to_tensor(expression.atom_sequence).to(device)
        emb = self.embedding(x)
        _, hidden = self.encoder(emb)

        mutations = []
        for _ in range(num_mutations):
            seq = []
            remaining_leaves = 1
            curr_hidden = hidden
            log_prob = 0.0

            # Start token (padding)
            inp = torch.tensor([[self.vocab_size]], dtype=torch.long, device=device)

            while remaining_leaves > 0:
                if len(seq) >= self.max_atoms:
                    break

                valid_atoms = sampling_grammar.get_valid_atoms(
                    remaining_leaves, self.max_atoms, len(seq)
                )
                if not valid_atoms:
                    valid_atoms = sampling_grammar.terminal_atoms

                valid_indices = [
                    self.atom_to_idx[atom_key(a)]
                    for a in valid_atoms
                    if atom_key(a) in self.atom_to_idx
                ]
                if not valid_indices:
                    break

                dec_emb = self.embedding(inp)
                out, curr_hidden = self.decoder(dec_emb, curr_hidden)
                logits = self.fc_out(out.squeeze(1))  # (1, vocab_size)

                # Masquage des atomes invalides
                mask = torch.full_like(logits, -float("inf"))
                mask[0, valid_indices] = logits[0, valid_indices]
                probs = F.softmax(mask, dim=-1)

                # Echantillonnage
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob += dist.log_prob(action).item()

                chosen_idx = action.item()
                chosen_atom = self.idx_to_atom[chosen_idx]
                seq.append(chosen_atom)
                remaining_leaves += chosen_atom.arity - 1

                inp = action.unsqueeze(0)

            if remaining_leaves == 0:
                mutations.append((Expression(seq), log_prob))

        # Sécurité : au cas où aucune mutation n'a abouti (taille depassée avant complétion)
        if not mutations:
            mutations.append((expression, 0.0))

        self.train(training)
        return mutations


class DummyMutator(MutatorInterface):
    """Mutateur basique générant des expressions aléatoires pour test."""

    def __init__(self, grammar: Grammar, max_atoms: int = 15):
        self.grammar = grammar
        self.max_atoms = max_atoms

    def mutate(
        self,
        expression: Expression,
        num_mutations: int,
        grammar: Optional[Grammar] = None,
    ) -> list[tuple[Expression, float]]:
        mutations = []
        import random

        sampling_grammar = grammar or self.grammar
        for _ in range(num_mutations):
            seq = []
            leaves = 1
            while leaves > 0 and len(seq) < self.max_atoms:
                valid = sampling_grammar.get_valid_atoms(
                    leaves, self.max_atoms, len(seq)
                )
                if not valid:
                    valid = sampling_grammar.terminal_atoms
                a = random.choice(valid)
                seq.append(a)
                leaves += a.arity - 1
            if leaves == 0:
                mutations.append((Expression(seq), 0.0))
        if not mutations:
            mutations.append((expression, 0.0))
        return mutations
