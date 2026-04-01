import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, loading

from mcsr.algos.uct import UCT
from mcsr.tree.grammar import Grammar
from mcsr.utils.loading import SRSDLoader
from mcsr.utils.predictor import PredictorNN, atom_key


class ImprovedSymbolicDataset(Dataset):
    """
    Generates training trajectories by running UCT on provided problems.
    Instead of using visit counts to guide the neural network prior,
    it computes a soft-target policy based on the Q-values (mean_score)
    of each valid action. It also optionally injects synthetic priors
    favoring simple shapes and expressions.
    """

    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.vocab_size = len(grammar.all_atoms)
        self.atom_to_index = {atom_key(a): i for i, a in enumerate(grammar.all_atoms)}
        self.data = []

    def generate_data(
        self,
        problems,
        num_iterations=10000,
        max_atoms=15,
        cache_file="data/trajectories_cache_v2.pt",
    ):
        if os.path.exists(cache_file):
            print(f"Loading MCTS trajectories from cache: {cache_file}")
            self.data = torch.load(cache_file, weights_only=True)
            return

        print(
            f"Generating optimized MCTS trajectories for {len(problems)} equations..."
        )
        for p_idx, problem in enumerate(problems):
            print(
                f"  [{p_idx + 1}/{len(problems)}] Searching equation {problem['name']}..."
            )
            # Use UCT to explore the tree
            uct = UCT(
                grammar=self.grammar, max_atoms=max_atoms, num_iterations=num_iterations
            )
            X_train, y_train = problem["train"]
            _ = uct.fit(X_train, y_train)

            tree_root = getattr(uct, "root", None)
            if tree_root is None:
                continue

            q = [(tree_root, [])]
            added = 0

            while q:
                node, seq_atoms = q.pop(0)
                if len(node.children) > 0:
                    seq_indices = []
                    for a in seq_atoms:
                        key = atom_key(a)
                        idx = self.atom_to_index.get(key, 0)
                        seq_indices.append(idx)

                    if not seq_indices:
                        seq_indices = [0]

                    mean_score = node.mean_score
                    if np.isnan(mean_score) or np.isinf(mean_score):
                        mean_score = -100.0
                    raw_fitness = max(mean_score, -100.0)
                    value = (raw_fitness / 100.0) + 1.0
                    value = max(0.0, min(1.0, value))
                    value = value * 2 - 1

                    # Policy target: Distribution over children's Q-values instead of visit count
                    policy = np.zeros(self.vocab_size, dtype=np.float32)

                    children_q_values = []
                    children_indices = []

                    for c in node.children:
                        if c.atom is not None:
                            key = atom_key(c.atom)
                            idx = self.atom_to_index.get(key, 0)

                            score = c.mean_score
                            if np.isnan(score) or np.isinf(score):
                                score = -100.0

                            children_q_values.append(score)
                            children_indices.append(idx)

                    if children_q_values:
                        scores = np.array(children_q_values)
                        temperature = 5.0
                        # Softmax with temperature
                        exp_scores = np.exp((scores - np.max(scores)) / temperature)
                        probs = exp_scores / np.sum(exp_scores)

                        for idx, prob in zip(children_indices, probs):
                            policy[idx] += prob

                        self.data.append(
                            (
                                torch.tensor(seq_indices, dtype=torch.long),
                                torch.tensor([value], dtype=torch.float32),
                                torch.tensor(policy, dtype=torch.float32),
                            )
                        )
                        added += 1

                for c in node.children:
                    q.append((c, seq_atoms + [c.atom]))
            print(f"  -> Extracted {added} states from {problem.name}")

        print(f"Saving generated trajectories to {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(self.data, cache_file)

    def inject_synthetic_patterns(self):
        print("Injecting synthetic priors for simple shapes...")
        v_atoms = [a for a in self.grammar.all_atoms if a.name.startswith("x")]
        c_atoms = [a for a in self.grammar.all_atoms if a.name.startswith("c")]
        add_atom = next((a for a in self.grammar.all_atoms if a.name == "+"), None)
        mul_atom = next((a for a in self.grammar.all_atoms if a.name == "*"), None)
        sin_atom = next((a for a in self.grammar.all_atoms if a.name == "sin"), None)

        synthetic_sequences = []

        for v in v_atoms:
            synthetic_sequences.append([v])
        if add_atom:
            for v1 in v_atoms:
                for v2 in v_atoms:
                    synthetic_sequences.append([add_atom, v1, v2])
        if mul_atom:
            for v1 in v_atoms:
                for v2 in v_atoms:
                    synthetic_sequences.append([mul_atom, v1, v2])
        if sin_atom:
            for v in v_atoms:
                synthetic_sequences.append([sin_atom, v])
        if add_atom and c_atoms:
            for c in c_atoms[:1]:
                for v in v_atoms:
                    synthetic_sequences.append([add_atom, c, v])

        # Weight of synthetic data copies
        multiplier = 20
        for seq in synthetic_sequences:
            for _ in range(multiplier):
                for i in range(len(seq)):
                    partial = seq[:i]
                    target_atom = seq[i]

                    seq_indices = []
                    for a in partial:
                        idx = self.atom_to_index.get(atom_key(a), 0)
                        seq_indices.append(idx)
                    if not seq_indices:
                        seq_indices = [0]

                    policy = np.zeros(self.vocab_size, dtype=np.float32)
                    target_idx = self.atom_to_index.get(atom_key(target_atom), 0)
                    policy[target_idx] = 1.0  # Hard label for ideal next choice

                    self.data.append(
                        (
                            torch.tensor(seq_indices, dtype=torch.long),
                            torch.tensor(
                                [1.0], dtype=torch.float32
                            ),  # High confidence/value
                            torch.tensor(policy, dtype=torch.float32),
                        )
                    )
        print(
            f"  -> Added {len(synthetic_sequences) * multiplier} synthetic trajectories constraints."
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    sequences, values, policies = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = nn.utils.rnn.pad_sequence(
        sequences, batch_first=True, padding_value=0
    )
    values = torch.stack(values)
    policies = torch.stack(policies)
    return padded_seqs, lengths, values, policies


def train():
    import argparse

    parser = argparse.ArgumentParser(
        description="Train AlphaZero PredictorNN - Phase 2"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--uct-iters",
        type=int,
        default=20000,
        help="UCT iterations for trajectory generation",
    )
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv()

    print("Loading datasets...")
    from huggingface_hub import list_repo_files

    repo_files = list_repo_files(
        "yoshitomo-matsubara/srsd-feynman_easy", repo_type="dataset"
    )
    equation_names = sorted(
        [
            f.split("/")[1].replace(".txt", "")
            for f in repo_files
            if f.startswith("train/") and f.endswith(".txt")
        ]
    )
    train_eq_names = equation_names[5:25]

    loader = SRSDLoader(splits=("train",))
    train_probs = [loader[name] for name in train_eq_names]

    print("Initializing Grammar and Predictor...")
    grammar = Grammar()
    grammar.set_variables(5)
    model = PredictorNN(grammar=grammar)

    checkpoint_dir = Path("artifacts/checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)

    # Load previously trained checkpoint if available
    prev_ckpt = checkpoint_dir / "predictor_epoch_final.pt"
    if os.path.exists(prev_ckpt):
        print(f"Loading prior model from phase 1: {prev_ckpt}")
        model.load_state_dict(
            torch.load(prev_ckpt, map_location="cpu", weights_only=True)
        )
    else:
        print("No prior checkpoint found. Training from scratch.")

    dataset = ImprovedSymbolicDataset(grammar)
    dataset.generate_data(train_probs, num_iterations=args.uct_iters, max_atoms=15)
    dataset.inject_synthetic_patterns()  # Add strong priors for fundamental sequences

    if len(dataset) == 0:
        print("No data extracted! Aborting.")
        return

    loading = loading(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-5
    )

    value_criterion = nn.MSELoss()
    policy_criterion = nn.KLDivLoss(reduction="batchmean")

    epochs = args.epochs
    print(f"Starting Phase 2 Training with {len(dataset)} items for {epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_v_loss = 0.0
        total_p_loss = 0.0

        for batch_idx, (seqs, lengths, values, policies) in enumerate(loading):
            optimizer.zero_grad()
            pred_values, pred_policies = model(seqs, lengths)

            loss_v = value_criterion(pred_values, values.squeeze(1))

            # Use KLDivLoss for density distributions
            # Target `policies` is a probability distribution.
            log_probs = torch.nn.functional.log_softmax(pred_policies, dim=1)
            loss_p = policy_criterion(log_probs, policies)

            loss = loss_v + loss_p
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            total_v_loss += loss_v.item()
            total_p_loss += loss_p.item()

        scheduler.step()
        n_batches = len(loading)
        avg_loss = total_loss / n_batches
        avg_v = total_v_loss / n_batches
        avg_p = total_p_loss / n_batches
        lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} (V:{avg_v:.4f} P:{avg_p:.4f}) | LR: {lr:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = checkpoint_dir / "predictor_v2_best.pt"
            torch.save(model.state_dict(), best_path)

    final_path = checkpoint_dir / "predictor_v2_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f" -> Final Checkpoint Phase 2 saved to {final_path}")
    print(
        f" -> Best Checkpoint Phase 2 saved to {checkpoint_dir / 'predictor_v2_best.pt'} (loss: {best_loss:.4f})"
    )
    print("Phase 2 training finished successfully.")


if __name__ == "__main__":
    train()
