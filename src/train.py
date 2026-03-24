import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import numpy as np

from mcsr.tree.grammar import Grammar
from mcsr.utils.predictor import PredictorNN
from mcsr.algos.uct import UCT
from mcsr.utils.data_loader import load_srsd_easy_problems

class RealSymbolicDataset(Dataset):
    """
    Generates real training trajectories by running MCTS (UCT) on provided problems.
    Extracts the visited sequences mappings to children visit counts (Policy) and fitness (Value).
    """
    def __init__(self, grammar: Grammar):
        self.grammar = grammar
        self.vocab_size = len(grammar.all_atoms)
        self.atom_to_index = {id(a): i for i, a in enumerate(grammar.all_atoms)}
        self.data = []
        
    def generate_data(self, problems, num_iterations=1000, cache_file="data/trajectories_cache.pt"):
        import os
        if os.path.exists(cache_file):
            print(f"Loading MCTS trajectories from cache: {cache_file}")
            self.data = torch.load(cache_file, weights_only=True)
            return

        print(f"Generating real MCTS trajectories for {len(problems)} held-out equations...")
        for p_idx, problem in enumerate(problems):
            print(f"  [{p_idx+1}/{len(problems)}] Searching equation {problem.name}...")
            # We use UCT to generate ground truth trajectories quickly
            uct = UCT(grammar=self.grammar, max_atoms=7, num_iterations=num_iterations, seed=42)
            _ = uct.fit(problem.train_x, problem.train_y)
            
            tree_root = getattr(uct, 'root', None)
            if tree_root is None:
                continue
                
            q = [(tree_root, [])]
            added = 0
            
            while q:
                node, seq_atoms = q.pop(0)
                if len(node.children) > 0:
                    # Extract sequence indices natively
                    seq_indices = []
                    for a in seq_atoms:
                        idx = self.atom_to_index.get(id(a), 0)
                        seq_indices.append(idx)
                    
                    if not seq_indices:
                        seq_indices = [0] # Handle root
                        
                    # Value target: Map UCT fitness [-1e6, 0] to [-1, 1] range
                    raw_fitness = max(node.mean_score, -100.0) 
                    value = (raw_fitness / 100.0) + 1.0 # Map -100 to 0.0, 0 to 1.0
                    value = max(0.0, min(1.0, value)) # cap [0, 1]
                    value = value * 2 - 1 # final range [-1, 1]
                    
                    # Policy target: Distribution over children (AlphaZero Prior formulation)
                    policy = np.zeros(self.vocab_size, dtype=np.float32)
                    total_visits = sum(c.visit_count for c in node.children)
                    
                    if total_visits > 0:
                        for c in node.children:
                            if c.atom is not None:
                                idx = self.atom_to_index.get(id(c.atom), 0)
                                policy[idx] += c.visit_count
                                
                        policy = policy / total_visits
                        
                        self.data.append((
                            torch.tensor(seq_indices, dtype=torch.long),
                            torch.tensor([value], dtype=torch.float32),
                            torch.tensor(policy, dtype=torch.float32)
                        ))
                        added += 1
                        
                for c in node.children:
                    q.append((c, seq_atoms + [c.atom]))
            print(f"  -> Extracted {added} states from {problem.name}")
            
        print(f"Saving generated trajectories to {cache_file}...")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(self.data, cache_file)

    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.data[idx]


def collate_fn(batch):
    sequences, values, policies = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded_seqs = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    values = torch.stack(values)
    policies = torch.stack(policies)
    return padded_seqs, lengths, values, policies


def train():
    import argparse
    parser = argparse.ArgumentParser(description="Train AlphaZero PredictorNN")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    
    print("Loading datasets...")
    # Fetch exactly the last 5 equations without downloading the full 120 eq repository first!
    from huggingface_hub import list_repo_files
    repo_files = list_repo_files("yoshitomo-matsubara/srsd-feynman_easy", repo_type="dataset")
    equation_names = sorted([f.split("/")[1].replace(".txt", "") for f in repo_files if f.startswith("train/") and f.endswith(".txt")])
    train_eq_names = equation_names[-5:]
    
    # This will load ONLY the 5 equations specified!
    train_probs = load_srsd_easy_problems(equation_filter=train_eq_names)
    
    print("Initializing Grammar and Predictor...")
    grammar = Grammar(num_variables=5)
    model = PredictorNN(grammar=grammar)
    
    dataset = RealSymbolicDataset(grammar)
    # Generate AlphaZero data using 1000 iter UCT searches
    dataset.generate_data(train_probs, num_iterations=1000)
    
    if len(dataset) == 0:
        print("No data extracted! Aborting.")
        return
        
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    value_criterion = nn.MSELoss()
    policy_criterion = nn.CrossEntropyLoss()
    
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    epochs = args.epochs
    print(f"Starting Machine Learning Phase on Trajectories for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        
        for batch_idx, (seqs, lengths, values, policies) in enumerate(dataloader):
            optimizer.zero_grad()
            pred_values, pred_policies = model(seqs, lengths)
            loss_v = value_criterion(pred_values, values.squeeze(1))
            loss_p = policy_criterion(pred_policies, policies)
            loss = loss_v + loss_p
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
    
    final_path = checkpoint_dir / "predictor_epoch_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f" -> Final Checkpoint saved to {final_path}")
    print("Training finished successfully.")


if __name__ == "__main__":
    train()
