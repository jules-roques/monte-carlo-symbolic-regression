import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import List

from mcsr.tree.grammar import Grammar
from mcsr.tree.expression import Expression
from mcsr.utils.mutator import MutatorNN, atom_key
from mcsr.algos.dgsr_mcts import DGSR_MCTS

def generate_synthetic_data(problem_id: int) -> tuple[np.ndarray, np.ndarray, str]:
    """Génère différents jeux de données x -> y pour varier l'apprentissage."""
    # On génère 5 variables en entrée
    x = np.random.uniform(-3, 3, size=(100, 5))
    
    if problem_id % 5 == 0:
        y = np.sin(x[:, 0]) + x[:, 0]**2
        name = "sin(x0) + x0^2"
    elif problem_id % 5 == 1:
        y = np.exp(x[:, 0]) - 2*x[:, 1]
        name = "exp(x0) - 2x1"
    elif problem_id % 5 == 2:
        y = x[:, 0] * x[:, 1] + x[:, 2]
        name = "x0*x1 + x2"
    elif problem_id % 5 == 3:
        y = np.cos(x[:, 0]) / (1 + x[:, 1]**2)
        name = "cos(x0) / (1 + x1^2)"
    else:
        y = x[:, 0]**3 - x[:, 0] + 1
        name = "x0^3 - x0 + 1"
        
    return x, y, name

def prepare_batch(trajectories, mutator, device):
    """Prépare des tensors paddés à partir des trajectoires (e, e', reward)."""
    parents = []
    children = []
    rewards = []
    
    vocab_size = mutator.vocab_size
    
    for p_expr, c_expr, r in trajectories:
        p_idx = [mutator.atom_to_idx.get(atom_key(a), vocab_size) for a in p_expr.atom_sequence]
        c_idx = [mutator.atom_to_idx.get(atom_key(a), vocab_size) for a in c_expr.atom_sequence]
        
        if not p_idx: p_idx = [vocab_size]
        if not c_idx: c_idx = [vocab_size]
        
        parents.append(torch.tensor(p_idx, dtype=torch.long))
        children.append(torch.tensor(c_idx, dtype=torch.long))
        rewards.append(r)
        
    if not parents:
        return None, None, None
        
    # Padding
    p_padded = nn.utils.rnn.pad_sequence(parents, batch_first=True, padding_value=vocab_size)
    c_padded = nn.utils.rnn.pad_sequence(children, batch_first=True, padding_value=vocab_size)
    r_tensor = torch.tensor(rewards, dtype=torch.float32, device=device)
    
    return p_padded.to(device), c_padded.to(device), r_tensor

def train_dgsr_mcts():
    print("=== Configuration DGSR-MCTS Training ===")
    # On utilise 5 variables pour être compatible avec la plupart des problèmes
    grammar = Grammar(num_variables=5)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device : {device}")
    
    # MutatorNN utilisera par défaut 5 vars si initialisé sans grammaire ou avec cette rammaire
    mutator = MutatorNN(grammar, max_atoms=15).to(device)
    optimizer = optim.Adam(mutator.parameters(), lr=1e-4)
    
    num_epochs = 10
    mcts_iters = 300
    num_problems = 5 # Plus de diversité
    
    print(f"Début de l'entraînement : {num_epochs} époques, {mcts_iters} itérations MCTS par problème.")
    
    for epoch in range(num_epochs):
        all_trajectories = []
        epoch_scores = []
        best_exprs_found = []
        
        # 1. Collecte de trajectoires sur plusieurs problèmes
        for prob_id in range(num_problems):
            x, y, name = generate_synthetic_data(prob_id + epoch * num_problems)
            
            algo = DGSR_MCTS(
                grammar=grammar, 
                max_atoms=15, 
                num_iterations=mcts_iters, 
                mutator=mutator,
                num_mutations_per_expansion=5
            )
            
            best_expr, best_score = algo.fit(x, y)
            all_trajectories.extend(algo.last_trajectories)
            epoch_scores.append(best_score)
            best_exprs_found.append(str(best_expr))
            
        if not all_trajectories:
            print(f"Epoch {epoch+1} - Aucune trajectoire collectée.")
            continue
            
        # 2. Mise à jour du modèle (REINFORCE + Entropy)
        p_tensor, c_tensor, r_tensor = prepare_batch(all_trajectories, mutator, device)
        if p_tensor is None:
            continue
            
        optimizer.zero_grad()
        
        # On a besoin d'accéder aux logits pour l'entropie
        # On va modifier un peu mutator pour qu'il retourne aussi les logits si besoin, 
        # ou on recalcule ici. Pour rester simple, on va juste faire l'update de log_probs.
        log_probs = mutator(p_tensor, c_tensor) # (batch_size,)
        
        # On s'assure que rewards ne sont pas infini
        r_tensor = torch.clamp(r_tensor, min=-100.0, max=1.0)
        
        avg_r = r_tensor.mean()
        std_r = r_tensor.std() + 1e-6
        advantage = (r_tensor - avg_r) / (std_r + 1e-6)
        
        # Perte REINFORCE
        policy_loss = -(advantage * log_probs).mean()
        
        # Entropie simplifiée (on favorise des log_probs pas trop extrêmes sur le batch)
        entropy_loss = -0.01 * log_probs.mean() # Hacky entropy approximation
        
        loss = policy_loss + entropy_loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Epoch {epoch+1} - ALERTE: loss est {loss.item()}, on saute l'update.")
            continue
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mutator.parameters(), max_norm=0.5)
        optimizer.step()
        
        avg_score = sum(epoch_scores) / len(epoch_scores)
        print(f"Epoch {epoch+1}/{num_epochs} | Traj: {len(all_trajectories)} | Loss: {loss.item():.4f} | Avg Best: {avg_score:.4f} | Baseline: {avg_r:.4f}")
        print(f"  Exemples best : {best_exprs_found}")
        
    # Sauvegarde du modèle
    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        'model_state_dict': mutator.state_dict(),
        'vocab': mutator.atom_to_idx,
    }, "checkpoints/mutator_final.pt")
    print("Entraînement terminé. Modèle sauvé dans checkpoints/mutator_final.pt")

if __name__ == "__main__":
    train_dgsr_mcts()
