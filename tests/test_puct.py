import os
import torch
import numpy as np

from mcsr.tree.grammar import Grammar
from mcsr.algos.puct import PUCT
from mcsr.utils.predictor import DummyPredictor, PredictorNN
from mcsr.utils.fitness import compute_r_squared

def test_dummy_puct():
    np.random.seed(42)
    data = np.random.uniform(0.1, 5.0, size=(50, 2))
    target = data[:, 0] * data[:, 1]
    
    grammar = Grammar(num_variables=2)
    predictor = DummyPredictor()
    puct = PUCT(
        grammar=grammar,
        max_atoms=7,
        num_iterations=200,
        exploration_constant=1.0,
        seed=42,
        predictor=predictor
    )
    best_expr, best_fit = puct.fit(data, target)
    
    assert best_expr is not None
    assert len(best_expr.atom_sequence) >= 0

def test_nn_puct():
    grammar = Grammar(num_variables=2)
    predictor = PredictorNN(grammar=grammar)
    
    # Load checkpoint if it exists
    ckpt_path = "checkpoints/predictor_epoch_2.pt"
    if os.path.exists(ckpt_path):
        predictor.load_state_dict(torch.load(ckpt_path, weights_only=True))
    
    np.random.seed(123)
    data = np.random.uniform(0.1, 5.0, size=(50, 2))
    target = data[:, 0] + data[:, 1]
    
    puct = PUCT(
        grammar=grammar,
        max_atoms=5,
        num_iterations=200,
        exploration_constant=1.0,
        seed=123,
        predictor=predictor
    )
    best_expr, best_fit = puct.fit(data, target)
    
    assert best_expr is not None
