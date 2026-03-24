from __future__ import annotations

import numpy as np


def compute_fitness(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute fitness as 1 - NRMSE. Returns value in (-inf, 1], 1.0 = perfect fit."""
    valid_mask = np.isfinite(predicted)
    valid_ratio = np.sum(valid_mask) / len(target)

    if valid_ratio < 0.5:
        return -1e6

    predicted_valid = predicted[valid_mask]
    target_valid = target[valid_mask]

    target_std = np.std(target_valid)
    with np.errstate(over="ignore"):
        if target_std < 1e-12:
            rmse = np.sqrt(np.mean((predicted_valid - target_valid) ** 2))
            return -rmse if rmse > 1e-12 else 1.0

        rmse = np.sqrt(np.mean((predicted_valid - target_valid) ** 2))
    nrmse = rmse / target_std
    fitness = (1.0 - nrmse) * valid_ratio
    return fitness


def compute_r_squared(predicted: np.ndarray, target: np.ndarray) -> float:
    """Compute R² (coefficient of determination). Returns value in (-inf, 1]."""
    valid_mask = np.isfinite(predicted)
    if np.sum(valid_mask) < 2:
        return -1e6

    predicted_valid = predicted[valid_mask]
    target_valid = target[valid_mask]

    ss_res = np.sum((target_valid - predicted_valid) ** 2)
    ss_tot = np.sum((target_valid - np.mean(target_valid)) ** 2)

    if ss_tot < 1e-12:
        return 1.0 if ss_res < 1e-12 else -1e6

    return 1.0 - ss_res / ss_tot
