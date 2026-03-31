import numpy as np
import sympy
import zss


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


def compute_ned(*, pred: sympy.Expr, truth: sympy.Expr) -> float:
    """
    Computes the Normalized Edit Distance of a predicted expression from the true expression.
    - Simplifies both to canonical forms.
    - Masks constants to focus on structure.
    - Normalizes by the size of the target expression.
    """

    size_truth = _count_nodes(truth)
    size_pred = _count_nodes(pred)

    if size_truth == 0:
        return 1.0 if size_pred > 0 else 0.0

    edit_distance = compute_edit_distance(pred, truth)

    return min(edit_distance / size_truth, 1.0)


def compute_edit_distance(expr_a: sympy.Expr, expr_b: sympy.Expr) -> float:
    """
    Computes the edit distance between two SymPy expressions.
    """

    canon_a = sympy.simplify(sympy.expand(expr_a))
    canon_b = sympy.simplify(sympy.expand(expr_b))

    tree_a = _sympy_to_zss_canonical(canon_a)
    tree_b = _sympy_to_zss_canonical(canon_b)

    edit_dist = zss.simple_distance(tree_a, tree_b)

    return edit_dist


def _sympy_to_zss_canonical(expr: sympy.Basic) -> zss.Node:
    """
    Recursively converts a SymPy expression to a ZSS tree with:
    1. Constant masking (all numbers become 'CONST').
    2. Commutative sorting (handled by SymPy's internal arg ordering).
    """

    if expr.is_Number:
        return zss.Node("CONST")

    if not expr.args:
        return zss.Node(str(expr))

    operator_name = expr.func.__name__
    node = zss.Node(operator_name)

    for arg in expr.args:
        node.addkid(_sympy_to_zss_canonical(arg))

    return node


def _count_nodes(expr: sympy.Expr) -> int:
    """Returns the total number of nodes in the SymPy expression tree."""
    return sum(1 for _ in sympy.preorder_traversal(expr))
