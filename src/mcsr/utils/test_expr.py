from pathlib import Path

from mcsr.utils.converters import expression_to_sympy
from mcsr.utils.loading import (
    SRSDLoader,
    load_pickled_expressions,
    load_true_sympy_expressions,
)
from mcsr.utils.metrics import compute_ned, compute_r_squared


def test_pickled_expressions(model_name: str) -> list[dict]:

    pickles_path = Path("artifacts/pickles/") / model_name
    name_to_pred_expr = load_pickled_expressions(pickles_path)
    name_to_true_sympy = load_true_sympy_expressions()

    loader = SRSDLoader(splits=("test",))

    evaluation_results = []
    for _, equation_data in enumerate(loader, start=1):
        name = equation_data["name"]
        x_test, y_test = equation_data["test"]

        true_sympy_expr = name_to_true_sympy[name]
        pred_expr = name_to_pred_expr[name]
        pred_sympy_expr = expression_to_sympy(pred_expr)

        y_pred = pred_expr.compute(x_test)

        r2 = compute_r_squared(y_pred, y_test)
        ned = compute_ned(pred=pred_sympy_expr, truth=true_sympy_expr)

        evaluation_results.append(
            {
                "name": name.replace("feynman-", ""),
                "method": model_name,
                "test_r2": r2,
                "ned": ned,
                "true_expression": str(true_sympy_expr),
                "discovered_expression": str(pred_sympy_expr),
            }
        )

    return evaluation_results
