import argparse
from pathlib import Path

import prettytable as pt
import sympy

from mcsr.tree.expression import Expression
from mcsr.utils.converters import expression_to_sympy
from mcsr.utils.dataloader import (
    SRSDLoader,
    load_pickled_expressions,
    load_true_sympy_expressions,
)
from mcsr.utils.metrics import compute_ned, compute_r_squared


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained SR model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model from which we test pickles",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        required=True,
        choices=["easy", "medium", "hard"],
        help="Difficulty of the equations to solve (easy, medium, hard)",
    )
    parser.add_argument(
        "--pickles_path",
        default="pickles",
        type=str,
        help="Path where are stored the predicted pickle files.",
    )
    return parser.parse_args()


def evaluate_equations(
    loader: SRSDLoader,
    name_to_true_sympy: dict[str, sympy.Expr],
    name_to_pred_expr: dict[str, Expression],
) -> list[dict]:
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
                "test_r2": r2,
                "ned": ned,
                "true_expression": str(true_sympy_expr),
                "discovered_expression": str(pred_sympy_expr),
            }
        )

    return evaluation_results


def print_results_table(model_name: str, difficulty: str, results: list[dict]) -> None:
    table = pt.PrettyTable(maxwidth=150)

    table.title = (
        f"Results of {model_name.upper()} on {difficulty.upper()} equations (TEST set)."
    )
    table.field_names = [
        "Equation",
        "Test R²",
        "NED",
        "True Expr",
        "Discovered Expr",
    ]

    for result in results:
        table.add_row(
            [
                result["name"],
                f"{result['test_r2']:.3f}",
                f"{result['ned']:.2f}",
                result["true_expression"],
                result["discovered_expression"],
            ]
        )

    print()
    print(table)


def main() -> None:
    args = parse_args()

    loader = SRSDLoader(difficulty=args.difficulty, splits=("test",))
    name_to_true_sympy = load_true_sympy_expressions(args.difficulty)
    name_to_pred_expr = load_pickled_expressions(
        Path(args.pickles_path) / args.difficulty / args.model_name
    )

    evaluation_results = evaluate_equations(
        loader, name_to_true_sympy, name_to_pred_expr
    )

    print_results_table(args.model_name, args.difficulty, evaluation_results)


if __name__ == "__main__":
    main()
