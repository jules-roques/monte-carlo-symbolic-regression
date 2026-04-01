"""
This file is the main script to test the symbolic
regression algorithms on the SRSD benchmark.
It uses the expressions predicted by the script 'find.py'
to confront the algorithm to the test set.
"""

import argparse

import prettytable as pt

from mcsr.utils.test_expr import test_pickled_expressions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained SR model.")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model from which we test pickles",
    )
    return parser.parse_args()


def print_results_table(model_name: str, results: list[dict]) -> None:
    table = pt.PrettyTable(maxwidth=150)

    table.set_style(pt.TableStyle.MARKDOWN)

    table.title = (
        f"Results of {model_name.upper()} on SRSD-Easy expressions (TEST set)."
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

    evaluation_results = test_pickled_expressions(args.model_name)

    print_results_table(args.model_name, evaluation_results)


if __name__ == "__main__":
    main()
