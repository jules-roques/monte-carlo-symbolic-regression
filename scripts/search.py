"""
This file is the main script to run the symbolic
regression algorithms on the SRSD benchmark.
It uses the config file to instantiate the algorithm
and then runs it on the dataset.
"""

import argparse
import importlib
import json
import pickle
import time
from pathlib import Path

import prettytable as pt
from tqdm import tqdm

from mcsr.algos.interface import SRAlgorithm
from mcsr.tree.grammar import Grammar
from mcsr.utils.loading import SRSDLoader
from mcsr.utils.metrics import compute_r_squared


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SR on SRSD-Easy (Validation)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON config file for algorithm",
    )
    parser.add_argument(
        "--pickles_path",
        type=str,
        default="artifacts/pickles",
        help="Path to pickle the expressions to a file",
    )

    return parser.parse_args()


def process_one_equation(model: SRAlgorithm, equation_data: dict) -> dict:
    name = equation_data["name"]
    X_train, y_train = equation_data["train"]
    X_val, y_val = equation_data["validation"]

    start_time = time.perf_counter()
    best_expression = model.fit(X_train, y_train)
    elapsed_time = time.perf_counter() - start_time

    y_train_pred = best_expression.compute(X_train)
    y_val_pred = best_expression.compute(X_val)

    return {
        "name": name,
        "train_r2": compute_r_squared(y_train, y_train_pred),
        "val_r2": compute_r_squared(y_val, y_val_pred),
        "discovered_expression": best_expression,
        "num_atoms": len(best_expression.atom_sequence),
        "elapsed_seconds": elapsed_time,
    }


def process_all_equations(model: SRAlgorithm, loader: SRSDLoader) -> list[dict]:
    results = []

    # Wrap the loader in tqdm for a clean progress bar
    for equation_data in tqdm(
        loader,
        desc=f"Searching SRSD-Easy expressions with {model.__class__.__name__}",
    ):
        result = process_one_equation(model, equation_data)
        results.append(result)

    return results


def save_expressions(results: list[dict], pickles_folder_path: Path) -> None:
    pickles_folder_path.mkdir(parents=True, exist_ok=True)

    for result in results:
        file_name = pickles_folder_path / f"{result['name']}.pkl"
        with open(file_name, "wb") as f:
            pickle.dump(result["discovered_expression"], f)

    print(f"Pickles saved to {pickles_folder_path}")


def print_results_table(model_name: str, results: list[dict]) -> None:
    table = pt.PrettyTable(maxwidth=150)

    table.title = (
        f"Results of {model_name.upper()} on SRSD-Easy expressions (TRAIN/VAL splits)"
    )
    table.field_names = [
        "Equation",
        "Train R²",
        "Val R²",
        "Time (s)",
        "Discovered Expr",
    ]

    for result in results:
        table.add_row(
            [
                result["name"].replace("feynman-", ""),
                f"{result['train_r2']:.4f}",
                f"{result['val_r2']:.4f}",
                f"{result['elapsed_seconds']:.1f}",
                str(result["discovered_expression"]),
            ]
        )

    print("\n")
    print(table)


def instantiate_algorithm(config_path: str) -> SRAlgorithm:
    with open(config_path, "r") as f:
        config = json.load(f)

    module_name = config["module"]
    class_name = config["class_name"]
    kwargs = config.get("kwargs", {})
    kwargs["grammar"] = Grammar()

    module = importlib.import_module(module_name)
    algo_class = getattr(module, class_name)

    return algo_class(**kwargs)


def main() -> None:
    args = parse_arguments()

    loader = SRSDLoader(splits=("train", "validation"))
    model = instantiate_algorithm(args.config)

    results = process_all_equations(model, loader)

    print_results_table(model.__class__.__name__, results)

    save_expressions(
        results,
        Path(args.pickles_path) / model.__class__.__name__.lower(),
    )


if __name__ == "__main__":
    main()
