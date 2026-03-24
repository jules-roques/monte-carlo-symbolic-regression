#!/usr/bin/env python3
"""Benchmark runner: evaluate UCT symbolic regression on SRSD-easy equations (Validation Set)."""
from __future__ import annotations

import argparse
import json
import importlib
import sys
import time
from typing import Optional

import numpy as np

from mcsr.utils.data_loader import load_srsd_easy_problems, SRSDProblem
from mcsr.utils.fitness import compute_fitness, compute_r_squared
from mcsr.tree.grammar import Grammar


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SR on SRSD-Easy (Validation)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file for algorithm")
    parser.add_argument("--equations", type=str, default=None, help="Comma-separated class names")
    parser.add_argument("--num-equations", type=int, default=5, help="Number of max equations to evaluate")
    parser.add_argument("--output", type=str, default=None, help="JSON output path")
    return parser.parse_args()


def run_single_problem(problem: SRSDProblem, config: dict) -> dict:
    grammar = Grammar(num_variables=problem.num_vars)
    
    algo_config = config.get("algorithm", {})
    module_name = algo_config.get("module", "mcsr.algos.uct")
    class_name = algo_config.get("class_name", "UCT")
    kwargs = algo_config.get("kwargs", {})
    
    try:
        module = importlib.import_module(module_name)
        AlgoClass = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading algorithm {module_name}.{class_name}: {e}")
        sys.exit(1)

    algo = AlgoClass(grammar=grammar, **kwargs)

    start_time = time.time()
    best_expression, best_fitness = algo.fit(
        input_data=problem.train_x,
        target=problem.train_y,
    )
    elapsed = time.time() - start_time

    expression_str = str(best_expression)
    train_predicted = best_expression.evaluate(problem.train_x)
    val_predicted = best_expression.evaluate(problem.val_x)

    train_r2 = compute_r_squared(train_predicted, problem.train_y)
    val_r2 = compute_r_squared(val_predicted, problem.val_y)
    train_fitness = compute_fitness(train_predicted, problem.train_y)
    val_fitness = compute_fitness(val_predicted, problem.val_y)

    return {
        "name": problem.name,
        "class_name": problem.class_name,
        "discovered_expression": expression_str,
        "num_atoms": len(best_expression.atom_sequence),
        "train_fitness": float(round(train_fitness, 6)),
        "val_fitness": float(round(val_fitness, 6)),
        "train_r2": float(round(train_r2, 6)),
        "val_r2": float(round(val_r2, 6)),
        "elapsed_seconds": round(elapsed, 2),
        "config": kwargs,
    }


def print_results_table(results: list[dict]) -> None:
    header = f"{'Equation':<25} {'Train R²':>10} {'Val R²':>10} {'Atoms':>6} {'Time(s)':>8}  Expression"
    print("\n" + "=" * 120)
    print("SR Validation Results")
    print("=" * 120)
    print(header)
    print("-" * 120)

    for r in results:
        print(
            f"{r['name']:<25} {r['train_r2']:>10.4f} {r['val_r2']:>10.4f} "
            f"{r['num_atoms']:>6} {r['elapsed_seconds']:>8.1f}  {r['discovered_expression']}"
        )

    print("-" * 120)
    val_r2_values = [r["val_r2"] for r in results]
    print(f"{'Mean':25} {'':>10} {float(np.mean(val_r2_values)):>10.4f}")
    print(f"{'Median':25} {'':>10} {float(np.median(val_r2_values)):>10.4f}")
    print("=" * 120)


def main() -> None:
    args = parse_arguments()
    
    with open(args.config, "r") as f:
        config = json.load(f)

    equation_filter = [e.strip() for e in args.equations.split(",")] if args.equations else None

    print("Loading SRSD-easy problems...")
    problems = load_srsd_easy_problems(equation_filter=equation_filter)
    if not equation_filter and args.num_equations is not None and args.num_equations > 0:
        problems = problems[:args.num_equations]
    
    results: list[dict] = []
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem.name} ({problem.num_vars} vars)")
        result = run_single_problem(problem, config)
        print(f"  → Train R²={result['train_r2']:.4f}  Val R²={result['val_r2']:.4f} ({result['elapsed_seconds']:.1f}s) {result['discovered_expression']}")
        results.append(result)

    print_results_table(results)

    import os
    import datetime
    
    output_path = args.output
    if not output_path:
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        algo_name = config.get("algorithm", {}).get("class_name", "Unknown")
        output_path = f"logs/val_{algo_name}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

if __name__ == "__main__":
    main()
