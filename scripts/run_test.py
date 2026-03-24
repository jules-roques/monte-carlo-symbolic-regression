#!/usr/bin/env python3
"""Benchmark runner: evaluate UCT symbolic regression on SRSD-easy equations (Test Set)."""
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
    parser = argparse.ArgumentParser(description="SR on SRSD-Easy (Test)")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file for algorithm")
    parser.add_argument("--equations", type=str, default=None, help="Comma-separated class names")
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
    test_predicted = best_expression.evaluate(problem.test_x)

    train_r2 = compute_r_squared(train_predicted, problem.train_y)
    test_r2 = compute_r_squared(test_predicted, problem.test_y)
    train_fitness = compute_fitness(train_predicted, problem.train_y)
    test_fitness = compute_fitness(test_predicted, problem.test_y)
    
    edit_distance = 1.0
    if problem.true_expression is not None:
        edit_distance = best_expression.distance_to(problem.true_expression)

    return {
        "name": problem.name,
        "class_name": problem.class_name,
        "discovered_expression": expression_str,
        "num_atoms": len(best_expression.atom_sequence),
        "train_fitness": float(round(train_fitness, 6)),
        "test_fitness": float(round(test_fitness, 6)),
        "train_r2": float(round(train_r2, 6)),
        "test_r2": float(round(test_r2, 6)),
        "edit_distance": float(round(edit_distance, 4)),
        "elapsed_seconds": round(elapsed, 2),
    }


def print_results_table(results: list[dict]) -> None:
    header = f"{'Equation':<25} {'Test R²':>10} {'Edit Dist':>10} {'Atoms':>6} {'Time(s)':>8}  Expression"
    print("\n" + "=" * 120)
    print("SR Test Results with Edit Distance")
    print("=" * 120)
    print(header)
    print("-" * 120)

    for r in results:
        print(
            f"{r['name']:<25} {r['test_r2']:>10.4f} {r['edit_distance']:>10.4f} "
            f"{r['num_atoms']:>6} {r['elapsed_seconds']:>8.1f}  {r['discovered_expression']}"
        )

    print("-" * 120)
    test_r2_values = [r["test_r2"] for r in results]
    edit_dists = [r["edit_distance"] for r in results]
    print(f"{'Mean':25} {float(np.mean(test_r2_values)):>10.4f} {float(np.mean(edit_dists)):>10.4f}")
    print(f"{'Median':25} {float(np.median(test_r2_values)):>10.4f} {float(np.median(edit_dists)):>10.4f}")
    print("=" * 120)


def main() -> None:
    args = parse_arguments()
    
    with open(args.config, "r") as f:
        config = json.load(f)

    equation_filter = [e.strip() for e in args.equations.split(",")] if args.equations else None

    print("Loading SRSD-easy problems...")
    problems = load_srsd_easy_problems(equation_filter=equation_filter)
    
    results: list[dict] = []
    for i, problem in enumerate(problems):
        print(f"\n[{i+1}/{len(problems)}] {problem.name} ({problem.num_vars} vars)")
        result = run_single_problem(problem, config)
        print(f"  → Test R²={result['test_r2']:.4f}  Edit Dist={result['edit_distance']:.4f} ({result['elapsed_seconds']:.1f}s) {result['discovered_expression']}")
        results.append(result)

    print_results_table(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
