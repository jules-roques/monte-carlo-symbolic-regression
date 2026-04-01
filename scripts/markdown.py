from pathlib import Path

import prettytable as pt
import sympy
from datasets import load_dataset

from mcsr.utils.test_expr import test_pickled_expressions


def get_latex_with_symbols(expr_str, symbols):
    """
    expr_str: string representation of the expression (e.g., 'exp(-x0/x1)')
    symbols: list of symbols from supp_info.json (e.g., ['$P$', '$\\Delta E$', '$k T$'])
    """
    try:
        expr = sympy.sympify(expr_str)
        # x0 -> symbols[1], x1 -> symbols[2], ...
        subs_map = {}
        for i in range(len(symbols) - 1):
            clean_symbol = symbols[i + 1].replace("$", "")
            subs_map[sympy.Symbol(f"x{i}")] = sympy.Symbol(clean_symbol)

        expr = expr.subs(subs_map)
        return f"${sympy.latex(expr)}$"
    except Exception as e:
        return f"Error: {e}"


def main():
    base_pickles_path = Path("artifacts/pickles")
    output_path = Path("results") / "equations.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not base_pickles_path.exists():
        print(f"Error: Path {base_pickles_path} does not exist.")
        return

    # 1. Load supp_info.json to get variable mappings
    repo_id = "yoshitomo-matsubara/srsd-feynman_easy"
    print(f"Loading metadata from {repo_id}...")
    try:
        # We load the whole dataset to get the full JSON structure
        info_ds = load_dataset(repo_id, data_files="supp_info.json", split="train")
        # supp_info is usually a single row containing a large dictionary
        # Based on my previous inspection, it returns a dictionary-like object
        supp_info = info_ds[0]
    except Exception as e:
        print(f"Failed to load supp_info.json: {e}")
        return

    # 2. Collect results for all models
    all_results = []
    for model_dir in base_pickles_path.iterdir():
        if model_dir.is_dir():
            print(f"Evaluating model: {model_dir.name}...")
            model_results = test_pickled_expressions(
                model_name=model_dir.name,
            )
            # Add method name to each result
            for res in model_results:
                res["method"] = model_dir.name
            all_results.extend(model_results)

    if not all_results:
        print("No results found.")
        return

    # 3. Group results by equation name
    # We want a dictionary: equation_name -> list of method results
    grouped_results = {}
    for res in all_results:
        name = res["name"]
        if name not in grouped_results:
            grouped_results[name] = []
        grouped_results[name].append(res)

    # 4. Create the PrettyTable
    table = pt.PrettyTable()
    table.set_style(pt.TableStyle.MARKDOWN)
    table.field_names = [
        "True Expression (Name & LaTeX)",
        "Guessed Expressions (Method: LaTeX)",
        "R²",
        "NED",
    ]
    table.align = "l"

    # Sort equations by name for a consistent table
    for eq_name in sorted(grouped_results.keys()):
        method_results = grouped_results[eq_name]

        # Get metadata for this equation
        # Note: eq_name in results might be without 'feynman-' prefix
        # but in supp_info it has it.
        full_name = f"feynman-{eq_name}"
        if full_name not in supp_info:
            # Try without prefix if it fails
            full_name = eq_name

        metadata = supp_info.get(full_name)
        if not metadata:
            print(f"Warning: Metadata for {full_name} not found in supp_info.json")
            symbols = ["$y$"] + [f"$x_{i}$" for i in range(10)]  # Fallback
        else:
            symbols = metadata.get("symbols", [])

        # Format True Equation column
        true_expr_raw = method_results[0]["true_expression"]
        true_latex = get_latex_with_symbols(true_expr_raw, symbols)
        true_col = f"**{eq_name}**<br>{true_latex}"

        # Format other columns with <br> for multiple methods
        guessed_col_parts = []
        r2_col_parts = []
        ned_col_parts = []

        # Sort by method name for consistency
        for res in sorted(method_results, key=lambda x: x["method"]):
            method = res["method"]
            guess_raw = res["discovered_expression"]
            guess_latex = get_latex_with_symbols(guess_raw, symbols)

            guessed_col_parts.append(f"**{method}**: {guess_latex}")
            r2_col_parts.append(f"**{method}**: {res['test_r2']:.4f}")
            ned_col_parts.append(f"**{method}**: {res['ned']:.3f}")

        table.add_row(
            [
                true_col,
                "<br>".join(guessed_col_parts),
                "<br>".join(r2_col_parts),
                "<br>".join(ned_col_parts),
            ]
        )

    # 5. Save to markdown file
    with open(output_path, "w") as f:
        f.write("# Performance Comparison - 'Easy' Equations\n\n")
        f.write(table.get_string())
        f.write("\n")

    print(f"Successfully generated markdown table: {output_path}")


if __name__ == "__main__":
    main()
