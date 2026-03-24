from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


import numpy as np


from mcsr.tree.expression import Expression


from dotenv import load_dotenv

load_dotenv()


@dataclass
class SRSDProblem:
    name: str
    class_name: str
    num_vars: int
    formula: str
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    target_name: str = "y"
    variable_names: Optional[list[str]] = None
    true_expression: Optional[Expression] = None

    def __post_init__(self):
        if self.variable_names is None:
            self.variable_names = [f"x{i}" for i in range(self.num_vars)]


def generate_problem(
    eq_name: str,
) -> SRSDProblem:
    """Generate a single SRSD problem with train/test split directly from HF datasets."""
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    import pickle
    from mcsr.tree.expression import Expression, sympy_to_expression

    # Load specific data files for this equation to avoid concatenating everything
    data_files = {
        "train": f"train/{eq_name}.txt",
        "val": f"val/{eq_name}.txt",
        "test": f"test/{eq_name}.txt",
    }

    ds = load_dataset(
        "yoshitomo-matsubara/srsd-feynman_easy",
        data_files=data_files,
    )

    def _parse_split(split: list[str]) -> np.ndarray:
        return np.array([[float(v) for v in line.split()] for line in split])

    train_data = _parse_split(ds["train"]["text"])
    val_data = _parse_split(ds["val"]["text"])
    test_data = _parse_split(ds["test"]["text"])

    num_vars = train_data.shape[1] - 1

    # Load supplemental info for variable names
    target_name = "y"
    variable_names = [f"x{i}" for i in range(num_vars)]

    try:
        import json
        import re

        def _clean_latex(s: str) -> str:
            # Remove dollar signs
            s = s.replace("$", "")
            # Remove \text{...} but keep the content
            s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
            # Remove backslashes
            s = s.replace("\\", "")
            # Remove curly braces
            s = s.replace("{", "").replace("}", "")
            return s

        supp_info_path = hf_hub_download(
            repo_id="yoshitomo-matsubara/srsd-feynman_easy",
            filename="supp_info.json",
            repo_type="dataset",
        )
        with open(supp_info_path, "r") as f:
            supp_info = json.load(f)

        if eq_name in supp_info:
            symbols = supp_info[eq_name].get("symbols", [])
            if symbols:
                target_name = _clean_latex(symbols[0])
                # The first symbol is target, then independent variables
                variable_names = [
                    _clean_latex(s) for s in symbols[1 : 1 + num_vars]
                ]
    except Exception as e:
        print(f"Could not load supplemental info for {eq_name}: {e}")

    try:
        pkl_path = hf_hub_download(
            repo_id="yoshitomo-matsubara/srsd-feynman_easy",
            filename=f"true_eq/{eq_name}.pkl",
            repo_type="dataset",
        )
        with open(pkl_path, "rb") as f:
            sy_expr = pickle.load(f)
        true_expr = sympy_to_expression(sy_expr, variable_names=variable_names)
    except Exception as e:
        print(f"Could not load GT eq for {eq_name}: {e}")
        true_expr = None

    return SRSDProblem(
        name=eq_name,
        class_name=eq_name,  # Map class_name to eq_name for compatibility
        num_vars=num_vars,
        formula="Unknown (loaded from HF)",
        train_x=train_data[:, :-1],
        train_y=train_data[:, -1],
        val_x=val_data[:, :-1],
        val_y=val_data[:, -1],
        test_x=test_data[:, :-1],
        test_y=test_data[:, -1],
        target_name=target_name,
        variable_names=variable_names,
        true_expression=true_expr,
    )


def load_srsd_easy_problems(
    equation_filter: Optional[list[str]] = None,
) -> list[SRSDProblem]:
    """Load all (or selected) SRSD-easy benchmark problems dynamically from HF."""
    from huggingface_hub import list_repo_files

    repo_files = list_repo_files(
        "yoshitomo-matsubara/srsd-feynman_easy", repo_type="dataset"
    )

    equation_names = sorted(
        [
            f.split("/")[1].replace(".txt", "")
            for f in repo_files
            if f.startswith("train/") and f.endswith(".txt")
        ]
    )

    problems: list[SRSDProblem] = []
    for eq_name in equation_names:

        if equation_filter and eq_name not in equation_filter:
            continue

        problems.append(generate_problem(eq_name))

    return problems
