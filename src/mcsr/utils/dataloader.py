import pickle
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sympy
from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download

from mcsr.tree.expression import Expression

BASE_PATH = "yoshitomo-matsubara/srsd-feynman"


class SRSDLoader:
    CHUNK_SIZES = {
        "train": 8000,
        "validation": 1000,
        "test": 1000,
    }

    def __init__(
        self, difficulty: str = "easy", splits: Iterable[str] = ("train", "validation")
    ) -> None:
        self.difficulty = difficulty.lower()
        self.splits = tuple(splits)
        self.repo_id = f"{BASE_PATH}_{self.difficulty}"

        self.equation_names = self._fetch_equation_names()
        self._cache: dict[str, dict[str, Any]] = {}
        self._build_cache()

    def _fetch_equation_names(self) -> list[str]:
        info_ds = load_dataset(self.repo_id, data_files="supp_info.json", split="train")
        return list(info_ds.features.keys())

    def _chunk_split(
        self, dataset_dict: DatasetDict, split_name: str
    ) -> list[np.ndarray]:
        raw_text = dataset_dict[split_name]["text"]
        chunk_size = self.CHUNK_SIZES[split_name]

        chunks = []
        for start_idx in range(0, len(raw_text), chunk_size):
            end_idx = start_idx + chunk_size
            lines = raw_text[start_idx:end_idx]
            matrix = np.array([np.fromstring(line, sep=" ") for line in lines])
            chunks.append(matrix)

        return chunks

    def _build_cache(self) -> None:
        dataset_dict = load_dataset(self.repo_id)
        split_data = {
            split: self._chunk_split(dataset_dict, split) for split in self.splits
        }

        for idx, name in enumerate(self.equation_names):
            equation_data = {"name": name}

            for split in self.splits:
                matrix = split_data[split][idx]
                equation_data[split] = (matrix[:, :-1], matrix[:, -1])

            self._cache[name] = equation_data

    def __getitem__(self, key: int | str) -> dict[str, Any]:
        if isinstance(key, str):
            return self._cache[key]
        return self._cache[self.equation_names[key]]

    def __len__(self) -> int:
        return len(self.equation_names)

    def __iter__(self):
        for name in self.equation_names:
            yield self._cache[name]


def load_true_sympy_expressions(difficulty: str) -> dict[str, sympy.Expr]:
    repo_id = f"{BASE_PATH}_{difficulty}"
    local_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        allow_patterns="true_eq/*.pkl",
    )

    true_eq_dir = Path(local_dir) / "true_eq"

    equations = {}
    for file_path in true_eq_dir.glob("*.pkl"):
        with open(file_path, "rb") as f:
            equations[file_path.stem] = pickle.load(f)

    return equations


def load_pickled_expressions(path: Path) -> dict[str, Expression]:
    equations = {}
    for file_path in path.glob("*.pkl"):
        with open(file_path, "rb") as f:
            equations[file_path.stem] = pickle.load(f)
    return equations
