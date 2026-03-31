from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Atom(ABC):
    name: str

    @property
    @abstractmethod
    def arity(self) -> int:
        pass

    @property
    @abstractmethod
    def operator(self) -> Callable:
        pass


@dataclass(frozen=True)
class Terminal(Atom):
    @property
    def arity(self) -> int:
        return 0


@dataclass(frozen=True)
class Constant(Terminal):
    value: float

    @property
    def operator(self) -> Callable:
        return lambda x: np.full(x.shape[0], self.value)

    def __repr__(self) -> str:
        return f"{self.value}"


@dataclass(frozen=True)
class Variable(Terminal):
    var_index: int

    @property
    def operator(self) -> Callable:
        return lambda x: x[:, self.var_index]

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class UnaryOperator(Atom):
    func: Callable

    @property
    def arity(self) -> int:
        return 1

    @property
    def operator(self) -> Callable:
        return self.func

    def __repr__(self) -> str:
        return self.name


@dataclass(frozen=True)
class BinaryOperator(Atom):
    func: Callable

    @property
    def arity(self) -> int:
        return 2

    @property
    def operator(self) -> Callable:
        return self.func

    def __repr__(self) -> str:
        return self.name
