"""Tests for expression evaluation."""

import numpy as np

from mcsr.tree.grammar import (
    ADD,
    SUB,
    MUL,
    DIV,
    SIN,
    COS,
    LOG,
    SQRT,
    EXP,
    make_variable,
)
from mcsr.tree.atom import Atom, Constant
from mcsr.tree.expression import Expression


def _make_data(x0: list, x1: list | None = None) -> np.ndarray:
    if x1 is None:
        return np.array(x0).reshape(-1, 1)
    return np.column_stack([x0, x1])


class TestEvaluateExpression:
    def test_single_variable(self):
        x0 = make_variable(0)
        data = _make_data([1.0, 2.0, 3.0])
        result = Expression([x0]).evaluate(data)
        np.testing.assert_allclose(result, [1.0, 2.0, 3.0])

    def test_single_constant(self):
        c = Constant(name="const", value=5.0)
        data = _make_data([1.0, 2.0])
        result = Expression([c]).evaluate(data)
        np.testing.assert_allclose(result, [5.0, 5.0])

    def test_addition(self):
        # + x0 x1 → x0 + x1
        x0, x1 = make_variable(0), make_variable(1)
        data = _make_data([1.0, 2.0], [3.0, 4.0])
        result = Expression([ADD, x0, x1]).evaluate(data)
        np.testing.assert_allclose(result, [4.0, 6.0])

    def test_subtraction(self):
        x0, x1 = make_variable(0), make_variable(1)
        data = _make_data([5.0, 10.0], [3.0, 4.0])
        result = Expression([SUB, x0, x1]).evaluate(data)
        np.testing.assert_allclose(result, [2.0, 6.0])

    def test_multiplication(self):
        x0, x1 = make_variable(0), make_variable(1)
        data = _make_data([2.0, 3.0], [4.0, 5.0])
        result = Expression([MUL, x0, x1]).evaluate(data)
        np.testing.assert_allclose(result, [8.0, 15.0])

    def test_division(self):
        x0, x1 = make_variable(0), make_variable(1)
        data = _make_data([6.0, 10.0], [3.0, 2.0])
        result = Expression([DIV, x0, x1]).evaluate(data)
        np.testing.assert_allclose(result, [2.0, 5.0])

    def test_division_by_zero(self):
        x0, x1 = make_variable(0), make_variable(1)
        data = _make_data([6.0], [0.0])
        result = Expression([DIV, x0, x1]).evaluate(data)
        assert np.isnan(result[0])

    def test_nested_expression(self):
        # * (+ x0 x1) x0 → (x0 + x1) * x0
        x0, x1 = make_variable(0), make_variable(1)
        data = _make_data([2.0, 3.0], [3.0, 4.0])
        result = Expression([MUL, ADD, x0, x1, x0]).evaluate(data)
        np.testing.assert_allclose(result, [10.0, 21.0])

    def test_sin(self):
        x0 = make_variable(0)
        data = _make_data([0.0, np.pi / 2])
        result = Expression([SIN, x0]).evaluate(data)
        np.testing.assert_allclose(result, [0.0, 1.0], atol=1e-10)

    def test_cos(self):
        x0 = make_variable(0)
        data = _make_data([0.0, np.pi])
        result = Expression([COS, x0]).evaluate(data)
        np.testing.assert_allclose(result, [1.0, -1.0], atol=1e-10)

    def test_log_negative(self):
        x0 = make_variable(0)
        data = _make_data([-1.0])
        result = Expression([LOG, x0]).evaluate(data)
        assert np.isnan(result[0])

    def test_sqrt_negative(self):
        x0 = make_variable(0)
        data = _make_data([-4.0])
        result = Expression([SQRT, x0]).evaluate(data)
        assert np.isnan(result[0])


class TestExpressionToString:
    def test_simple_add(self):
        x0, x1 = make_variable(0), make_variable(1)
        s = str(Expression([ADD, x0, x1]))
        assert s == "(x0 + x1)"

    def test_nested(self):
        x0, x1 = make_variable(0), make_variable(1)
        s = str(Expression([MUL, ADD, x0, x1, x0]))
        assert s == "((x0 + x1) * x0)"

    def test_unary(self):
        x0 = make_variable(0)
        s = str(Expression([SIN, x0]))
        assert s == "sin(x0)"
