from mcsr.tree.expression import Expression
from mcsr.tree.grammar import (
    ADD,
    CONSTANTS,
    COS,
    MUL,
    SIN,
    SQRT,
    SQUARE,
    make_variable,
)
from mcsr.utils.converters import expression_to_sympy

expr_1 = Expression(
    [
        ADD,
        MUL,
        SIN,
        COS,
        SQRT,
        SQUARE,
        make_variable(0),
        CONSTANTS[0],
        CONSTANTS[1],
    ]
)

expr_2 = Expression(
    [
        ADD,
        MUL,
        COS,
        SIN,
        SQRT,
        SQUARE,
        make_variable(0),
        CONSTANTS[0],
        CONSTANTS[2],
    ]
)


sy_expr_1 = expression_to_sympy(expr_1)
sy_expr_2 = expression_to_sympy(expr_2)
