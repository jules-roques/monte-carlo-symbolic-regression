from mcsr.utils.dataloader import load_true_sympy_expressions

true_pickles = load_true_sympy_expressions("easy")
for name, expr in true_pickles.items():
    print(name, expr)
