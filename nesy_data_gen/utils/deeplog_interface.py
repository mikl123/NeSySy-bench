from deeplog import (
    SymTensor,
    parse_formula_to_module,
    parse_symbol,
    reshape,
    with_structure,
)


def probability_input(n_outputs):
    """SymTensor of structure-tagged probability symbols, i.e. (x1 _ probability) ... (xn _ probability)."""
    return SymTensor(
        [
            with_structure(parse_symbol(f"x{i+1}"), "probability")
            for i in range(n_outputs)
        ]
    )


def to_deeplog_str(constraint, n_outputs):
    """Convert a single CNF constraint to deeplog string representation."""
    clause_strs = []
    for clause in constraint._clauses:
        literals = []
        for lit in clause:
            var = abs(lit)
            if lit > 0:
                literals.append(f"(x{var}_boolean)")
            else:
                literals.append(f"(not x{var}_boolean)")
        clause_strs.append(f"({' or '.join(literals)})")
    formula = " and ".join(clause_strs)
    variables = [f"x{i+1}" for i in range(n_outputs)]
    formula = f"expectation({','.join(variables)}) : {formula}"
    return formula


def constraint_to_deeplog_module(constraint, n_outputs):
    deeplog_formula = to_deeplog_str(constraint, n_outputs)
    constraint_module = parse_formula_to_module(deeplog_formula)
    expected_input = probability_input(n_outputs)
    constraint_module = reshape(constraint_module, expected_input)
    return constraint_module


if __name__ == "__main__":
    deeplog_formula = "expectation(x1,x2,x3,x4,x5) : ((not x1_boolean) or (x2_boolean) or (x5_boolean))"
    n_outputs = 5
    constraint_module = parse_formula_to_module(deeplog_formula)
    expected_input = probability_input(n_outputs)
    constraint_module = reshape(constraint_module, expected_input)
    print(constraint_module)
