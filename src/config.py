import ast
import operator

# Allowed operators
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.USub: operator.neg,
}


def conf_expr(expr: str, variables: dict):
    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Only numeric constants allowed")

        elif isinstance(node, ast.BinOp):
            return OPS[type(node.op)](_eval(node.left), _eval(node.right))

        elif isinstance(node, ast.UnaryOp):
            return OPS[type(node.op)](_eval(node.operand))

        elif isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"Unknown variable '{node.id}'")
            return variables[node.id]

        else:
            raise TypeError(f"Unsupported expression: {ast.dump(node)}")

    tree = ast.parse(expr, mode="eval")
    return _eval(tree)
