"""String expression type for simulation framework."""

import ast

from pydantic.functional_validators import AfterValidator
from typing_extensions import Annotated

from flow360.component.utils import process_expressions

# pylint: disable=fixme
# TODO: Add units to expression?
# TODO: Add variable existence check?
StringExpression = Annotated[str, AfterValidator(process_expressions)]


def validate_angle_expression_of_t_seconds(expr: str):
    """
    [TEMPORARY SOLUTION TO ENABLE DIMENSIONED EXPRESSION FOR ROTATION]
    Validate that:
      1. The only allowed names in the expression are those in ALLOWED_NAMES.
      2. Every occurrence of 't_seconds' is used only in a multiplicative context.
        Allowed forms:
            - as part of a multiplication (e.g. 0.58*t_seconds),
            - or as a standalone with a unary minus (i.e. -t_seconds) provided that
              this unary minus is not immediately part of an addition/subtraction.
        This check ensure that the later _preprocess() results in correct non-dimensioned expression

        E.g.
        "2*sin(1*t_seconds+2)"
            = 2rad*sin(1/s*t_seconds+2rad)
            = 2*sin(1/s*(t*seconds_per_flow360_time)+2rad) (After calling _preprocess())
        (v) = "2*sin(1*(t*seconds_per_flow360_time)+2)" (seen by solver)

        whereas

        "2*sin(1*(t_seconds+2))"
            = 2rad*sin(1/s*(t_seconds+2second))
            != 2*sin(1/s*((t*seconds_per_flow360_time)+2second)) (After calling _preprocess())
        (x) = "2*sin(1*((t*seconds_per_flow360_time)+2))" (seen by solver)

    Returns a list of error messages (empty if valid).
    """
    ALLOWED_NAMES = {  # pylint:disable=invalid-name
        "t_seconds",
        "t",
        "pi",
        "sin",
        "cos",
        "tan",
        "atan",
        "min",
        "max",
        "pow",
        "powf",
        "log",
        "exp",
        "sqrt",
        "abs",
        "ceil",
        "floor",
    }

    errors = []

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return [f"Syntax error in expression `{e.text}`: {e.msg}."]

    def visit(node, ancestors=None):
        if ancestors is None:
            ancestors = []
        # Check for Name nodes: allow only ALLOWED_NAMES.
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES:
                errors.append(f"Unexpected variable `{node.id}` found.")
            # If the name is t_seconds, check its context.
            if node.id == "t_seconds":
                if not is_valid_t_seconds_usage(ancestors):
                    errors.append(
                        "t_seconds must be used as a multiplicative factor, "
                        "not directly added/subtracted with a number."
                    )
        # Recurse into children while keeping track of ancestors.
        for child in ast.iter_child_nodes(node):
            visit(child, ancestors + [node])

    def is_valid_t_seconds_usage(ancestors):
        """
        Check the usage of t_seconds based on its ancestors.
        We consider the immediate parent (and grandparent if needed) to decide.

        Allowed if:
         - Immediate parent is a BinOp with operator Mult.
         - Immediate parent is a UnaryOp with USub and its own parent is not a BinOp with Add or Sub.
           (This allows a standalone -t_seconds or as the argument of a function.)
        """
        if not ancestors:
            # t_seconds is at the top-level; not allowed because it isn't scaled by multiplication.
            return False
        parent = ancestors[-1]
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Mult):
            return True
        if isinstance(parent, ast.UnaryOp) and isinstance(parent.op, ast.USub):
            # Check the grandparent if it exists.
            if len(ancestors) >= 2:
                grandparent = ancestors[-2]
                if isinstance(grandparent, ast.BinOp) and isinstance(
                    grandparent.op, (ast.Add, ast.Sub)
                ):
                    return False
            # Otherwise, standalone -t_seconds or inside a function call is acceptable.
            return True
        return False

    visit(tree)
    return errors
