"""Core blueprint functionality."""

from .context import EvaluationContext, ReturnValue
from .expressions import (
    BinOp,
    CallModel,
    Constant,
    Expression,
    ExpressionType,
    List,
    ListComp,
    Name,
    RangeCall,
    Subscript,
    Tuple,
)
from .function import Function
from .generator import expr_to_code, model_to_function, stmt_to_code
from .parser import function_to_model
from .statements import (
    Assign,
    AugAssign,
    ForLoop,
    IfElse,
    Return,
    Statement,
    StatementType,
    TupleUnpack,
)
from .types import Evaluable, TargetSyntax


def _model_rebuild() -> None:
    """Update forward references in the correct order."""
    namespace = {
        # Expression types
        "Name": Name,
        "Constant": Constant,
        "BinOp": BinOp,
        "RangeCall": RangeCall,
        "CallModel": CallModel,
        "Tuple": Tuple,
        "List": List,
        "ListComp": ListComp,
        "Subscript": Subscript,
        "ExpressionType": ExpressionType,
        # Statement types
        "Assign": Assign,
        "AugAssign": AugAssign,
        "IfElse": IfElse,
        "ForLoop": ForLoop,
        "Return": Return,
        "TupleUnpack": TupleUnpack,
        "StatementType": StatementType,
        # Function type
        "Function": Function,
    }

    # First update expression classes that only depend on ExpressionType
    BinOp.model_rebuild(_types_namespace=namespace)
    RangeCall.model_rebuild(_types_namespace=namespace)
    CallModel.model_rebuild(_types_namespace=namespace)
    Tuple.model_rebuild(_types_namespace=namespace)
    List.model_rebuild(_types_namespace=namespace)
    ListComp.model_rebuild(_types_namespace=namespace)
    Subscript.model_rebuild(_types_namespace=namespace)

    # Then update statement classes that depend on both types
    Assign.model_rebuild(_types_namespace=namespace)
    AugAssign.model_rebuild(_types_namespace=namespace)
    IfElse.model_rebuild(_types_namespace=namespace)
    ForLoop.model_rebuild(_types_namespace=namespace)
    Return.model_rebuild(_types_namespace=namespace)
    TupleUnpack.model_rebuild(_types_namespace=namespace)

    # Finally update Function class
    Function.model_rebuild(_types_namespace=namespace)


# Update forward references
_model_rebuild()


__all__ = [
    "Expression",
    "Name",
    "Constant",
    "BinOp",
    "RangeCall",
    "CallModel",
    "Tuple",
    "List",
    "ListComp",
    "ExpressionType",
    "Statement",
    "Assign",
    "AugAssign",
    "IfElse",
    "ForLoop",
    "Return",
    "TupleUnpack",
    "StatementType",
    "Function",
    "EvaluationContext",
    "ReturnValue",
    "Evaluable",
    "expr_to_code",
    "stmt_to_code",
    "model_to_function",
    "function_to_model",
]
