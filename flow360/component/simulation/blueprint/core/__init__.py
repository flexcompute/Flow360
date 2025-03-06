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
    Tuple,
)
from .function import Function
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


def _update_forward_refs() -> None:
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
    BinOp.update_forward_refs(**namespace)
    RangeCall.update_forward_refs(**namespace)
    CallModel.update_forward_refs(**namespace)
    Tuple.update_forward_refs(**namespace)
    List.update_forward_refs(**namespace)
    ListComp.update_forward_refs(**namespace)

    # Then update statement classes that depend on both types
    Assign.update_forward_refs(**namespace)
    AugAssign.update_forward_refs(**namespace)
    IfElse.update_forward_refs(**namespace)
    ForLoop.update_forward_refs(**namespace)
    Return.update_forward_refs(**namespace)
    TupleUnpack.update_forward_refs(**namespace)

    # Finally update Function class
    Function.update_forward_refs(**namespace)


# Update forward references
_update_forward_refs()


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
]
