"""Core blueprint functionality."""

from .context import EvaluationContext, ReturnValue
from .expressions import (
    BinOpNode,
    CallModelNode,
    ConstantNode,
    ExpressionNode,
    ExpressionNodeType,
    ListCompNode,
    ListNode,
    NameNode,
    RangeCallNode,
    SubscriptNode,
    TupleNode,
    UnaryOpNode,
)
from .function import FunctionNode
from .generator import expr_to_code, model_to_function, stmt_to_code
from .parser import function_to_model
from .statements import (
    AssignNode,
    AugAssignNode,
    ForLoopNode,
    IfElseNode,
    ReturnNode,
    StatementNode,
    StatementNodeType,
    TupleUnpackNode,
)
from .types import Evaluable


def _model_rebuild() -> None:
    """Update forward references in the correct order."""
    namespace = {
        # Expression types
        "NameNode": NameNode,
        "ConstantNode": ConstantNode,
        "BinOpNode": BinOpNode,
        "RangeCallNode": RangeCallNode,
        "CallModelNode": CallModelNode,
        "TupleNode": TupleNode,
        "ListNode": ListNode,
        "ListCompNode": ListCompNode,
        "SubscriptNode": SubscriptNode,
        "UnaryOpNode": UnaryOpNode,
        "ExpressionNodeType": ExpressionNodeType,
        # Statement types
        "AssignNode": AssignNode,
        "AugAssignNode": AugAssignNode,
        "IfElseNode": IfElseNode,
        "ForLoopNode": ForLoopNode,
        "ReturnNode": ReturnNode,
        "TupleUnpackNode": TupleUnpackNode,
        "StatementNodeType": StatementNodeType,
        # Function type
        "FunctionNode": FunctionNode,
    }

    # First update expression classes that only depend on ExpressionType
    BinOpNode.model_rebuild(_types_namespace=namespace)
    RangeCallNode.model_rebuild(_types_namespace=namespace)
    CallModelNode.model_rebuild(_types_namespace=namespace)
    TupleNode.model_rebuild(_types_namespace=namespace)
    ListNode.model_rebuild(_types_namespace=namespace)
    ListCompNode.model_rebuild(_types_namespace=namespace)
    SubscriptNode.model_rebuild(_types_namespace=namespace)

    # Then update statement classes that depend on both types
    AssignNode.model_rebuild(_types_namespace=namespace)
    AugAssignNode.model_rebuild(_types_namespace=namespace)
    IfElseNode.model_rebuild(_types_namespace=namespace)
    ForLoopNode.model_rebuild(_types_namespace=namespace)
    ReturnNode.model_rebuild(_types_namespace=namespace)
    TupleUnpackNode.model_rebuild(_types_namespace=namespace)

    # Finally update Function class
    FunctionNode.model_rebuild(_types_namespace=namespace)


# Update forward references
_model_rebuild()


__all__ = [
    "ExpressionNode",
    "NameNode",
    "ConstantNode",
    "BinOpNode",
    "RangeCallNode",
    "CallModelNode",
    "TupleNode",
    "ListNode",
    "ListCompNode",
    "ExpressionNodeType",
    "StatementNode",
    "AssignNode",
    "AugAssignNode",
    "IfElseNode",
    "ForLoopNode",
    "ReturnNode",
    "TupleUnpackNode",
    "StatementNodeType",
    "FunctionNode",
    "EvaluationContext",
    "ReturnValue",
    "Evaluable",
    "expr_to_code",
    "stmt_to_code",
    "model_to_function",
    "function_to_model",
]
