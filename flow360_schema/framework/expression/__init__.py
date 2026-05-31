"""Expression engine: AST parsing, evaluation, and code generation.

Note: parser and generator are NOT imported at module level to keep
JSON schema generation lightweight (they pull in ast/inspect).
Access them via:
    from flow360_schema.framework.expression.engine.parser import expr_to_model, function_to_model
    from flow360_schema.framework.expression.engine.generator import expr_to_code, stmt_to_code, model_to_function
"""

from flow360_schema.framework.expression.dependency_graph import DependencyGraph
from flow360_schema.framework.expression.engine.ast_nodes import (
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
from flow360_schema.framework.expression.engine.eval_context import EvaluationContext, ReturnValue
from flow360_schema.framework.expression.engine.function import FunctionNode
from flow360_schema.framework.expression.engine.operators import BINARY_OPERATORS, UNARY_OPERATORS
from flow360_schema.framework.expression.engine.resolver import CallableResolver
from flow360_schema.framework.expression.engine.statements import (
    AssignNode,
    AugAssignNode,
    ForLoopNode,
    IfElseNode,
    ReturnNode,
    StatementNode,
    StatementNodeType,
    TupleUnpackNode,
)
from flow360_schema.framework.expression.engine.types import Evaluable, TargetSyntax
from flow360_schema.framework.expression.utils import (
    StringExpression,
    convert_caret_to_power,
    convert_if_else,
    convert_legacy_names,
    is_runtime_expression,
    normalize_string_expression,
    process_expressions,
    validate_angle_expression_of_t_seconds,
)
from flow360_schema.framework.expression.value_or_expression import (
    AnyNumericType,
    SerializedValueOrExpression,
    UnytArray,
    UnytQuantity,
    ValueOrExpression,
)
from flow360_schema.framework.expression.variable import (
    Expression,
    RedeclaringVariableError,
    SolverVariable,
    UserVariable,
    Variable,
    VariableContextInfo,
    batch_get_user_variable_units,
    compute_surface_integral_unit,
    get_input_value_dimensions,
    get_input_value_length,
    get_referenced_expressions_and_user_variables,
    get_user_variable,
    remove_user_variable,
    restore_variable_space,
    save_user_variables,
    show_user_variables,
    solver_variable_to_user_variable,
)


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
    UnaryOpNode.model_rebuild(_types_namespace=namespace)
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


_model_rebuild()

# Patch VariableContextInfo.value with the real ValueOrExpression type now that
# all expression modules are fully loaded (no circular import at this point).
from flow360_schema.framework.expression.variable import _patch_variable_context_info_value_field  # noqa: E402

_patch_variable_context_info_value_field()


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
    "SubscriptNode",
    "UnaryOpNode",
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
    "TargetSyntax",
    "CallableResolver",
    "DependencyGraph",
    "BINARY_OPERATORS",
    "UNARY_OPERATORS",
    "SerializedValueOrExpression",
    "ValueOrExpression",
    "UnytQuantity",
    "UnytArray",
    "AnyNumericType",
    "Variable",
    "UserVariable",
    "SolverVariable",
    "Expression",
    "RedeclaringVariableError",
    "VariableContextInfo",
    "batch_get_user_variable_units",
    "compute_surface_integral_unit",
    "get_input_value_dimensions",
    "get_input_value_length",
    "get_referenced_expressions_and_user_variables",
    "get_user_variable",
    "remove_user_variable",
    "restore_variable_space",
    "save_user_variables",
    "show_user_variables",
    "solver_variable_to_user_variable",
    "convert_if_else",
    "convert_caret_to_power",
    "convert_legacy_names",
    "is_runtime_expression",
    "normalize_string_expression",
    "process_expressions",
    "StringExpression",
    "validate_angle_expression_of_t_seconds",
]
