"""Re-import relay: multi-constructor model base now lives in flow360-schema."""

# pylint: disable=unused-import
from flow360_schema.framework.multi_constructor_model_base import (  # noqa: F401
    MultiConstructorBaseModel,
    get_class_by_name,
    get_class_method,
    model_custom_constructor_parser,
    parse_model_dict,
)
