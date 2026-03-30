"""Flow360BaseModel — re-exports from flow360-schema.

All SDK methods (copy, preprocess, file I/O, help, hash) now live in
flow360_schema.framework.base_model.Flow360BaseModel.
"""

# pylint: disable=unused-import
from flow360_schema.framework.base_model import Flow360BaseModel  # noqa: F401
from flow360_schema.framework.base_model_utils import (  # noqa: F401
    _preprocess_any_model,
    _preprocess_nested_list,
    need_conversion,
)
