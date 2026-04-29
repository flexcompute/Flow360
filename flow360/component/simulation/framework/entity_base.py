"""Re-import relay: entity base classes now live in flow360-schema."""

# pylint: disable=unused-import
# Re-import relay: all entity classes now live in flow360-schema
from flow360_schema.framework.entity.entity_base import EntityBase  # noqa: F401
from flow360_schema.framework.entity.entity_list import (  # noqa: F401
    EntityList,
    _CombinedMeta,
    _EntityListMeta,
)
