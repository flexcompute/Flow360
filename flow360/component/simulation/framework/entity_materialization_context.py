"""Re-import relay: entity_materialization_context moved to flow360_schema."""

# pylint: disable=unused-import
from flow360_schema.framework.entity.entity_materialization_context import (  # noqa: F401
    EntityMaterializationContext,
    get_entity_builder,
    get_entity_cache,
    get_entity_registry,
)
