"""Re-import relay: entity registry classes now live in flow360-schema."""

# pylint: disable=unused-import
# Re-import relay: all entity registry classes now live in flow360-schema
from flow360_schema.framework.entity.entity_registry import (  # noqa: F401
    EntityRegistry,
    EntityRegistryView,
    SnappyBodyRegistry,
    StringIndexableList,
)
