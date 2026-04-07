"""Re-import relay: entity utilities now live in flow360-schema."""

# pylint: disable=unused-import
from flow360_schema.framework.entity.entity_utils import (  # noqa: F401
    DEFAULT_NOT_MERGED_TYPES,
    compile_glob_cached,
    deduplicate_entities,
    generate_uuid,
    get_entity_key,
    get_entity_type,
    walk_object_tree_with_cycle_detection,
)
