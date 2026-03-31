"""
Entity selector models — re-import relay.

All definitions have been migrated to flow360_schema.framework.entity.entity_selector.
This module re-exports them for backward compatibility.
"""

# pylint: disable=unused-import
from flow360_schema.framework.entity.entity_selector import (  # noqa: F401
    BodyGroupSelector,
    EdgeSelector,
    EntityNode,
    EntitySelector,
    Predicate,
    SurfaceSelector,
    TargetClass,
    VolumeSelector,
    _process_selectors,
    collect_and_tokenize_selectors_in_place,
    expand_entity_list_selectors,
    expand_entity_list_selectors_in_place,
)

# Re-export transitive imports that external code depends on via this module
from flow360_schema.framework.entity.entity_utils import (  # noqa: F401
    compile_glob_cached,
)
