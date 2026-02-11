"""Utility functions for the simulation services."""

from typing import TYPE_CHECKING, Any

from flow360.component.simulation.framework.entity_expansion_utils import (
    get_registry_from_params,
)
from flow360.component.simulation.framework.entity_selector import _process_selectors
from flow360.component.simulation.framework.entity_utils import (
    walk_object_tree_with_cycle_detection,
)

_MIRRORED_ENTITY_TYPE_NAMES = ("MirroredSurface", "MirroredGeometryBodyGroup")

if TYPE_CHECKING:
    from flow360.component.simulation.simulation_params import SimulationParams


def strip_implicit_edge_split_layers_inplace(params: "SimulationParams", params_dict: dict) -> dict:
    """
    Remove implicitly injected `edge_split_layers` from serialized params.
    This extra and specific function was added due to a change in schema during lifecycle of a release (uncommon)

    Why not use `exclude_unset` or `exclude_defaults` globally during `model_dump()`?
    - `exclude_unset` strips many unrelated defaulted fields and can affect downstream workflows.
    - `exclude_defaults` also strips explicitly user-set values that equal the default.
    """
    meshing = getattr(params, "meshing", None)
    defaults = getattr(meshing, "defaults", None)
    if defaults is None:
        return params_dict

    if "edge_split_layers" in defaults.model_fields_set:
        # Keep explicit user setting (including explicit value equal to default).
        return params_dict

    meshing_dict = params_dict.get("meshing")
    if not isinstance(meshing_dict, dict):
        return params_dict

    defaults_dict = meshing_dict.get("defaults")
    if not isinstance(defaults_dict, dict):
        return params_dict

    defaults_dict.pop("edge_split_layers", None)
    return params_dict


def strip_selector_matches_and_broken_entities_inplace(params) -> Any:
    """
    In stored_entities:
    1. Remove entities matched by selectors from each EntityList's stored_entities, in place.
    2. Remove registry-backed entities that are not valid for the current params registry (broken/foreign),
       in place. This primarily targets mirrored entities, but also protects against stale persistent entities.

    Rationale:
    - Keep user hand-picked entities distinguishable for the UI by stripping items that are
      implied by EntitySelector expansion.
    - Operate on the deserialized params object to avoid dict-level selector handling.

    Behavior:
    - For every EntityList-like object that has a non-empty `selectors` list, compute the set
      of entities implied by those selectors over the registry, and remove those implied entities
      from its `stored_entities` list.
    - For every EntityList-like object, check if it contains any mirror entities that no longer
      have a corresponding geometry entity, and remove them from the list.

    Returns the same object for chaining.
    """
    asset_cache = getattr(params, "private_attribute_asset_cache", None)
    project_entity_info = getattr(asset_cache, "project_entity_info", None)
    if asset_cache is None or project_entity_info is None:
        # Compatibility with some unit tests.
        return params

    selector_cache: dict = {}
    registry = get_registry_from_params(params)

    valid_mirrored_registry_keys = {
        (entity.private_attribute_entity_type_name, entity.private_attribute_id)
        for entity in registry.find_by_type_name(list(_MIRRORED_ENTITY_TYPE_NAMES))
    }

    def _extract_entity_key(item) -> tuple:
        """Extract stable key from entity object."""
        entity_type = getattr(item, "private_attribute_entity_type_name", None)
        entity_id = getattr(item, "private_attribute_id", None)
        return (entity_type, entity_id)

    def _matched_keyset_for_selectors(selectors_list: list) -> set[tuple]:
        additions_by_class, _ = _process_selectors(
            registry,
            selectors_list,
            selector_cache,
        )
        keys: set = set()
        for items in additions_by_class.values():
            for entity in items:
                keys.add(_extract_entity_key(entity))
        return keys

    def _strip_selector_matches_and_broken_entities(obj) -> bool:
        """
        Strip entities matched by selectors from EntityList's stored_entities, then drop mirrored entities
        that are not present in the current params registry.
        Returns True to continue traversing, False to stop.
        """
        selectors_list = getattr(obj, "selectors", None)
        stored_entities = getattr(obj, "stored_entities", None)
        if not isinstance(stored_entities, list):
            return True

        if not stored_entities:
            obj.stored_entities = []
            return False

        updated_entities = stored_entities

        if isinstance(selectors_list, list) and selectors_list:
            matched_keys = _matched_keyset_for_selectors(selectors_list)
            if matched_keys:
                updated_entities = [
                    item
                    for item in updated_entities
                    if _extract_entity_key(item) not in matched_keys
                ]
            if not updated_entities:
                obj.stored_entities = []
                return False

        cleaned_entities = []
        for item in updated_entities:
            entity_type, entity_id = _extract_entity_key(item)
            # Keep non-entity objects (or entities without stable keys) untouched.
            if entity_type is None or entity_id is None:
                cleaned_entities.append(item)
                continue

            if entity_type in _MIRRORED_ENTITY_TYPE_NAMES:
                if (entity_type, entity_id) in valid_mirrored_registry_keys:
                    cleaned_entities.append(item)
                continue

            cleaned_entities.append(item)

        obj.stored_entities = cleaned_entities

        return False  # Don't traverse into EntityList internals
        # Continue traversing

    walk_object_tree_with_cycle_detection(
        params, _strip_selector_matches_and_broken_entities, check_dict=False
    )
    return params
