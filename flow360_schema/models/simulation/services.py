"""Schema-owned simulation services."""

import copy
import types
from typing import Any, Literal, Union, get_args, get_origin

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.entity.entity_registry import EntityRegistry
from flow360_schema.framework.physical_dimensions import Length
from flow360_schema.framework.unit_system import _UNIT_SYSTEMS, UnitSystem
from flow360_schema.framework.validation.context import unit_system_manager
from flow360_schema.models.entity_info import EntityInfoModel, GeometryEntityInfo, parse_entity_info_model
from flow360_schema.models.entity_info import (
    merge_geometry_entity_info as merge_geometry_entity_info_obj,
)
from flow360_schema.models.entities.surface_entities import Surface
from flow360_schema.models.reference_geometry import ReferenceGeometry
from flow360_schema.models.simulation.meshing_param.params import MeshingParams
from flow360_schema.models.simulation.meshing_param.volume_params import AutomatedFarfield
from flow360_schema.models.simulation.models.surface_models import Freestream, Wall
from flow360_schema.models.simulation.operating_condition.operating_condition import (
    AerospaceCondition,
)
from flow360_schema.models.simulation.outputs.outputs import SurfaceOutput
from flow360_schema.models.simulation.simulation_params import SimulationParams
from flow360_schema.models.simulation.units import validate_length
from flow360_schema.models.simulation.validation.validation_service import (
    ALL,
    ValidationCalledBy,
    _determine_validation_level,
    _insert_forward_compatibility_notice,
    _intersect_validation_levels,
    _normalize_union_branch_error_location,
    _populate_error_context,
    _sanitize_stack_trace,
    _traverse_error_location,
    clean_unrelated_setting_from_params_dict,
    clear_context,
    handle_generic_exception,
    initialize_variable_space,
    validate_error_locations,
    validate_model,
)


def _parse_root_item_type_from_simulation_json(*, param_as_dict: dict):
    """[External] Deduct the root item entity type from simulation.json."""
    try:
        entity_info_type = param_as_dict["private_attribute_asset_cache"]["project_entity_info"]["type_name"]
        if entity_info_type == "GeometryEntityInfo":
            return "Geometry"
        if entity_info_type == "SurfaceMeshEntityInfo":
            return "SurfaceMesh"
        if entity_info_type == "VolumeMeshEntityInfo":
            return "VolumeMesh"
        raise ValueError(f"[INTERNAL] Invalid type of the entity info found: {entity_info_type}")
    except KeyError as error:
        raise ValueError("[INTERNAL] Failed to get the root item from the simulation.json!!!") from error


def merge_geometry_entity_info(draft_param_as_dict: dict, geometry_dependencies_param_as_dict: list[dict]):
    """
    Merge the geometry entity info from geometry dependencies into the draft simulation param dict.

    Parameters
    ----------
    draft_param_as_dict : dict
        The draft simulation parameters dictionary.
    geometry_dependencies_param_as_dict : list of dict
        The list of geometry dependencies simulation parameters dictionaries.

    Returns
    -------
    dict
        The merged geometry entity info dictionary.
    """
    # pylint:disable = protected-access
    draft_param_as_dict, _ = SimulationParams._update_param_dict(draft_param_as_dict)
    draft_param_entity_info_dict = draft_param_as_dict.get("private_attribute_asset_cache", {}).get(
        "project_entity_info", {}
    )
    if draft_param_entity_info_dict.get("type_name") != "GeometryEntityInfo":
        return draft_param_entity_info_dict

    current_entity_info = GeometryEntityInfo.deserialize(draft_param_entity_info_dict)

    entity_info_components = []
    for geometry_param_as_dict in geometry_dependencies_param_as_dict:
        geometry_param_as_dict, _ = SimulationParams._update_param_dict(geometry_param_as_dict)
        dependency_entity_info_dict = geometry_param_as_dict.get("private_attribute_asset_cache", {}).get(
            "project_entity_info", {}
        )
        if dependency_entity_info_dict.get("type_name") != "GeometryEntityInfo":
            continue
        entity_info_components.append(GeometryEntityInfo.deserialize(dependency_entity_info_dict))

    merged_entity_info = merge_geometry_entity_info_obj(
        current_entity_info=current_entity_info,
        entity_info_components=entity_info_components,
    )
    return merged_entity_info.model_dump(mode="json", exclude_none=True)


def update_simulation_json(*, params_as_dict: dict, target_python_api_version: str):
    """
    Run the SimulationParams updater to update to the specified version.
    """
    errors = []
    updated_params_as_dict: dict | None = None
    try:
        updated_params_as_dict, input_has_higher_version = SimulationParams._update_param_dict(
            params_as_dict,
            target_python_api_version,
        )
        if input_has_higher_version:
            raise ValueError(
                f"[Internal] API misuse. Input version "
                f"({SimulationParams._get_version_from_dict(model_dict=params_as_dict)}) is higher than "
                f"requested target version ({target_python_api_version})."
            )
    except (Flow360ValueError, ValueError, KeyError) as error:
        errors.append(str(error))
    return updated_params_as_dict, errors


def init_unit_system(unit_system_name: str) -> UnitSystem:
    """Return a unit system by name for default-parameter construction."""
    unit_system = _UNIT_SYSTEMS.get(unit_system_name)
    if unit_system is None:
        raise ValueError(f"Unknown unit system: {unit_system_name!r}. Available: {list(_UNIT_SYSTEMS)}")
    if unit_system_manager.current is not None:
        raise RuntimeError(
            "Services cannot be used inside unit system context. " f"Used: {unit_system_manager.current.system_repr()}."
        )
    return unit_system


def _store_project_length_unit(
    project_length_unit: Length.Float64 | None,
    params: SimulationParams,
) -> SimulationParams:
    if project_length_unit is None:
        return params
    # Store the length unit so downstream services and pipelines can reuse it.
    params.private_attribute_asset_cache._force_set_attr("project_length_unit", project_length_unit)
    return params


def _get_default_reference_geometry(length_unit: Length.Float64) -> ReferenceGeometry:
    return ReferenceGeometry(
        area=1 * length_unit**2,
        moment_center=(0, 0, 0) * length_unit,
        moment_length=(1, 1, 1) * length_unit,
    )


def get_default_params(
    unit_system_name: str,
    length_unit,
    root_item_type: Literal["Geometry", "SurfaceMesh", "VolumeMesh"],
) -> dict:
    """Return default simulation parameters for the requested root-item workflow."""
    unit_system = init_unit_system(unit_system_name)
    project_length_unit = validate_length(length_unit)
    with unit_system:
        reference_geometry = _get_default_reference_geometry(project_length_unit)
        operating_condition = AerospaceCondition(velocity_magnitude=0.1)
        surface_output = SurfaceOutput(
            name="Surface output",
            entities=[Surface(name="*")],
            output_fields=["Cp", "yPlus", "Cf", "CfVec"],
        )

    if root_item_type == "VolumeMesh":
        with unit_system:
            params = SimulationParams(
                reference_geometry=reference_geometry,
                operating_condition=operating_condition,
                models=[
                    Wall(
                        name="Wall",
                        surfaces=[Surface(name="placeholder1")],
                        roughness_height=0 * project_length_unit,
                    ),
                    Freestream(name="Freestream", surfaces=[Surface(name="placeholder2")]),
                ],
                outputs=[surface_output],
            )
        params.models[0].entities.stored_entities = []
        params.models[1].entities.stored_entities = []
        params = _store_project_length_unit(project_length_unit, params)
        return params.model_dump(
            mode="json",
            exclude_none=True,
            exclude={
                "operating_condition": {"velocity_magnitude": True},
                "private_attribute_asset_cache": {"registry": True},
                "meshing": True,
            },
        )

    if root_item_type not in ("Geometry", "SurfaceMesh"):
        raise ValueError(
            f"Unknown root item type: {root_item_type}. " "Expected one of Geometry or SurfaceMesh or VolumeMesh"
        )

    automated_farfield = AutomatedFarfield(name="Farfield")
    with unit_system:
        params = SimulationParams(
            reference_geometry=reference_geometry,
            meshing=MeshingParams(volume_zones=[automated_farfield]),
            operating_condition=operating_condition,
            models=[
                Wall(
                    name="Wall",
                    surfaces=[Surface(name="*")],
                    roughness_height=0 * project_length_unit,
                ),
                Freestream(name="Freestream", surfaces=[automated_farfield.farfield]),
            ],
            outputs=[surface_output],
        )
    params = _store_project_length_unit(project_length_unit, params)
    return params.model_dump(
        mode="json",
        exclude_none=True,
        exclude={
            "operating_condition": {"velocity_magnitude": True},
            "private_attribute_asset_cache": {"registry": True},
            "meshing": {"defaults": {"edge_split_layers": True}},
        },
    )


def _get_draft_entity_type_names() -> set[str]:
    """Extract entity type names from DraftEntityTypes in entity_info.py."""
    type_names = set()

    draft_field = EntityInfoModel.model_fields["draft_entities"]
    draft_annotation = draft_field.annotation
    inner_type = get_args(draft_annotation)[0]
    union_args = get_args(inner_type)
    if union_args:
        actual_union = union_args[0]
        if get_origin(actual_union) is Union or isinstance(actual_union, types.UnionType):
            for cls in get_args(actual_union):
                type_names.add(cls.__name__)

    return type_names


DRAFT_ENTITY_TYPE_NAMES = _get_draft_entity_type_names()


def _replace_entities_by_type_and_name(
    template_dict: dict,
    target_registry: EntityRegistry,
) -> tuple[dict, list[dict[str, Any]]]:
    """
    Traverse template_dict and replace stored_entities with matching entities from target_registry.
    """
    warnings = []

    entity_lookup: dict[tuple[str, str], dict] = {}
    for entity_list in target_registry.internal_registry.values():
        for entity in entity_list:
            key = (entity.private_attribute_entity_type_name, entity.name)
            entity_lookup[key] = entity.model_dump(mode="json", exclude_none=True)

    def process_stored_entities(stored_entities: list) -> list:
        new_stored = []
        for entity_dict in stored_entities:
            entity_type_name = entity_dict.get("private_attribute_entity_type_name")
            entity_name = entity_dict.get("name")

            if entity_type_name in DRAFT_ENTITY_TYPE_NAMES:
                new_stored.append(entity_dict)
                continue

            key = (entity_type_name, entity_name)
            if key in entity_lookup:
                new_stored.append(entity_lookup[key])
            else:
                warnings.append(
                    {
                        "type": "entity_not_found",
                        "loc": ["stored_entities"],
                        "msg": f"Entity '{entity_name}' (type: {entity_type_name}) not found in target entity info",
                        "ctx": {},
                    }
                )
        return new_stored

    def traverse_and_replace(obj):
        if isinstance(obj, dict):
            if "stored_entities" in obj and isinstance(obj["stored_entities"], list):
                obj["stored_entities"] = process_stored_entities(obj["stored_entities"])
            for value in obj.values():
                traverse_and_replace(value)
        elif isinstance(obj, list):
            for item in obj:
                traverse_and_replace(item)

    traverse_and_replace(template_dict)
    return template_dict, warnings


def apply_simulation_setting_to_entity_info(
    simulation_setting_dict: dict,
    entity_info_dict: dict,
):
    """
    Apply simulation settings from one project to another project with different entity info.
    """
    simulation_setting_dict = SimulationParams._sanitize_params_dict(simulation_setting_dict)
    simulation_setting_dict, _ = SimulationParams._update_param_dict(simulation_setting_dict)
    entity_info_dict = SimulationParams._sanitize_params_dict(entity_info_dict)
    entity_info_dict, _ = SimulationParams._update_param_dict(entity_info_dict)

    target_entity_info_data = entity_info_dict.get("private_attribute_asset_cache", {}).get("project_entity_info", {})
    source_entity_info = simulation_setting_dict.get("private_attribute_asset_cache", {}).get("project_entity_info", {})

    merged_entity_info = copy.deepcopy(target_entity_info_data)
    merged_entity_info["draft_entities"] = source_entity_info.get("draft_entities", [])

    if target_entity_info_data.get("type_name") == "GeometryEntityInfo":
        tag_to_attr_names = {
            "face_group_tag": "face_attribute_names",
            "body_group_tag": "body_attribute_names",
            "edge_group_tag": "edge_attribute_names",
        }
        for tag_key, attr_names_key in tag_to_attr_names.items():
            source_tag = source_entity_info.get(tag_key)
            if source_tag is not None:
                target_attr_names = target_entity_info_data.get(attr_names_key, [])
                if source_tag in target_attr_names:
                    merged_entity_info[tag_key] = source_tag

    merged_entity_info_obj = parse_entity_info_model(merged_entity_info)
    target_registry = EntityRegistry.from_entity_info(merged_entity_info_obj)

    simulation_setting_dict["private_attribute_asset_cache"]["project_entity_info"] = merged_entity_info

    simulation_setting_dict, entity_warnings = _replace_entities_by_type_and_name(
        simulation_setting_dict,
        target_registry,
    )

    root_item_type = _parse_root_item_type_from_simulation_json(param_as_dict=simulation_setting_dict)
    _, errors, validation_warnings = validate_model(
        params_as_dict=copy.deepcopy(simulation_setting_dict),
        validated_by=ValidationCalledBy.SERVICE,
        root_item_type=root_item_type,
        validation_level=ALL,
    )

    all_warnings = entity_warnings + validation_warnings
    return simulation_setting_dict, errors, all_warnings


__all__ = [
    "ALL",
    "ValidationCalledBy",
    "_determine_validation_level",
    "_get_default_reference_geometry",
    "_insert_forward_compatibility_notice",
    "_intersect_validation_levels",
    "_normalize_union_branch_error_location",
    "_parse_root_item_type_from_simulation_json",
    "_populate_error_context",
    "_sanitize_stack_trace",
    "_store_project_length_unit",
    "_traverse_error_location",
    "clean_unrelated_setting_from_params_dict",
    "clear_context",
    "get_default_params",
    "handle_generic_exception",
    "init_unit_system",
    "initialize_variable_space",
    "apply_simulation_setting_to_entity_info",
    "merge_geometry_entity_info",
    "update_simulation_json",
    "validate_error_locations",
    "validate_model",
]
