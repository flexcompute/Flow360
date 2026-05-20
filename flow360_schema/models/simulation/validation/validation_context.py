"""Simulation-specific validation context and helper state."""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from pydantic import TypeAdapter

from flow360_schema.framework.bounding_box import BoundingBoxType
from flow360_schema.framework.physical_dimensions import Length
from flow360_schema.framework.validation.context import (
    ALL,
    CASE,
    DeserializationContext,
    SURFACE_MESH,
    VOLUME_MESH,
    ValidationContext,
    add_validation_warning,
    get_validation_info,
    get_validation_levels,
)
from flow360_schema.framework.validation.fields import CaseField, ConditionalField, ContextField
from flow360_schema.framework.validation.validators import (
    context_validator,
    contextual_field_validator,
    contextual_model_validator,
)

if TYPE_CHECKING:
    from flow360_schema.framework.entity.entity_base import EntityBase


def get_value_with_path(param_as_dict: dict, path: list[str]):
    """
    Get the value from the dictionary with the given path.
    Return None if the path is not found.
    """

    value = param_as_dict
    for key in path:
        value = value.get(key, None)
        if value is None:
            return None
    return value


class TimeSteppingType(Enum):
    """
    Enum for time stepping type.

    Attributes
    ----------
    STEADY : str
        Represents a steady simulation.
    UNSTEADY : str
        Represents an unsteady simulation.
    UNSET : str
        The time stepping is unset.
    """

    STEADY = "Steady"
    UNSTEADY = "Unsteady"
    UNSET = "Unset"


class FeatureUsageInfo:
    """
    Model that provides the information for each individual feature usage.
    """

    __slots__ = [
        "turbulence_model_type",
        "transition_model_type",
        "rotation_zone_count",
        "bet_disk_count",
    ]

    def __init__(self, param_as_dict: dict):
        self.turbulence_model_type = None
        self.transition_model_type = None
        self.rotation_zone_count = 0
        self.bet_disk_count = 0

        if "models" in param_as_dict and param_as_dict["models"]:
            for model in param_as_dict["models"]:
                if model["type"] == "Fluid":
                    self.turbulence_model_type = model.get("turbulence_model_solver", {}).get("type_name", None)
                    self.transition_model_type = model.get("transition_model_solver", {}).get("type_name", None)

                if model["type"] == "Rotation":
                    self.rotation_zone_count += 1

                if model["type"] == "BETDisk":
                    self.bet_disk_count += 1


class ParamsValidationInfo:
    """
    Model that provides the information for each individual validator that is out of their scope.

    This can be considered as a partially validated `SimulationParams`.
    """

    __slots__ = [
        "farfield_method",
        "farfield_domain_type",
        "is_beta_mesher",
        "use_geometry_AI",
        "use_snappy",
        "using_liquid_as_material",
        "time_stepping",
        "feature_usage",
        "referenced_expressions",
        "project_length_unit",
        "global_bounding_box",
        "planar_face_tolerance",
        "output_dict",
        "physics_model_dict",
        "half_model_symmetry_plane_center_y",
        "quasi_3d_symmetry_planes_center_y",
        "entity_transformation_detected",
        "to_be_generated_custom_volumes",
        "farfield_enclosed_entities",
        "root_asset_type",
        "_entity_info",
        "_entity_registry",
        "_selector_cache",
    ]

    @classmethod
    def _get_farfield_method_(cls, param_as_dict: dict):
        meshing = param_as_dict.get("meshing")
        if meshing is None:
            return None

        if meshing["type_name"] == "MeshingParams":
            volume_zones = meshing.get("volume_zones")
        else:
            volume_zones = meshing.get("zones")
        if volume_zones:
            has_custom_zones = False
            for zone in volume_zones:
                if zone["type"] == "AutomatedFarfield":
                    return zone["method"]
                if zone["type"] == "UserDefinedFarfield":
                    return "user-defined"
                if zone["type"] == "WindTunnelFarfield":
                    return "wind-tunnel"
                if zone["type"] in ("CustomZones", "SeedpointVolume"):
                    has_custom_zones = True
            if has_custom_zones:
                return "user-defined"

        return None

    @classmethod
    def _get_farfield_domain_type_(cls, param_as_dict: dict):
        try:
            if param_as_dict["meshing"]:
                volume_zones = param_as_dict["meshing"]["volume_zones"]
            else:
                return None
        except KeyError:
            return None
        if not volume_zones:
            return None
        for zone in volume_zones:
            if zone.get("type") in (
                "AutomatedFarfield",
                "UserDefinedFarfield",
                "WindTunnelFarfield",
            ):
                return zone.get("domain_type")
        return None

    @classmethod
    def _get_using_liquid_as_material_(cls, param_as_dict: dict):
        try:
            if param_as_dict["operating_condition"]:
                return param_as_dict["operating_condition"]["type_name"] == "LiquidOperatingCondition"
        except KeyError:
            return False
        return False

    @classmethod
    def _get_is_beta_mesher_(cls, param_as_dict: dict):
        try:
            return param_as_dict["private_attribute_asset_cache"]["use_inhouse_mesher"]
        except KeyError:
            return False

    @classmethod
    def _get_use_snappy_(cls, param_as_dict: dict):
        if param_as_dict.get("meshing") and param_as_dict["meshing"].get("surface_meshing"):
            return param_as_dict["meshing"]["surface_meshing"]["type_name"] == "SnappySurfaceMeshingParams"

        return False

    @classmethod
    def _get_use_geometry_AI_(cls, param_as_dict: dict):
        try:
            return param_as_dict["private_attribute_asset_cache"]["use_geometry_AI"]
        except KeyError:
            return False

    @classmethod
    def _get_time_stepping_(cls, param_as_dict: dict):
        try:
            if param_as_dict["time_stepping"]["type_name"] == "Unsteady":
                return TimeSteppingType.UNSTEADY
            return TimeSteppingType.STEADY
        except KeyError:
            return TimeSteppingType.UNSET

    @classmethod
    def _get_feature_usage_info(cls, param_as_dict: dict):
        return FeatureUsageInfo(param_as_dict=param_as_dict)

    @classmethod
    def _get_project_length_unit_(cls, param_as_dict: dict):
        try:
            project_length_unit_dict = param_as_dict["private_attribute_asset_cache"]["project_length_unit"]
            if project_length_unit_dict:
                adapter = TypeAdapter(Length.PositiveFloat64)
                with DeserializationContext():
                    return adapter.validate_python(project_length_unit_dict)
            return None
        except KeyError:
            return None

    @classmethod
    def _get_global_bounding_box(cls, param_as_dict: dict):
        global_bounding_box = get_value_with_path(
            param_as_dict,
            ["private_attribute_asset_cache", "project_entity_info", "global_bounding_box"],
        )
        if global_bounding_box:
            return TypeAdapter(BoundingBoxType).validate_python(global_bounding_box)
        return None

    @classmethod
    def _get_planar_face_tolerance(cls, param_as_dict: dict):
        planar_face_tolerance = None
        if "meshing" in param_as_dict and param_as_dict["meshing"]:
            if param_as_dict["meshing"]["type_name"] == "MeshingParams":
                planar_face_tolerance = get_value_with_path(
                    param_as_dict, ["meshing", "defaults", "planar_face_tolerance"]
                )
            else:
                planar_face_tolerance = get_value_with_path(
                    param_as_dict, ["meshing", "volume_meshing", "planar_face_tolerance"]
                )
        return planar_face_tolerance

    @classmethod
    def _get_root_asset_type(cls, param_as_dict: dict):
        try:
            project_entity_info = param_as_dict["private_attribute_asset_cache"]["project_entity_info"]
        except KeyError:
            return None
        if project_entity_info is None:
            return None
        type_name = (
            project_entity_info.get("type_name")
            if isinstance(project_entity_info, dict)
            else getattr(project_entity_info, "type_name", None)
        )
        if type_name == "GeometryEntityInfo":
            return "geometry"
        if type_name == "SurfaceMeshEntityInfo":
            return "surface_mesh"
        if type_name == "VolumeMeshEntityInfo":
            return "volume_mesh"
        return None

    @classmethod
    def _get_half_model_symmetry_plane_center_y(cls, param_as_dict: dict):
        ghost_entities = get_value_with_path(
            param_as_dict,
            ["private_attribute_asset_cache", "project_entity_info", "ghost_entities"],
        )
        if not ghost_entities:
            return None
        for ghost_entity in ghost_entities:
            if ghost_entity["private_attribute_entity_type_name"] != "GhostCircularPlane":
                continue
            if ghost_entity["name"] == "symmetric":
                return ghost_entity["center"][1]
        return None

    @classmethod
    def _get_quasi_3d_symmetry_planes_center_y(cls, param_as_dict: dict):
        ghost_entities = get_value_with_path(
            param_as_dict,
            ["private_attribute_asset_cache", "project_entity_info", "ghost_entities"],
        )
        if not ghost_entities:
            return None
        symmetric_1_center_y = None
        symmetric_2_center_y = None
        for ghost_entity in ghost_entities:
            if ghost_entity["private_attribute_entity_type_name"] != "GhostCircularPlane":
                continue
            if ghost_entity["name"] == "symmetric-1":
                symmetric_1_center_y = ghost_entity["center"][1]
            if ghost_entity["name"] == "symmetric-2":
                symmetric_2_center_y = ghost_entity["center"][1]
        if symmetric_1_center_y is None or symmetric_2_center_y is None:
            return None
        return (symmetric_1_center_y, symmetric_2_center_y)

    @classmethod
    def _get_entity_transformation_detected(cls, param_as_dict: dict):
        coordinate_system_status_dict = get_value_with_path(
            param_as_dict, ["private_attribute_asset_cache", "coordinate_system_status"]
        )
        if coordinate_system_status_dict and coordinate_system_status_dict.get("assignments"):
            return True

        mirror_status_dict = get_value_with_path(param_as_dict, ["private_attribute_asset_cache", "mirror_status"])
        if mirror_status_dict and (
            mirror_status_dict.get("mirrored_geometry_body_groups") or mirror_status_dict.get("mirrored_surfaces")
        ):
            return True

        return False

    def _get_boundary_surface_ids(self, entity) -> set:
        """Extract boundary surface IDs from a CustomVolume entity, expanding selectors if needed."""
        if entity.private_attribute_entity_type_name != "CustomVolume":
            return set()
        bounding_entities = getattr(entity, "bounding_entities", None)
        if not bounding_entities:
            return set()
        expanded_entities = self.expand_entity_list(bounding_entities)
        return {expanded_entity.private_attribute_id for expanded_entity in expanded_entities}

    def _get_to_be_generated_custom_volumes(self, param_as_dict: dict):
        volume_zones = get_value_with_path(param_as_dict, ["meshing", "volume_zones"])

        if not volume_zones:
            volume_zones = get_value_with_path(param_as_dict, ["meshing", "zones"])

        if not volume_zones:
            return {}

        custom_volume_info = {}
        for zone in volume_zones:
            if zone.get("type") != "CustomZones":
                continue
            enforce_tetrahedra = zone.get("element_type") == "tetrahedra"
            stored_entities = zone.get("entities", {}).get("stored_entities", [])

            for entity in stored_entities:
                if entity.private_attribute_entity_type_name not in (
                    "CustomVolume",
                    "SeedpointVolume",
                ):
                    continue
                custom_volume_info[entity.name] = {
                    "enforce_tetrahedra": enforce_tetrahedra,
                    "boundary_surface_ids": self._get_boundary_surface_ids(entity),
                }
        return custom_volume_info

    def _get_farfield_enclosed_entities(self, param_as_dict: dict) -> dict[str, str]:
        """Extract enclosed surface {id: name} from farfield zones."""
        volume_zones = get_value_with_path(param_as_dict, ["meshing", "volume_zones"])
        if not volume_zones:
            volume_zones = get_value_with_path(param_as_dict, ["meshing", "zones"])
        if not volume_zones:
            return {}

        for zone in volume_zones:
            if zone.get("type") not in (
                "AutomatedFarfield",
                "UserDefinedFarfield",
                "WindTunnelFarfield",
            ):
                continue
            enclosed = zone.get("enclosed_entities")
            if not enclosed:
                return {}
            enclosed_obj = SimpleNamespace(
                stored_entities=enclosed.get("stored_entities", []),
                selectors=enclosed.get("selectors"),
            )
            surfaces = self.expand_entity_list(enclosed_obj)
            return {surface.private_attribute_id: surface.name for surface in surfaces}

        return {}

    @property
    def farfield_cv_dual_belonging_ids(self) -> set[str]:
        """Surface IDs that appear in both farfield enclosed_entities and CustomVolume bounding_entities."""
        if not self.farfield_enclosed_entities:
            return set()
        enclosed_ids = set(self.farfield_enclosed_entities.keys())
        custom_volume_boundary_ids: set[str] = set()
        for custom_volume_info in self.to_be_generated_custom_volumes.values():
            custom_volume_boundary_ids |= custom_volume_info.get("boundary_surface_ids", set())
        return enclosed_ids & custom_volume_boundary_ids

    def __init__(self, param_as_dict: dict, referenced_expressions: list):
        self.farfield_method = self._get_farfield_method_(param_as_dict=param_as_dict)
        self.farfield_domain_type = self._get_farfield_domain_type_(param_as_dict=param_as_dict)
        self.is_beta_mesher = self._get_is_beta_mesher_(param_as_dict=param_as_dict)
        self.use_geometry_AI = self._get_use_geometry_AI_(param_as_dict=param_as_dict)
        self.use_snappy = self._get_use_snappy_(param_as_dict=param_as_dict)
        self.using_liquid_as_material = self._get_using_liquid_as_material_(param_as_dict=param_as_dict)
        self.time_stepping = self._get_time_stepping_(param_as_dict=param_as_dict)
        self.feature_usage = self._get_feature_usage_info(param_as_dict=param_as_dict)
        self.referenced_expressions = referenced_expressions
        self.project_length_unit = self._get_project_length_unit_(param_as_dict=param_as_dict)
        self.global_bounding_box = self._get_global_bounding_box(param_as_dict=param_as_dict)
        self.planar_face_tolerance = self._get_planar_face_tolerance(param_as_dict=param_as_dict)
        self.output_dict = None
        self.physics_model_dict = None
        self.half_model_symmetry_plane_center_y = self._get_half_model_symmetry_plane_center_y(
            param_as_dict=param_as_dict
        )
        self.quasi_3d_symmetry_planes_center_y = self._get_quasi_3d_symmetry_planes_center_y(
            param_as_dict=param_as_dict
        )
        self.entity_transformation_detected = self._get_entity_transformation_detected(param_as_dict=param_as_dict)
        self.root_asset_type = self._get_root_asset_type(param_as_dict=param_as_dict)

        self._entity_info, self._entity_registry = self._build_entity_info_and_registry(param_as_dict)
        self._selector_cache = None

        self.to_be_generated_custom_volumes = self._get_to_be_generated_custom_volumes(param_as_dict=param_as_dict)
        self.farfield_enclosed_entities = self._get_farfield_enclosed_entities(param_as_dict=param_as_dict)

    def will_generate_forced_symmetry_plane(self) -> bool:
        """Check if the forced symmetry plane will be generated."""
        return (
            self.use_geometry_AI
            and self.is_beta_mesher
            and self.farfield_domain_type in ("half_body_positive_y", "half_body_negative_y")
        )

    @classmethod
    def _build_entity_info_and_registry(cls, param_as_dict: dict):
        """Build entity_info and entity_registry from param_as_dict."""
        from flow360_schema.framework.entity.entity_expansion_utils import (
            get_entity_info_and_registry_from_dict,
        )

        try:
            return get_entity_info_and_registry_from_dict(param_as_dict)
        except (KeyError, ValueError):
            return None, None

    def _ensure_selector_cache(self):
        """Lazily initialize selector cache."""
        if self._selector_cache is None:
            self._selector_cache = {}

    def get_entity_info(self):
        """Get the deserialized entity_info."""
        return self._entity_info

    def get_entity_registry(self):
        """Get the entity_registry."""
        return self._entity_registry

    def expand_entity_list(self, entity_list: Any) -> list[EntityBase]:
        """
        Expand selectors and return the combined list of entities.

        Concrete ``EntityList`` instances use the resolved expansion path so selector
        additions are filtered against the declared EntityList types. Duck-typed
        objects that only provide ``stored_entities`` / ``selectors`` fall back to
        raw selector expansion without EntityList-specific filtering.
        """
        stored_entities = list(getattr(entity_list, "stored_entities", []) or [])
        raw_selectors = list(getattr(entity_list, "selectors", []) or [])

        if not raw_selectors or self._entity_registry is None:
            return stored_entities

        self._ensure_selector_cache()
        from flow360_schema.framework.entity.entity_list import EntityList
        from flow360_schema.framework.entity.entity_selector import (
            expand_entity_list_selectors,
            resolve_entity_list_selectors,
        )

        if isinstance(entity_list, EntityList):
            # With filtering based on EntityList definition.
            return resolve_entity_list_selectors(
                self._entity_registry,
                entity_list,
                selector_cache=self._selector_cache,
                merge_mode="merge",
            )

        return expand_entity_list_selectors(
            self._entity_registry,
            entity_list,
            selector_cache=self._selector_cache,
            merge_mode="merge",
        )


__all__ = [
    "ALL",
    "CASE",
    "SURFACE_MESH",
    "VOLUME_MESH",
    "CaseField",
    "ConditionalField",
    "ContextField",
    "FeatureUsageInfo",
    "ParamsValidationInfo",
    "TimeSteppingType",
    "ValidationContext",
    "add_validation_warning",
    "context_validator",
    "contextual_field_validator",
    "contextual_model_validator",
    "get_validation_info",
    "get_validation_levels",
    "get_value_with_path",
]
