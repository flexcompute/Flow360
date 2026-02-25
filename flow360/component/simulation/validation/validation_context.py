# pylint: disable=too-many-lines
"""
Module for validation context handling in the simulation component of Flow360.

This module provides context management for validation levels and specialized field
definitions for conditional validation scenarios.

Features
--------
- This module allows for defining conditionally (context-based) required fields,
  such as fields required only for specific scenarios like surface mesh or volume mesh.
- It supports running validation only for specific scenarios (surface mesh, volume mesh, or case),
  allowing for targeted validation flows based on the current context.
- This module does NOT ignore validation errors; instead, it enriches errors with context
  information, enabling downstream processes to filter and interpret errors based on scenario-specific requirements.
"""

import contextvars
import inspect
from enum import Enum
from functools import wraps
from typing import Any, Callable, List, Literal, Union

import pydantic as pd
from pydantic import Field, TypeAdapter

from flow360.component.simulation.unit_system import LengthType
from flow360.component.simulation.utils import BoundingBoxType

SURFACE_MESH = "SurfaceMesh"
VOLUME_MESH = "VolumeMesh"
CASE = "Case"
# when running validation with ALL, it will report errors happing in all scenarios in one validation pass
ALL = "All"


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
    Enum for time stepping type

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

    # pylint: disable=too-few-public-methods
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
                    self.turbulence_model_type = model.get("turbulence_model_solver", {}).get(
                        "type_name", None
                    )
                    self.transition_model_type = model.get("transition_model_solver", {}).get(
                        "type_name", None
                    )

                if model["type"] == "Rotation":
                    self.rotation_zone_count += 1

                if model["type"] == "BETDisk":
                    self.bet_disk_count += 1


_validation_level_ctx = contextvars.ContextVar("validation_levels", default=None)
_validation_info_ctx = contextvars.ContextVar("validation_info", default=None)
_validation_warnings_ctx = contextvars.ContextVar("validation_warnings", default=None)


class ParamsValidationInfo:  # pylint:disable=too-few-public-methods,too-many-instance-attributes
    """
    Model that provides the information for each individual validator that is out of their scope.

    This can be considered as a partially validated `SimulationParams`.

    - Why this model?

    -> Some validators needs information from other parts of the SimulationParams that is impossible to
    get due to the information is out the scope of the validator. We can use a model validator on the
    SimulationParams instead but then the validator implementation needs to represent the
    structure of the SimulationParams and future feature change needs to be aware of this to make sure
    the validation is performed.
    E.g: All `Surface` entities needs to check if it will be deleted by the mesher depending
    on mesher option (auto or quasi 3d).
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
        "farfield_enclosed_surfaces",
        "root_asset_type",
        # Entity expansion support
        "_entity_info",  # Owns the entities (keeps them alive), initialized eagerly
        "_entity_registry",  # References entities from _entity_info, initialized eagerly
        "_selector_cache",  # Lazy, populated as selectors are expanded
    ]

    @classmethod
    def _get_farfield_method_(cls, param_as_dict: dict):
        meshing = param_as_dict.get("meshing")
        modular = False
        if meshing is None:
            # No meshing info.
            return None

        if meshing["type_name"] == "MeshingParams":
            volume_zones = meshing.get("volume_zones")
        else:
            volume_zones = meshing.get("zones")
            modular = True
        if volume_zones:
            for zone in volume_zones:
                if zone["type"] == "AutomatedFarfield":
                    return zone["method"]
                if zone["type"] == "UserDefinedFarfield":
                    return "user-defined"
                if zone["type"] == "WindTunnelFarfield":
                    return "wind-tunnel"
                if (
                    zone["type"]
                    in [
                        "CustomZones",
                        "SeedpointVolume",
                    ]
                    and modular
                ):
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
                return (
                    param_as_dict["operating_condition"]["type_name"] == "LiquidOperatingCondition"
                )
        except KeyError:
            # No liquid operating condition info.
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
            return (
                param_as_dict["meshing"]["surface_meshing"]["type_name"]
                == "SnappySurfaceMeshingParams"
            )

        return False

    @classmethod
    def _get_use_geometry_AI_(cls, param_as_dict: dict):  # pylint:disable=invalid-name
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
        # 1. Turbulence model type
        # 2. Transition model type
        # 3. Usage of Rotation zone
        # 4. Usage of BETDisk
        return FeatureUsageInfo(param_as_dict=param_as_dict)

    @classmethod
    def _get_project_length_unit_(cls, param_as_dict: dict):
        try:
            project_length_unit_dict = param_as_dict["private_attribute_asset_cache"][
                "project_length_unit"
            ]
            if project_length_unit_dict:
                # pylint: disable=no-member
                return LengthType.validate(project_length_unit_dict)
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
            # pylint: disable=no-member
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
        """
        Returns root asset type based on project_entity_info.type_name
        geometry -> GeometryEntityInfo
        surface_mesh -> SurfaceMeshEntityInfo
        volume_mesh -> VolumeMeshEntityInfo
        """
        try:
            pei = param_as_dict["private_attribute_asset_cache"]["project_entity_info"]
        except KeyError:
            return None
        if pei is None:
            return None
        type_name = (
            pei.get("type_name") if isinstance(pei, dict) else getattr(pei, "type_name", None)
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
            if not ghost_entity["private_attribute_entity_type_name"] == "GhostCircularPlane":
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
            if not ghost_entity["private_attribute_entity_type_name"] == "GhostCircularPlane":
                continue
            if ghost_entity["name"] == "symmetric-1":
                symmetric_1_center_y = ghost_entity["center"][1]
            if ghost_entity["name"] == "symmetric-2":
                symmetric_2_center_y = ghost_entity["center"][1]
        if symmetric_1_center_y is None or symmetric_2_center_y is None:
            return None
        return (symmetric_1_center_y, symmetric_2_center_y)

    @classmethod
    def _get_entity_transformation_detected(
        cls, param_as_dict: dict
    ):  # pylint:disable=invalid-name
        """
        Get the flag indicating if at least one body was transformed or mirrored.
        This is used to skip boundary deletion/assignment checks since once translated
        the bounding box as well as the boundary existence is no longer valid.
        """
        # 1. Check for coordinate system transformations
        coordinate_system_status_dict = get_value_with_path(
            param_as_dict, ["private_attribute_asset_cache", "coordinate_system_status"]
        )
        if coordinate_system_status_dict:
            # Check if assignments list is non-empty
            if coordinate_system_status_dict.get("assignments"):
                return True

        # 2. Check for mirroring
        mirror_status_dict = get_value_with_path(
            param_as_dict, ["private_attribute_asset_cache", "mirror_status"]
        )
        if mirror_status_dict:
            # Check if either mirrored groups or surfaces list is non-empty
            if mirror_status_dict.get("mirrored_geometry_body_groups") or mirror_status_dict.get(
                "mirrored_surfaces"
            ):
                return True

        return False

    def _get_boundary_surface_ids(self, entity) -> set:
        """Extract boundary surface IDs from a CustomVolume entity, expanding selectors if needed."""
        if entity.private_attribute_entity_type_name != "CustomVolume":
            return set()
        boundaries = getattr(entity, "boundaries", None)
        if not boundaries:
            return set()
        # Expand selectors to get all boundary surfaces
        expanded_boundaries = self.expand_entity_list(boundaries)
        return {surface.private_attribute_id for surface in expanded_boundaries}

    def _get_to_be_generated_custom_volumes(self, param_as_dict: dict):
        volume_zones = get_value_with_path(
            param_as_dict,
            ["meshing", "volume_zones"],
        )

        if not volume_zones:
            volume_zones = get_value_with_path(
                param_as_dict,
                ["meshing", "zones"],
            )

        if not volume_zones:
            return {}

        # Return a mapping: { custom_volume_name: {enforce_tetrahedra, boundary_surface_ids} }
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

    def _get_farfield_enclosed_surfaces(self, param_as_dict: dict) -> dict[str, str]:
        """Extract enclosed surface {id: name} from AutomatedFarfield zones.

        Only returns non-empty when an AutomatedFarfield zone has enclosed_surfaces set.
        """
        volume_zones = get_value_with_path(param_as_dict, ["meshing", "volume_zones"])
        if not volume_zones:
            volume_zones = get_value_with_path(param_as_dict, ["meshing", "zones"])
        if not volume_zones:
            return {}

        for zone in volume_zones:
            if zone.get("type") != "AutomatedFarfield":
                continue
            enclosed = zone.get("enclosed_surfaces")
            if not enclosed:
                return {}
            # At this stage enclosed_surfaces is a dict with materialized entity objects
            # in stored_entities (same pattern as _get_to_be_generated_custom_volumes).
            surfaces = enclosed.get("stored_entities", [])
            return {s.private_attribute_id: s.name for s in surfaces}

        return {}

    def __init__(self, param_as_dict: dict, referenced_expressions: list):
        self.farfield_method = self._get_farfield_method_(param_as_dict=param_as_dict)
        self.farfield_domain_type = self._get_farfield_domain_type_(param_as_dict=param_as_dict)
        self.is_beta_mesher = self._get_is_beta_mesher_(param_as_dict=param_as_dict)
        self.use_geometry_AI = self._get_use_geometry_AI_(  # pylint:disable=invalid-name
            param_as_dict=param_as_dict
        )
        self.use_snappy = self._get_use_snappy_(param_as_dict=param_as_dict)
        self.using_liquid_as_material = self._get_using_liquid_as_material_(
            param_as_dict=param_as_dict
        )
        self.time_stepping = self._get_time_stepping_(param_as_dict=param_as_dict)
        self.feature_usage = self._get_feature_usage_info(param_as_dict=param_as_dict)
        self.referenced_expressions = referenced_expressions
        self.project_length_unit = self._get_project_length_unit_(param_as_dict=param_as_dict)
        self.global_bounding_box = self._get_global_bounding_box(param_as_dict=param_as_dict)
        self.planar_face_tolerance = self._get_planar_face_tolerance(param_as_dict=param_as_dict)
        # Initialized as None. When SimulationParams field validation succeeds, the
        # field validators will populate these with validated objects.
        # None = validation not yet complete (or failed)
        # {} or {id: obj} = validation succeeded
        self.output_dict = None
        self.physics_model_dict = None
        self.half_model_symmetry_plane_center_y = self._get_half_model_symmetry_plane_center_y(
            param_as_dict=param_as_dict
        )
        self.quasi_3d_symmetry_planes_center_y = self._get_quasi_3d_symmetry_planes_center_y(
            param_as_dict=param_as_dict
        )
        self.entity_transformation_detected = self._get_entity_transformation_detected(
            param_as_dict=param_as_dict
        )
        self.root_asset_type = self._get_root_asset_type(param_as_dict=param_as_dict)

        # Entity expansion support
        # Eagerly deserialize entity_info and build registry (needed for selector expansion)
        self._entity_info, self._entity_registry = self._build_entity_info_and_registry(
            param_as_dict
        )
        # Lazy initialization for selector-specific data
        self._selector_cache = None

        # Must be after _entity_registry initialization (needs selector expansion)
        self.to_be_generated_custom_volumes = self._get_to_be_generated_custom_volumes(
            param_as_dict=param_as_dict
        )
        self.farfield_enclosed_surfaces = self._get_farfield_enclosed_surfaces(
            param_as_dict=param_as_dict
        )

    def will_generate_forced_symmetry_plane(self) -> bool:
        """
        Check if the forced symmetry plane will be generated.
        """
        return (
            self.use_geometry_AI
            and self.is_beta_mesher
            and self.farfield_domain_type in ("half_body_positive_y", "half_body_negative_y")
        )

    @classmethod
    def _build_entity_info_and_registry(cls, param_as_dict: dict):
        """Build entity_info and entity_registry from param_as_dict.

        The entity_info owns the deserialized entities, and entity_registry
        holds references to them.

        Returns
        -------
        tuple[EntityInfo, EntityRegistry] or (None, None)
            The deserialized entity_info and registry, or (None, None) if not available.
        """
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.framework.entity_expansion_utils import (
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
        """Get the deserialized entity_info.

        This allows reusing the already-deserialized entity_info in the
        SimulationParams constructor to avoid double deserialization.

        Returns
        -------
        EntityInfo or None
            The deserialized entity_info, or None if not available.
        """
        return self._entity_info

    def get_entity_registry(self):
        """Get the entity_registry.

        Returns
        -------
        EntityRegistry or None
            The entity_registry, or None if not available.
        """
        return self._entity_registry

    def expand_entity_list(self, entity_list) -> list:
        """
        Expand selectors in an EntityList and return the combined list of entities.

        This method performs on-demand expansion without modifying the original input.
        Results are cached per selector to avoid recomputation across multiple validator calls.

        Why expand on-demand?
        - This helps to avoid SimulationParams object being stateful (expanded vs not expanded)
          With this function, we can always assume the entity_list is not expanded for all SimulationParams objects.

        Parameters
        ----------
        entity_list : EntityList
            A deserialized EntityList object with `stored_entities` and `selectors` attributes.

        Returns
        -------
        list
            Combined list of stored_entities and selector-matched entities.
            Returns stored_entities directly if no selectors present.
        """
        stored_entities = list(entity_list.stored_entities or [])
        raw_selectors = entity_list.selectors or []

        # Fast path: no selectors or no registry available
        if not raw_selectors or self._entity_registry is None:
            return stored_entities

        # Lazily initialize selector-specific infrastructure
        self._ensure_selector_cache()
        # pylint: disable=import-outside-toplevel
        from flow360.component.simulation.framework.entity_selector import (
            expand_entity_list_selectors,
        )

        return expand_entity_list_selectors(
            self._entity_registry,
            entity_list,
            selector_cache=self._selector_cache,
            merge_mode="merge",
        )


class ValidationContext:
    """
    Context manager for setting the validation level and additional background.

    1. Allows setting a specific validation level within a context, which influences
    the conditional validation of fields based on the defined levels.

    2. Allow associating additional information (usually info from the params) to serve as the
    background for validators.

    Note: We cannot use Pydantic validation context
    (see https://docs.pydantic.dev/latest/concepts/validators/#validation-context) because explicitly
    defining constructor blocks context from passing in.
    """

    def __init__(self, levels: Union[str, List[str]], info: ParamsValidationInfo = None):
        valid_levels = {SURFACE_MESH, VOLUME_MESH, CASE, ALL}
        if isinstance(levels, str):
            levels = [levels]
        if (
            levels is None
            or isinstance(levels, list)
            and all(lvl in valid_levels for lvl in levels)
        ):
            self.levels = levels
            self.level_token = None
            self.validation_warnings = []
        else:
            raise ValueError(f"Invalid validation level: {levels}")

        self.info = info
        self.info_token = None
        self.warnings_token = None

    def __enter__(self):
        self.level_token = _validation_level_ctx.set(self.levels)
        self.info_token = _validation_info_ctx.set(self.info)
        self.warnings_token = _validation_warnings_ctx.set(self.validation_warnings)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _validation_level_ctx.reset(self.level_token)
        _validation_info_ctx.reset(self.info_token)
        if self.warnings_token is not None:
            _validation_warnings_ctx.reset(self.warnings_token)


def get_validation_levels() -> list:
    """
    Retrieves the current validation level from the context.

    Returns:
        The current validation level, which can influence field validation behavior.
    """
    return _validation_level_ctx.get()


def get_validation_info() -> ParamsValidationInfo:
    """
    Retrieves the current validation background knowledge from the context.

    Returns:
        The validation info, which can influence validation behavior.
    """
    return _validation_info_ctx.get()


def add_validation_warning(message: str) -> None:
    """
    Append a validation warning message to the active ValidationContext.

    Parameters
    ----------
    message : str
        Warning message to record. Converted to string if needed.

    Notes
    -----
    No action is taken if there is no active ValidationContext.
    """
    warnings_list = _validation_warnings_ctx.get()
    if warnings_list is None:
        return
    message_str = str(message)
    if any(
        isinstance(existing, dict) and existing.get("msg") == message_str
        for existing in warnings_list
    ):
        return
    warnings_list.append(
        {
            "loc": (),
            "msg": message_str,
            "type": "value_error",
            "ctx": {},
        }
    )


# pylint: disable=invalid-name
def ContextField(
    default=None, *, context: Literal["SurfaceMesh", "VolumeMesh", "Case"] = None, **kwargs
):
    """
    Creates a field with context validation (the field is not required).

    Parameters
    ----------
    default : any, optional
        The default value for the field.
    context : str, optional
        Specifies the scenario for which this field is relevant. The relevant_for=context is included in error->ctx
    **kwargs : dict
        Additional keyword arguments passed to Pydantic's Field.

    Returns
    -------
    Field
        A Pydantic Field configured with conditional validation context.

    Notes
    -----
    Use this field for not required fields to provide context information during validation.
    """
    return Field(
        default,
        json_schema_extra={"relevant_for": context, "conditionally_required": False},
        **kwargs,
    )


# pylint: disable=invalid-name
def ConditionalField(
    default=None,
    *,
    context: Union[
        None,
        Literal["SurfaceMesh", "VolumeMesh", "Case"],
        List[Literal["SurfaceMesh", "VolumeMesh", "Case"]],
    ] = None,
    **kwargs,
):
    """
    Creates a field with conditional context validation requirements.

    Parameters
    ----------
    default : any, optional
        The default value for the field.
    context : Union[
        None, Literal['SurfaceMesh', 'VolumeMesh', 'Case'],
        List[Literal['SurfaceMesh', 'VolumeMesh', 'Case']]
    ], optional
        Specifies the context(s) in which this field is relevant. This value is
        included in the validation error context (`ctx`) as `relevant_for`.
    **kwargs : dict
        Additional keyword arguments passed to Pydantic's Field.

    Returns
    -------
    Field
        A Pydantic Field configured with conditional validation context.

    Notes
    -----
    Use this field for required fields but only for certain scenarios, such as volume meshing only.
    """
    return Field(
        default,
        json_schema_extra={"relevant_for": context, "conditionally_required": True},
        **kwargs,
    )


# pylint: disable=invalid-name
def CaseField(default=None, **kwargs):
    """
    Creates a field specifically relevant for the Case scenario.

    Parameters
    ----------
    default : any, optional
        The default value for the field.
    **kwargs : dict
        Additional keyword arguments passed to the pd.Field().

    Returns
    -------
    Field
        A Pydantic Field configured for fields relevant only to the Case scenario.

    Notes
    -----
    Use this field for fields that are not required but make sense only for Case.
    """
    return ContextField(default, context=CASE, **kwargs)


def context_validator(context: Literal["SurfaceMesh", "VolumeMesh", "Case"]):
    """
    Decorator to conditionally run a validator based on the current validation context.

    This decorator runs the decorated validator function only if the current validation
    level matches the specified context or if the validation level is set to ALL.

    Parameters
    ----------
    context : Literal["SurfaceMesh", "VolumeMesh", "Case"]
        The specific validation context in which the validator should be run.

    Returns
    -------
    Callable
        The decorated function that will only run when the specified context condition is met.

    Notes
    -----
    This decorator is designed to be used with Pydantic model validators to ensure that
    certain validations are only executed when the validation level matches the given context.
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(self: Any, *args, **kwargs):
            current_levels = get_validation_levels()
            # Run the validator only if the current levels matches the specified context or is ALL
            if current_levels is None or any(lvl in (context, ALL) for lvl in current_levels):
                return func(self, *args, **kwargs)
            return self

        return wrapper

    return decorator


# pylint: disable=invalid-name
def contextual_field_validator(*fields, mode="after", required_context=None, **kwargs):
    """
    Wrapper around pydantic.field_validator that automatically skips validation
    if get_validation_info() returns None or if required param_info attributes are None.

    This function accepts the same parameters as pydantic.field_validator and
    returns a decorator that wraps the validator function. The validator will
    be skipped if validation_info is not available or if any required attributes
    in param_info are None.

    Parameters
    ----------
    *fields : str
        Field names to validate (same as pydantic.field_validator)
    mode : str, optional
        Validation mode: "before", "after", "wrap" (default: "after")
    required_context : list of str, optional
        List of ParamsValidationInfo attribute names that must be not None for validation to run.
        If any of these attributes is None, the validator returns early without running.
        Common values: ["output_dict"], ["physics_model_dict"], ["output_dict", "physics_model_dict"]
    **kwargs : dict
        Additional keyword arguments passed to pydantic.field_validator

    Returns
    -------
    Callable
        A decorator that wraps the validator function

    Usage
    -----
    # Basic usage without requirements
    @contextual_field_validator("volume_zones", mode="after")
    @classmethod
    def _check_volume_zones_have_unique_names(cls, v):
        # No need to manually check get_validation_info()
        # Validation logic here
        return v

    # With required context attributes
    @contextual_field_validator("monitor_output", mode="after", required_context=["output_dict"])
    @classmethod
    def _check_monitor_exists_in_output_list(cls, v, param_info: ParamsValidationInfo):
        # No need to manually check if output_dict is None
        # param_info.output_dict is guaranteed to be not None here
        if param_info.output_dict.get(v) is None:
            raise ValueError("The monitor output does not exist in the outputs list.")
        return v

    # With multiple required context attributes
    @contextual_field_validator("models", mode="after", required_context=["output_dict", "physics_model_dict"])
    @classmethod
    def _check_models_with_outputs(cls, v, param_info: ParamsValidationInfo):
        # Both output_dict and physics_model_dict are guaranteed to be not None
        # Validation logic here
        return v

    Notes
    -----
    This is equivalent to using pd.field_validator with manual checks:
    @pd.field_validator("volume_zones", mode="after")
    @classmethod
    def _check_volume_zones_have_unique_names(cls, v, param_info: ParamsValidationInfo):
        param_info = get_validation_info()
        if param_info is None:
            return v
        if param_info.output_dict is None:
            return v
        # Validation logic here
        return v
    """

    def decorator(func: Callable):
        # Handle classmethod and staticmethod
        is_classmethod = isinstance(func, classmethod)
        is_staticmethod = isinstance(func, staticmethod)

        if is_classmethod:
            original_func = func.__func__
        elif is_staticmethod:
            original_func = func.__func__
        else:
            original_func = func

        original_sig = inspect.signature(original_func)
        pass_param_info = "param_info" in original_sig.parameters
        new_signature = original_sig
        original_signature_backup = getattr(original_func, "__signature__", None)

        if pass_param_info:
            params_without = tuple(
                param for name, param in original_sig.parameters.items() if name != "param_info"
            )
            new_signature = original_sig.replace(parameters=params_without)
            original_func.__signature__ = new_signature

        @wraps(original_func)
        def wrapper(*args, **kwargs_inner):
            param_info = get_validation_info()
            if param_info is None:
                if not args:
                    return None
                # Determine the index of the value argument.
                value_idx = 1 if isinstance(args[0], type) and len(args) >= 2 else 0
                return args[value_idx]

            # Check if required context attributes are available
            if required_context:
                for attr_name in required_context:
                    if not hasattr(param_info, attr_name):
                        raise ValueError(f"Invalid validation context attribute: {attr_name}")
                    if getattr(param_info, attr_name) is None:
                        # Required context attribute is None, skip validation
                        if not args:
                            return None
                        value_idx = 1 if isinstance(args[0], type) and len(args) >= 2 else 0
                        return args[value_idx]

            # Call the original function (not the classmethod/staticmethod wrapper)
            call_kwargs = dict(kwargs_inner)
            if pass_param_info:
                call_kwargs["param_info"] = param_info
            return original_func(*args, **call_kwargs)

        if pass_param_info:
            if original_signature_backup is None:
                if hasattr(original_func, "__signature__"):
                    del original_func.__signature__
            else:
                original_func.__signature__ = original_signature_backup

        # If original was classmethod/staticmethod, wrap so pydantic recognizes it
        if is_classmethod:
            wrapped_func = classmethod(wrapper)
        elif is_staticmethod:
            wrapped_func = staticmethod(wrapper)
        else:
            wrapped_func = wrapper

        return pd.field_validator(*fields, mode=mode, **kwargs)(wrapped_func)

    return decorator


# pylint: disable=invalid-name
def contextual_model_validator(mode="after", **kwargs):
    """
    Wrapper around pydantic.model_validator that automatically skips validation
    if get_validation_info() returns None.

    This function accepts the same parameters as pydantic.model_validator and
    returns a decorator that wraps the validator function. The validator will
    be skipped if validation_info is not available.

    Parameters
    ----------
    mode : str, optional
        Validation mode: "before", "after", "wrap" (default: "after")
    **kwargs : dict
        Additional keyword arguments passed to pydantic.model_validator

    Returns
    -------
    Callable
        A decorator that wraps the validator function

    Usage
    -----
    @contextual_model_validator(mode="after")
    def _check_no_reused_volume_entities(self):
        # No need to manually check get_validation_info()
        # Validation logic here
        return self

    Notes
    -----
    This is equivalent to using pd.model_validator with a manual check:
    @pd.model_validator(mode="after")
    def _check_no_reused_volume_entities(self):
        if not get_validation_info():
            return self
        # Validation logic here
        return self
    """

    def decorator(func: Callable):
        original_sig = inspect.signature(func)
        pass_param_info = "param_info" in original_sig.parameters
        new_signature = original_sig
        original_signature_backup = getattr(func, "__signature__", None)

        if pass_param_info:
            params_without = tuple(
                param for name, param in original_sig.parameters.items() if name != "param_info"
            )
            new_signature = original_sig.replace(parameters=params_without)
            func.__signature__ = new_signature

        @wraps(func)
        def wrapper(*args, **kwargs_inner):
            param_info = get_validation_info()
            if param_info is None:
                if args:
                    return args[0]
                return None
            call_kwargs = dict(kwargs_inner)
            if pass_param_info:
                call_kwargs["param_info"] = param_info
            return func(*args, **call_kwargs)

        if pass_param_info:
            if original_signature_backup is None:
                if hasattr(func, "__signature__"):
                    del func.__signature__
            else:
                func.__signature__ = original_signature_backup

        return pd.model_validator(mode=mode, **kwargs)(wrapper)

    return decorator
