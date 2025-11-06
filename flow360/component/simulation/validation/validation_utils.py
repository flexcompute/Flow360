"""
validation utility functions
"""

from functools import wraps
from typing import get_args

from flow360.component.simulation.entity_info import DraftEntityTypes
from flow360.component.simulation.primitives import (
    Surface,
    _SurfaceEntityBase,
    _VolumeEntityBase,
)
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
)


def _validator_append_instance_name(func):
    """
    If the validation throw ValueError (expected), append the instance name to the error message.
    """

    def _get_prepend_message(type_object, instance_name):
        if instance_name is None:
            prepend_message = "In one of the " + type_object.__name__
        else:
            prepend_message = f"{type_object.__name__} with name '{instance_name}'"
        return prepend_message

    @wraps(func)
    def wrapper(*args, **kwargs):
        prepend_message = None
        # for field validator
        if len(args) == 3:
            model_class = args[0]
            validation_info = args[2]
            if validation_info is not None:
                name = validation_info.data.get("name", None)
                prepend_message = _get_prepend_message(model_class, name)
            else:
                raise NotImplementedError(
                    "[Internal] Make sure your field_validator has validationInfo in the args or"
                    " this wrapper is used with a field_validator!!"
                )
        # for model validator
        elif len(args) == 1:
            instance = args[0]
            if "name" not in instance.model_dump():
                raise NotImplementedError("[Internal] Make sure the model has name field.")
            prepend_message = _get_prepend_message(type(instance), instance.name)
        else:
            raise NotImplementedError(
                f"[Internal] {_validator_append_instance_name.__qualname__} decorator only supports 1 or 3 arguments."
            )

        try:
            result = func(*args, **kwargs)  # Call the original function
            return result
        except ValueError as e:
            raise ValueError(f"{prepend_message}: {str(e)}") from e

    return wrapper


def check_deleted_surface_in_entity_list(value):
    """
    Check if any boundary is meant to be deleted
    value--> EntityList
    """
    validation_info = get_validation_info()
    if validation_info is None:
        # validation not necessary now.
        return value

    # - Check if the surfaces are deleted.
    for surface in value.stored_entities:
        if isinstance(
            surface, Surface
        ) and surface._will_be_deleted_by_mesher(  # pylint:disable=protected-access
            at_least_one_body_transformed=validation_info.at_least_one_body_transformed,
            farfield_method=validation_info.farfield_method,
            global_bounding_box=validation_info.global_bounding_box,
            planar_face_tolerance=validation_info.planar_face_tolerance,
            half_model_symmetry_plane_center_y=validation_info.half_model_symmetry_plane_center_y,
            quasi_3d_symmetry_planes_center_y=validation_info.quasi_3d_symmetry_planes_center_y,
        ):
            raise ValueError(
                f"Boundary `{surface.name}` will likely be deleted after mesh generation. "
                "Therefore it cannot be used."
            )

    return value


def check_deleted_surface_pair(value):
    """
    Check if any boundary is meant to be deleted
    value--> SurfacePair
    """

    validation_info = get_validation_info()
    if validation_info is None:
        # validation not necessary now.
        return value

    # - Check if the surfaces are deleted.
    for surface in value.pair:
        if isinstance(
            surface, Surface
        ) and surface._will_be_deleted_by_mesher(  # pylint:disable=protected-access
            at_least_one_body_transformed=validation_info.at_least_one_body_transformed,
            farfield_method=validation_info.farfield_method,
            global_bounding_box=validation_info.global_bounding_box,
            planar_face_tolerance=validation_info.planar_face_tolerance,
            half_model_symmetry_plane_center_y=validation_info.half_model_symmetry_plane_center_y,
            quasi_3d_symmetry_planes_center_y=validation_info.quasi_3d_symmetry_planes_center_y,
        ):
            raise ValueError(
                f"Boundary `{surface.name}` will likely be deleted after mesh generation. "
                "Therefore it cannot be used."
            )

    return value


def check_user_defined_farfield_symmetry_existence(stored_entities):
    """
    Ensure that when:
    1. UserDefinedFarfield is used
    2. GhostSurface(name="symmetric") is assigned

    That:
    1. GAI and beta mesher is used.
    2. Domain type is half_body_positive_y or half_body_negative_y
    """
    validation_info = get_validation_info()

    if validation_info is None:
        return stored_entities

    if validation_info.farfield_method != "user-defined":
        return stored_entities

    for item in stored_entities:
        if (
            item.private_attribute_entity_type_name != "GhostCircularPlane"
            or item.name != "symmetric"
        ):
            continue
        if not validation_info.use_geometry_AI or not validation_info.is_beta_mesher:
            raise ValueError(
                "Symmetry plane of user defined farfield will only be generated when both GAI and beta mesher are used."
            )
        if validation_info.farfield_domain_type not in (
            "half_body_positive_y",
            "half_body_negative_y",
        ):
            raise ValueError(
                "Symmetry plane of user defined farfield is only supported for half body domains."
            )
    return stored_entities


def check_symmetric_boundary_existence(stored_entities):
    """For automated farfield, check according to the criteria if the symmetric plane exists."""
    validation_info = get_validation_info()

    if validation_info is None:
        return stored_entities

    for item in stored_entities:
        if item.private_attribute_entity_type_name != "GhostCircularPlane":
            continue

        if not item.exists(validation_info):
            # pylint: disable=protected-access
            y_min, y_max, tolerance, largest_dimension = item._get_existence_dependency(
                validation_info
            )
            error_msg = (
                "`symmetric` boundary will not be generated: "
                + f"model spans: [{y_min:.2g}, {y_max:.2g}], "
                + f"tolerance = {validation_info.planar_face_tolerance:.2g} x {largest_dimension:.2g}"
                + f" = {tolerance:.2g}."
            )

            raise ValueError(error_msg)

    return stored_entities


def _ghost_surface_names(stored_entities) -> list[str]:
    """Collect names of ghost-type boundaries in the list."""
    names = []
    for item in stored_entities:
        entity_type = getattr(item, "private_attribute_entity_type_name", None)
        if isinstance(entity_type, str) and entity_type.startswith("Ghost"):
            names.append(getattr(item, "name", ""))
    return names


def check_ghost_surface_usage_policy_for_face_refinements(stored_entities, *, feature_name: str):
    """
    Enforce GhostSurface usage policy for face-based refinements (SurfaceRefinement, PassiveSpacing).

    Rules provided by product spec:
    - If starting from Geometry, SurfaceRefinement and PassiveSpacing both can use GhostSurface, if:
        - Automated farfield: if using beta mesher.
        - User-defined farfield: if using GAI and beta mesher.
    - If starting from Surface mesh:
        - Automated farfield: allow GhostSurface for PassiveSpacing only.
        - User-defined farfield: do not allow any GhostSurface.
    """
    validation_info = get_validation_info()
    if validation_info is None:
        return stored_entities

    if not stored_entities:
        return stored_entities

    ghost_names = _ghost_surface_names(stored_entities)
    if not ghost_names:
        return stored_entities

    root_asset_type = getattr(validation_info, "root_asset_type", None)
    farfield_method = validation_info.farfield_method
    use_beta = validation_info.is_beta_mesher
    use_gai = validation_info.use_geometry_AI

    # Default error messages
    def _err(msg):
        raise ValueError(msg)

    names_str = ", ".join(sorted(set(ghost_names)))

    if root_asset_type == "geometry":
        if farfield_method == "user-defined":
            if not (use_beta and use_gai):
                _err(
                    (
                        f"Face refinements on '{names_str}' require both Geometry AI and the beta mesher "
                        "when using user-defined farfield."
                    )
                )
        else:
            # automated variants (auto / quasi-3d / quasi-3d-periodic)
            if not use_beta:
                _err(
                    (
                        f"Face refinements on '{names_str}' for automated farfield "
                        "requires beta mesher."
                    )
                )
        return stored_entities

    if root_asset_type == "surface_mesh":
        if farfield_method == "user-defined":
            _err(
                (
                    f"Boundary '{names_str}' is not allowed when starting from an uploaded surface mesh "
                    "with user-defined farfield."
                )
            )
        # automated variants
        if feature_name == "SurfaceRefinement":
            _err(
                (
                    f"Boundary '{names_str}' is not allowed for SurfaceRefinement when starting from an "
                    "uploaded surface mesh with automated farfield."
                )
            )
        return stored_entities

    # Other asset type: proceed without additional restriction
    return stored_entities


class EntityUsageMap:  # pylint:disable=too-few-public-methods
    """
    A customized dict to store the entity name and its usage.
    {"$EntityID": [$UsedInWhatModel]}
    """

    def __init__(self):
        self.dict_entity = {"Surface": {}, "Volume": {}}

    @classmethod
    def _get_entity_key(cls, entity) -> str:
        """
        Get unique identifier for the entity.
        """
        draft_entity_types = get_args(get_args(DraftEntityTypes)[0])
        if isinstance(entity, draft_entity_types):
            return entity.private_attribute_id
        return entity.name

    def add_entity_usage(self, entity, model_type):
        """
        Add the entity usage to the dictionary.
        """
        entity_type = None
        if isinstance(entity, _SurfaceEntityBase):
            entity_type = "Surface"
        elif isinstance(entity, _VolumeEntityBase):
            entity_type = "Volume"
        else:
            raise ValueError(
                f"[Internal Error] Entity `{entity.name}` in the {model_type} model "
                f"cannot be registered as a valid Surface or Volume entity."
            )
        entity_key = self._get_entity_key(entity=entity)
        entity_log = self.dict_entity[entity_type].get(
            entity_key, {"entity_name": entity.name, "model_list": []}
        )
        entity_log["model_list"].append(model_type)
        self.dict_entity[entity_type][entity_key] = entity_log
