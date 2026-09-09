"""Meshing related parameters for volume and surface mesher."""

import logging
from typing import Annotated, Literal

import pydantic as pd
from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.models.entities.volume_entities import (
    AxisymmetricBody,
    Box,
    Cylinder,
    SeedpointVolume,
    Sphere,
)
from flow360_schema.models.simulation.framework.updater import (
    DEFAULT_PLANAR_FACE_TOLERANCE,
    DEFAULT_SLIDING_INTERFACE_TOLERANCE,
)
from flow360_schema.models.simulation.meshing_param import snappy
from flow360_schema.models.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360_schema.models.simulation.meshing_param.face_params import (
    BoundaryLayer,
    GeometryRefinement,
    PassiveSpacing,
    SurfaceRefinement,
)
from flow360_schema.models.simulation.meshing_param.meshing_specs import (
    MeshingDefaults,
    VolumeMeshingDefaults,
)
from flow360_schema.models.simulation.meshing_param.meshing_validators import (
    validate_geometry_ai_size_ordering,
    validate_snappy_uniform_refinement_entities,
)
from flow360_schema.models.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    CustomVolume,
    CustomZones,
    MeshSliceOutput,
    RotationCylinder,
    RotationSphere,
    RotationVolume,
    StructuredBoxRefinement,
    UniformRefinement,
    UserDefinedFarfield,
    WindTunnelFarfield,
    _FarfieldAllowingEnclosedEntities,
)
from flow360_schema.models.simulation.validation.validation_context import (
    SURFACE_MESH,
    VOLUME_MESH,
    ContextField,
    ParamsValidationInfo,
    add_validation_warning,
    contextual_field_validator,
    contextual_model_validator,
)
from flow360_schema.models.simulation.validation.validation_utils import EntityUsageMap
from typing_extensions import Self

logger = logging.getLogger(__name__)

RefinementTypes = Annotated[
    SurfaceEdgeRefinement
    | SurfaceRefinement
    | GeometryRefinement
    | BoundaryLayer
    | PassiveSpacing
    | UniformRefinement
    | StructuredBoxRefinement
    | AxisymmetricRefinement,
    pd.Field(discriminator="refinement_type"),
]

VolumeZonesTypes = Annotated[
    RotationVolume
    | RotationCylinder
    | RotationSphere
    | AutomatedFarfield
    | UserDefinedFarfield
    | CustomZones
    | WindTunnelFarfield,
    pd.Field(discriminator="type"),
]

ZoneTypesModular = Annotated[
    RotationVolume | RotationSphere | AutomatedFarfield | UserDefinedFarfield | CustomZones,
    pd.Field(discriminator="type"),
]

VolumeRefinementTypes = Annotated[
    UniformRefinement | AxisymmetricRefinement | BoundaryLayer | PassiveSpacing | StructuredBoxRefinement,
    pd.Field(discriminator="refinement_type"),
]


def _collect_rotation_entity_names(zones, param_info, zone_types):
    """Collect entity names associated with RotationVolume/RotationCylinder/RotationSphere zones."""
    names: set[str] = set()
    for zone in zones:
        if isinstance(zone, zone_types):
            for entity in param_info.expand_entity_list(zone.entities):
                names.add(entity.name)
    return names


def _validate_farfield_enclosed_entities(zones, rotation_entity_names, has_custom_volumes, param_info):
    """Validate farfield enclosed_entities: require CustomVolumes and rotation-volume association.
    Only applies to farfield types that support enclosed_entities (Automated, WindTunnel).
    """
    for zone in zones:
        if not isinstance(zone, _FarfieldAllowingEnclosedEntities):
            continue

        if zone.enclosed_entities is None:
            if has_custom_volumes:
                raise ValueError(
                    "`enclosed_entities` for farfield must be specified when "
                    "`CustomVolume` entities are present in volume zones."
                )
            continue

        if not has_custom_volumes:
            raise ValueError(
                "`enclosed_entities` for farfield is only allowed when "
                "`CustomVolume` entities are present in volume zones."
            )

        for entity in param_info.expand_entity_list(zone.enclosed_entities):
            if isinstance(entity, (Cylinder, AxisymmetricBody, Sphere)) and entity.name not in rotation_entity_names:
                raise ValueError(
                    f"`{type(entity).__name__}` entity `{entity.name}` in "
                    f"`enclosed_entities` must be associated with a `RotationVolume` or `RotationSphere`."
                )


def _collect_all_custom_volumes(zones):
    """Collect all CustomVolume instances from CustomZones."""
    custom_volumes: list[CustomVolume] = []
    for zone in zones:
        if isinstance(zone, CustomZones):
            for cv in zone.entities.stored_entities:
                if isinstance(cv, CustomVolume):
                    custom_volumes.append(cv)
    return custom_volumes


def _collect_all_seedpoint_volumes(zones):
    """Collect all SeedpointVolume instances from CustomZones."""
    seedpoint_volumes: list[SeedpointVolume] = []
    for zone in zones:
        if isinstance(zone, CustomZones):
            for volume in zone.entities.stored_entities:
                if isinstance(volume, SeedpointVolume):
                    seedpoint_volumes.append(volume)
    return seedpoint_volumes


def _warn_when_refinement_factor_is_ignored(refinement_factor, param_info: ParamsValidationInfo) -> None:
    """Warn when a refinement_factor is set that the mesher in use will not act on.

    Only the legacy mesher scales its spacings by it, so the volume meshing translator does not
    even emit the setting for the beta mesher. A factor of 1 is the default and a no-op for the
    legacy mesher too, so it is not worth warning about.
    """
    if refinement_factor in (None, 1) or not param_info.is_beta_mesher:
        return
    add_validation_warning(
        f"'refinement_factor' ({refinement_factor}) is not supported by the beta mesher and will be ignored. "
        "Scale the refinement spacings and first layer thickness directly instead."
    )


def _count_generated_volume_zones(zones) -> int:
    """Number of volume zones the surface mesher generates: custom volumes, seedpoint
    volumes, and a non-UDF farfield. Mirrors the surface mesher's own zone count, which
    increments once per CustomVolume and once per SeedpointVolume.

    AutomatedFarfield and WindTunnelFarfield each generate a farfield zone; UserDefinedFarfield does not.
    """
    if not zones:
        return 0
    has_non_udf_farfield = any(isinstance(zone, (AutomatedFarfield, WindTunnelFarfield)) for zone in zones)
    return (
        len(_collect_all_custom_volumes(zones))
        + len(_collect_all_seedpoint_volumes(zones))
        + (1 if has_non_udf_farfield else 0)
    )


def _validate_seedpoint_volume_mesher_compatibility(seedpoint_volumes, param_info: ParamsValidationInfo):
    """Validate SeedpointVolume usage against mesher capabilities."""
    if seedpoint_volumes and not (param_info.use_snappy or param_info.is_beta_mesher):
        raise ValueError("`SeedpointVolume` is supported only when using snappyHexMeshing or the beta mesher.")


def _validate_seedpoint_volume_snappy_single_point(seedpoint_volumes):
    """Each `SeedpointVolume` reaching the snappy path must carry exactly one seed.

    snappy's `locationInMesh` takes a single triplet per zone; multi-seed is supported
    only by the GAI / beta mesher paths.
    """
    for sv in seedpoint_volumes:
        if len(sv.point_in_mesh) != 1:
            raise ValueError(
                f"`SeedpointVolume` `{sv.name}` has {len(sv.point_in_mesh)} seed points; "
                "Snappy requires exactly one point per zone."
            )


def _validate_seedpoint_volume_geometry_ai_exclusivity(zones, param_info: ParamsValidationInfo):
    """When geometry AI is enabled, SeedpointVolumes and explicit CustomVolume-based
    CustomZones cannot coexist. If any SeedpointVolume is present, there must be exactly
    one CustomZone, and it must contain only SeedpointVolume entities."""
    if not param_info.use_geometry_AI or param_info.use_snappy:
        return

    custom_zones = [zone for zone in zones if isinstance(zone, CustomZones)]
    seedpoint_zones = [
        zone
        for zone in custom_zones
        if any(isinstance(entity, SeedpointVolume) for entity in zone.entities.stored_entities)
    ]
    if not seedpoint_zones:
        return

    message = (
        "When using `SeedpointVolume` with geometry AI, exactly one `CustomZones` "
        "containing only `SeedpointVolume` entities is allowed; mixing with other "
        "`CustomZones` or `CustomVolume` entities is not supported."
    )
    if len(custom_zones) > 1:
        raise ValueError(message)
    sole_zone = seedpoint_zones[0]
    if any(not isinstance(entity, SeedpointVolume) for entity in sole_zone.entities.stored_entities):
        raise ValueError(message)


def _validate_custom_volume_rotation_association(custom_volumes, rotation_entity_names, param_info):
    """Validate that Cylinder/AxisymmetricBody/Sphere in CustomVolume.bounding_entities
    are associated with a RotationVolume or RotationSphere."""
    for cv in custom_volumes:
        for entity in param_info.expand_entity_list(cv.bounding_entities):
            if isinstance(entity, (Cylinder, AxisymmetricBody, Sphere)) and entity.name not in rotation_entity_names:
                raise ValueError(
                    f"`{type(entity).__name__}` entity `{entity.name}` in "
                    f"`CustomVolume` `{cv.name}` `bounding_entities` must be "
                    f"associated with a `RotationVolume` or `RotationSphere`."
                )


class MeshingParams(Flow360BaseModel):
    """
    Meshing parameters for volume and/or surface mesher. This contains all the meshing related settings.

    Example
    -------

      >>> fl.MeshingParams(
      ...     refinement_factor=1.0,
      ...     gap_treatment_strength=0.5,
      ...     defaults=fl.MeshingDefaults(
      ...         surface_max_edge_length=1*fl.u.m,
      ...         boundary_layer_first_layer_thickness=1e-5*fl.u.m
      ...     ),
      ...     volume_zones=[farfield],
      ...     refinements=[
      ...         fl.SurfaceEdgeRefinement(
      ...             edges=[geometry["edge1"], geometry["edge2"]],
      ...             method=fl.AngleBasedRefinement(value=8*fl.u.deg)
      ...         ),
      ...         fl.SurfaceRefinement(
      ...             faces=[geometry["face1"], geometry["face2"]],
      ...             max_edge_length=0.001*fl.u.m
      ...         ),
      ...         fl.UniformRefinement(
      ...             entities=[cylinder, box],
      ...             spacing=1*fl.u.cm
      ...         )
      ...     ]
      ... )

    ====
    """

    type_name: Literal["MeshingParams"] = pd.Field("MeshingParams", frozen=True)
    refinement_factor: pd.PositiveFloat | None = pd.Field(
        default=1,
        description="All spacings in refinement regions"
        + "and first layer thickness will be adjusted to generate `r`-times"
        + " finer mesh where r is the refinement_factor value.",
    )

    gap_treatment_strength: float | None = ContextField(
        default=None,
        ge=0,
        le=1,
        description="Narrow gap treatment strength used when two surfaces are in close proximity."
        " Use a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment."
        " This parameter has a global impact where the anisotropic transition into the isotropic mesh."
        " However the impact on regions without close proximity is negligible.",
        context=VOLUME_MESH,
    )

    defaults: MeshingDefaults = pd.Field(
        MeshingDefaults(),
        description="Default settings for meshing."
        " In other words the settings specified here will be applied"
        " as a default setting for all `Surface` (s) and `Edge` (s).",
    )

    refinements: list[RefinementTypes] = pd.Field(
        default=[],
        description="Additional fine-tunning for refinements on top of :py:attr:`defaults`",
    )
    # Will add more to the Union
    volume_zones: list[VolumeZonesTypes] | None = pd.Field(default=None, description="Creation of new volume zones.")

    # Meshing outputs (for now, volume mesh slices)
    outputs: list[MeshSliceOutput] = pd.Field(
        default=[],
        description="Mesh output settings.",
    )

    private_attribute_dict: dict | None = pd.Field(None)

    @pd.field_validator("volume_zones", mode="after")
    @classmethod
    def _check_volume_zones_has_farfield(cls, v):
        if v is None:
            # User did not put anything in volume_zones so may not want to use volume meshing
            return v

        total_farfield = sum(
            isinstance(
                volume_zone,
                (AutomatedFarfield, WindTunnelFarfield, UserDefinedFarfield),
            )
            for volume_zone in v
        )
        if total_farfield == 0:
            raise ValueError("Farfield zone is required in `volume_zones`.")

        if total_farfield > 1:
            raise ValueError("Only one farfield zone is allowed in `volume_zones`.")

        return v

    @contextual_field_validator("volume_zones", mode="after")
    @classmethod
    def _check_automated_farfield_custom_volumes(cls, v, param_info):
        if v is None:
            return v

        automated_farfield = next((zone for zone in v if isinstance(zone, AutomatedFarfield)), None)
        if automated_farfield is not None:
            custom_volumes = _collect_all_custom_volumes(v)
            if any(cv.name == "farfield" for cv in custom_volumes):
                raise ValueError(
                    "CustomVolume name 'farfield' is reserved when using AutomatedFarfield. "
                    "The 'farfield' zone will be automatically generated using `AutomatedFarfield.enclosed_entities`. "
                    "Please choose a different name."
                )

            enclosed_entities = (
                param_info.expand_entity_list(automated_farfield.enclosed_entities)
                if automated_farfield.enclosed_entities is not None
                else []
            )

            if custom_volumes and not enclosed_entities:
                raise ValueError(
                    "When using AutomatedFarfield with CustomVolumes, `enclosed_entities` must be "
                    "specified on the AutomatedFarfield to define the exterior farfield zone boundary."
                )
            if enclosed_entities and not custom_volumes:
                raise ValueError(
                    "`enclosed_entities` on AutomatedFarfield is only allowed when CustomVolume entities are used. "
                    "Without custom volumes, the farfield zone will be automatically detected."
                )
            if any(s.name == "farfield" for s in enclosed_entities):
                raise ValueError(
                    "Surface name 'farfield' in `enclosed_entities` will conflict with the automatically "
                    "generated farfield boundary. Please choose a different surface."
                )

        return v

    @contextual_field_validator("volume_zones", mode="after")
    @classmethod
    def _check_enclosed_entities_rotation_volume_association(cls, v, param_info: ParamsValidationInfo):
        """
        Ensure that:
        - enclosed_entities on any farfield requires at least one CustomZone
        - Cylinder, AxisymmetricBody, and Sphere entities in enclosed_entities
          are associated with a RotationVolume or RotationSphere
        """
        if v is None:
            return v

        rotation_entity_names = _collect_rotation_entity_names(
            v, param_info, (RotationVolume, RotationCylinder, RotationSphere)
        )
        custom_volumes = _collect_all_custom_volumes(v)
        has_custom_volumes = len(custom_volumes) > 0
        _validate_farfield_enclosed_entities(v, rotation_entity_names, has_custom_volumes, param_info)
        _validate_custom_volume_rotation_association(custom_volumes, rotation_entity_names, param_info)

        return v

    @contextual_field_validator("volume_zones", mode="after")
    @classmethod
    def _check_volume_zones_have_unique_names(cls, v):
        """Ensure there won't be duplicated volume zone names."""

        if v is None:
            return v

        to_be_generated_volume_zone_names = set()
        for volume_zone in v:
            if not isinstance(volume_zone, CustomZones):
                continue
            # Extract CustomVolume from CustomZones
            for custom_volume in volume_zone.entities.stored_entities:
                if custom_volume.name in to_be_generated_volume_zone_names:
                    raise ValueError(
                        f"Multiple CustomVolume with the same name `{custom_volume.name}` are not allowed."
                    )
                to_be_generated_volume_zone_names.add(custom_volume.name)

        return v

    @contextual_field_validator("volume_zones", mode="after")
    @classmethod
    def _check_seedpoint_volume_usage(cls, v, param_info: ParamsValidationInfo):
        """Validate SeedpointVolume usage in legacy meshing schema."""
        if v is None:
            return v
        seedpoint_volumes = _collect_all_seedpoint_volumes(v)
        _validate_seedpoint_volume_mesher_compatibility(seedpoint_volumes, param_info)
        _validate_seedpoint_volume_geometry_ai_exclusivity(v, param_info)
        return v

    @contextual_model_validator(mode="after")
    def _check_no_reused_volume_entities(self) -> Self:
        """
        Meshing entities reuse check.
        +------------------------+------------------------+------------------------+------------------------+
        |                        | RotationCylinder       | AxisymmetricRefinement | UniformRefinement      |
        +------------------------+------------------------+------------------------+------------------------+
        | RotationCylinder       |          NO            |           --           |           --           |
        +------------------------+------------------------+------------------------+------------------------+
        | AxisymmetricRefinement |          NO            |           NO           |           --           |
        +------------------------+------------------------+------------------------+------------------------+
        | UniformRefinement      |          YES           |           NO           |           NO           |
        +------------------------+------------------------+------------------------+------------------------+

        +------------------------+------------------------+------------------------+
        |                        |StructuredBoxRefinement | UniformRefinement      |
        +------------------------+------------------------+------------------------+
        |StructuredBoxRefinement |          NO            |           --           |
        +------------------------+------------------------+------------------------+
        | UniformRefinement      |          NO            |           NO           |
        +------------------------+------------------------+------------------------+

        """

        usage = EntityUsageMap()

        for volume_zone in self.volume_zones if self.volume_zones is not None else []:
            if isinstance(volume_zone, (RotationVolume, RotationCylinder, RotationSphere)):
                _ = [usage.add_entity_usage(item, volume_zone.type) for item in volume_zone.entities.stored_entities]

        for refinement in self.refinements if self.refinements is not None else []:
            if isinstance(
                refinement,
                (UniformRefinement, AxisymmetricRefinement, StructuredBoxRefinement),
            ):
                _ = [
                    usage.add_entity_usage(item, refinement.refinement_type)
                    for item in refinement.entities.stored_entities
                ]

        error_msg = ""
        for entity_type, entity_model_map in usage.dict_entity.items():
            for entity_info in entity_model_map.values():
                if len(entity_info["model_list"]) == 1 or sorted(entity_info["model_list"]) in [
                    sorted(["RotationCylinder", "UniformRefinement"]),
                    sorted(["RotationVolume", "UniformRefinement"]),
                    sorted(["RotationSphere", "UniformRefinement"]),
                ]:
                    continue

                model_set = set(entity_info["model_list"])
                if len(model_set) == 1:
                    error_msg += (
                        f"{entity_type} entity `{entity_info['entity_name']}` "
                        + f"is used multiple times in `{model_set.pop()}`."
                    )
                else:
                    model_string = ", ".join(f"`{x}`" for x in sorted(model_set))
                    error_msg += (
                        f"Using {entity_type} entity `{entity_info['entity_name']}` "
                        + f"in {model_string} at the same time is not allowed."
                    )

        if error_msg:
            raise ValueError(error_msg)

        return self

    @contextual_model_validator(mode="after")
    def _warn_refinement_factor_ignored_by_beta_mesher(self, param_info: ParamsValidationInfo) -> Self:
        """Warn when refinement_factor is set but the beta mesher will ignore it."""
        _warn_when_refinement_factor_is_ignored(self.refinement_factor, param_info)
        return self

    @contextual_model_validator(mode="after")
    def _check_sizing_against_octree_series(self, param_info: ParamsValidationInfo):
        """Validate that UniformRefinement spacings align with the octree series."""
        if not param_info.is_beta_mesher:
            return self
        if self.defaults.octree_spacing is None:
            logger.warning(
                "No `octree_spacing` configured in `%s`; "
                "octree spacing validation for UniformRefinement will be skipped.",
                type(self.defaults).__name__,
            )
            return self

        if self.refinements is not None:
            for refinement in self.refinements:
                if isinstance(refinement, UniformRefinement):
                    self.defaults.octree_spacing.check_spacing(refinement.spacing, type(refinement).__name__)
        return self

    @contextual_model_validator(mode="after")
    def _warn_multi_body_without_remove_hidden_geometry(self, param_info: ParamsValidationInfo) -> Self:
        """Recommend remove_hidden_geometry for GAI geometry with more than one mesh-exterior
        (mesh_exterior=True) body, where bodies may intersect or occlude one another."""
        if not param_info.use_geometry_AI or self.defaults.remove_hidden_geometry:
            return self

        entity_info = param_info.get_entity_info()
        if getattr(entity_info, "type_name", None) != "GeometryEntityInfo":
            return self
        if entity_info.get_num_exterior_bodies() <= 1:
            return self

        add_validation_warning(
            "More than one mesh-exterior body was detected. If the bodies intersect or have parts "
            "hidden from the flow, enabling 'remove_hidden_geometry' is recommended."
        )
        return self

    @contextual_model_validator(mode="after")
    def _warn_geometry_refinement_settings_without_remove_hidden_geometry(self) -> Self:
        """Warn when any GeometryRefinement sets hidden-geometry-removal-only options
        (min_passage_size, baffle_face_treatment) without enabling remove_hidden_geometry."""
        if self.defaults.remove_hidden_geometry:
            return self
        refinements = [r for r in self.refinements or [] if isinstance(r, GeometryRefinement)]
        clauses = []
        min_passage_names = [r.name for r in refinements if r.min_passage_size is not None]
        if min_passage_names:
            clauses.append(f"'min_passage_size' on {', '.join(min_passage_names)}")
        baffle_names = [r.name for r in refinements if r.baffle_face_treatment not in (None, "default")]
        if baffle_names:
            clauses.append(f"'baffle_face_treatment' on {', '.join(baffle_names)}")
        if clauses:
            add_validation_warning(
                "The following GeometryRefinements will have no effect because "
                f"'remove_hidden_geometry' is not enabled: {'; '.join(clauses)}."
            )
        return self

    @pd.model_validator(mode="after")
    def _check_geometry_refinement_size_ordering(self) -> Self:
        """Ensure each GeometryRefinement's effective sizes are mutually consistent."""
        for refinement in self.refinements or []:
            if not isinstance(refinement, GeometryRefinement):
                continue
            geometry_accuracy = (
                refinement.geometry_accuracy
                if refinement.geometry_accuracy is not None
                else self.defaults.geometry_accuracy
            )
            sealing_size = (
                refinement.sealing_size if refinement.sealing_size is not None else self.defaults.sealing_size
            )
            # The per-face min_passage_size only takes effect when hidden geometry removal is enabled,
            # otherwise it is ignored (see _warn_min_passage_size_without_remove_hidden_geometry).
            min_passage_size = None
            if self.defaults.remove_hidden_geometry:
                min_passage_size = (
                    refinement.min_passage_size
                    if refinement.min_passage_size is not None
                    else self.defaults.min_passage_size
                )
            validate_geometry_ai_size_ordering(
                geometry_accuracy=geometry_accuracy,
                sealing_size=sealing_size,
                min_passage_size=min_passage_size,
                location=f"for GeometryRefinement '{refinement.name}'",
            )
        return self

    @contextual_model_validator(mode="after")
    def _check_default_size_ordering_against_faces(self, param_info: ParamsValidationInfo) -> Self:
        """Apply the default size ordering only to faces that fall through to the defaults.

        ``geometry_accuracy``, ``sealing_size`` and ``min_passage_size`` are all effectively per-face
        (overridable via :class:`~flow360.GeometryRefinement`), so the defaults constrain a face only
        when no refinement covers it. Covered faces are validated in
        ``_check_geometry_refinement_size_ordering`` using their effective sizes; here we cover the
        rest. When every face is covered the defaults are unused and the check is skipped. When the
        boundary list is unavailable the check is kept (conservative).
        """
        boundaries_with_refinement: set[str] = set()
        for refinement in self.refinements or []:
            # Box regions override spatially, not per whole face, so faces touched only by a
            # box conservatively remain under the defaults check.
            if isinstance(refinement, GeometryRefinement):
                boundaries_with_refinement.update(
                    entity.name
                    for entity in param_info.expand_entity_list(refinement.entities)
                    if not isinstance(entity, Box)
                )

        # Both sides are Surface.name resolved from the same entity_info grouping, so the subset
        # test is well defined; a name mismatch could only drop a boundary from the "covered" set,
        # which keeps the check (never skips it incorrectly).
        all_boundaries = param_info.get_boundary_names()
        every_boundary_has_refinement = bool(all_boundaries) and all_boundaries <= boundaries_with_refinement
        if every_boundary_has_refinement:
            return self

        min_passage_size = self.defaults.min_passage_size if self.defaults.remove_hidden_geometry else None
        validate_geometry_ai_size_ordering(
            geometry_accuracy=self.defaults.geometry_accuracy,
            sealing_size=self.defaults.sealing_size,
            min_passage_size=min_passage_size,
            location="in meshing defaults",
        )
        return self

    @contextual_model_validator(mode="after")
    def _warn_multi_zone_remove_hidden_geometry(self, param_info: ParamsValidationInfo) -> Self:
        """Warn when remove_hidden_geometry is enabled with multiple farfield/custom/seedpoint volume zones.

        Removal only runs under GeometryAI, so the caveat is scoped to it rather than to the flag alone.
        """
        if not param_info.use_geometry_AI or not self.defaults.remove_hidden_geometry:
            return self
        if _count_generated_volume_zones(self.volume_zones) > 1:
            add_validation_warning(
                "Multiple farfield/custom/seedpoint volume zones detected. Removal of hidden geometry "
                "for multi-zone cases is not fully supported and may not work as intended."
            )
        return self

    @property
    def farfield_method(self):
        """Returns the farfield method used."""
        if self.volume_zones:
            for zone in self.volume_zones:
                if isinstance(zone, AutomatedFarfield):
                    return zone.method
                if isinstance(zone, WindTunnelFarfield):
                    return "wind-tunnel"
                if isinstance(zone, UserDefinedFarfield):
                    return "user-defined"
        return None


class VolumeMeshingParams(Flow360BaseModel):
    """
    Volume meshing parameters.
    """

    type_name: Literal["VolumeMeshingParams"] = pd.Field("VolumeMeshingParams", frozen=True)
    defaults: VolumeMeshingDefaults = pd.Field()
    refinement_factor: pd.PositiveFloat | None = pd.Field(
        default=1,
        description="All spacings in refinement regions"
        + "and first layer thickness will be adjusted to generate `r`-times"
        + " finer mesh where r is the refinement_factor value.",
    )

    refinements: list[VolumeRefinementTypes] = pd.Field(
        default=[],
        description="Additional fine-tunning for refinements on top of the global settings",
    )

    planar_face_tolerance: pd.NonNegativeFloat = pd.Field(
        DEFAULT_PLANAR_FACE_TOLERANCE,
        description="Tolerance used for detecting planar faces in the input surface mesh"
        " that need to be remeshed, such as symmetry planes."
        " This tolerance is non-dimensional, and represents a distance"
        " relative to the largest dimension of the bounding box of the input surface mesh."
        " This is only supported by the beta mesher and can not be overridden per face.",
    )

    gap_treatment_strength: float | None = pd.Field(
        default=None,
        ge=0,
        le=1,
        description="Narrow gap treatment strength used when two surfaces are in close proximity."
        " Use a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment."
        " This parameter has a global impact where the anisotropic transition into the isotropic mesh."
        " However the impact on regions without close proximity is negligible.",
    )

    sliding_interface_tolerance: pd.NonNegativeFloat = pd.Field(
        DEFAULT_SLIDING_INTERFACE_TOLERANCE,
        strict=True,
        description="Tolerance used for detecting / creating curves in the input surface mesh / geometry lying on"
        " sliding interfaces. This tolerance is non-dimensional, and represents a distance"
        " relative to the smallest radius of all sliding interfaces specified in meshing parameters."
        " This cannot be overridden per sliding interface.",
    )

    @contextual_model_validator(mode="after")
    def _warn_refinement_factor_ignored_by_beta_mesher(self, param_info: ParamsValidationInfo) -> Self:
        """Warn when refinement_factor is set but the beta mesher will ignore it."""
        _warn_when_refinement_factor_is_ignored(self.refinement_factor, param_info)
        return self

    @contextual_model_validator(mode="after")
    def _check_sizing_against_octree_series(self, param_info: ParamsValidationInfo):
        """Validate that UniformRefinement spacings align with the octree series."""
        if not param_info.is_beta_mesher:
            return self
        if self.defaults.octree_spacing is None:
            logger.warning(
                "No `octree_spacing` configured in `%s`; "
                "octree spacing validation for UniformRefinement will be skipped.",
                type(self.defaults).__name__,
            )
            return self

        if self.refinements is not None:
            for refinement in self.refinements:
                if isinstance(refinement, UniformRefinement):
                    self.defaults.octree_spacing.check_spacing(refinement.spacing, type(refinement).__name__)

        return self

    @contextual_model_validator(mode="after")
    def _check_snappy_uniform_refinement_entities(self, param_info: ParamsValidationInfo):
        """Validate projected UniformRefinement entities are compatible with snappyHexMesh."""
        if not param_info.use_snappy:
            return self
        for refinement in self.refinements:
            if isinstance(refinement, UniformRefinement) and refinement.project_to_surface is not False:
                validate_snappy_uniform_refinement_entities(refinement)
        return self


SurfaceMeshingParams = Annotated[snappy.SurfaceMeshingParams, pd.Field(discriminator="type_name")]


class ModularMeshingWorkflow(Flow360BaseModel):
    """
    Structure consolidating surface and volume meshing parameters.
    """

    type_name: Literal["ModularMeshingWorkflow"] = pd.Field("ModularMeshingWorkflow", frozen=True)
    surface_meshing: SurfaceMeshingParams | None = ContextField(default=None, context=SURFACE_MESH)
    volume_meshing: VolumeMeshingParams | None = ContextField(default=None, context=VOLUME_MESH)
    zones: list[ZoneTypesModular]

    # Meshing outputs (for now, volume mesh slices)
    outputs: list[MeshSliceOutput] = pd.Field(
        default=[],
        description="Mesh output settings.",
    )

    private_attribute_dict: dict | None = pd.Field(None)

    @pd.field_validator("zones", mode="after")
    @classmethod
    def _check_volume_zones_has_farfield(cls, v):
        total_automated_farfield = sum(isinstance(volume_zone, AutomatedFarfield) for volume_zone in v)
        total_user_defined_farfield = sum(isinstance(volume_zone, UserDefinedFarfield) for volume_zone in v)
        total_custom_zones = sum(isinstance(volume_zone, CustomZones) for volume_zone in v)

        if total_custom_zones and total_user_defined_farfield:
            raise ValueError("When using `CustomZones` the `UserDefinedFarfield` will be ignored.")

        if total_automated_farfield > 1:
            raise ValueError("Only one `AutomatedFarfield` zone is allowed in `zones`.")

        if total_user_defined_farfield > 1:
            raise ValueError("Only one `UserDefinedFarfield` zone is allowed in `zones`.")

        if total_automated_farfield + total_user_defined_farfield > 1:
            raise ValueError("Cannot use `AutomatedFarfield` and `UserDefinedFarfield` simultaneously.")

        if (total_user_defined_farfield + total_automated_farfield + total_custom_zones) == 0:
            raise ValueError("At least one zone defining the farfield is required.")

        if total_automated_farfield and total_custom_zones:
            raise ValueError("`CustomZones` cannot be used with `AutomatedFarfield`.")

        return v

    @pd.field_validator("zones", mode="after")
    @classmethod
    def _check_volume_zones_have_unique_names(cls, v):
        """Ensure there won't be duplicated volume zone names."""

        if v is None:
            return v
        to_be_generated_volume_zone_names = set()
        for volume_zone in v:
            if isinstance(volume_zone, CustomZones):
                for custom_volume in volume_zone.entities.stored_entities:
                    if custom_volume.name in to_be_generated_volume_zone_names:
                        raise ValueError(
                            f"Multiple `CustomVolume` with the same name `{custom_volume.name}` are not allowed."
                        )
                    to_be_generated_volume_zone_names.add(custom_volume.name)

        return v

    @contextual_field_validator("zones", mode="after")
    @classmethod
    def _check_enclosed_entities_rotation_volume_association(cls, v, param_info: ParamsValidationInfo):
        """
        Ensure that:
        - enclosed_entities on any farfield requires at least one CustomZone
        - Cylinder, AxisymmetricBody, and Sphere entities in enclosed_entities
          are associated with a RotationVolume or RotationSphere
        """
        if v is None:
            return v

        has_custom_zones = any(isinstance(zone, CustomZones) for zone in v)
        rotation_entity_names = _collect_rotation_entity_names(v, param_info, (RotationVolume, RotationSphere))
        _validate_farfield_enclosed_entities(v, rotation_entity_names, has_custom_zones, param_info)
        custom_volumes = _collect_all_custom_volumes(v)
        _validate_custom_volume_rotation_association(custom_volumes, rotation_entity_names, param_info)

        return v

    @pd.model_validator(mode="after")
    def _check_snappy_zones(self) -> Self:
        total_custom_volumes = 0
        seedpoint_volumes: list[SeedpointVolume] = []
        for zone in self.zones:
            if isinstance(zone, CustomZones):
                for custom_volume in zone.entities.stored_entities:
                    if isinstance(custom_volume, CustomVolume):
                        total_custom_volumes += 1
                    if isinstance(custom_volume, SeedpointVolume):
                        seedpoint_volumes.append(custom_volume)

        if isinstance(self.surface_meshing, snappy.SurfaceMeshingParams):
            if seedpoint_volumes and total_custom_volumes:
                raise ValueError(
                    "Volume zones with snappyHexMeshing are defined using `SeedpointVolume`, not `CustomZones`."
                )

            if self.farfield_method != "auto" and not seedpoint_volumes:
                raise ValueError(
                    "snappyHexMeshing requires at least one `SeedpointVolume` when not using `AutomatedFarfield`."
                )

            _validate_seedpoint_volume_snappy_single_point(seedpoint_volumes)

        return self

    @contextual_field_validator("zones", mode="after")
    @classmethod
    def _check_seedpoint_volume_usage(cls, v, param_info: ParamsValidationInfo):
        """Validate SeedpointVolume usage in modular meshing schema."""
        _validate_seedpoint_volume_mesher_compatibility(_collect_all_seedpoint_volumes(v), param_info)
        _validate_seedpoint_volume_geometry_ai_exclusivity(v, param_info)
        return v

    @contextual_model_validator(mode="after")
    def _check_uniform_refinement_names_not_in_body_ids(self, param_info: ParamsValidationInfo) -> Self:
        """Ensure no UniformRefinement entity shares a name with a body id in the geometry."""

        if not isinstance(self.surface_meshing, snappy.SurfaceMeshingParams):
            return self

        entity_info = param_info.get_entity_info()
        if entity_info is None or getattr(entity_info, "type_name", None) != "GeometryEntityInfo":
            return self

        body_ids = set(entity_info.all_body_ids)
        if not body_ids:
            return self

        conflicting: list[str] = []

        # Surface meshing: all UniformRefinement entities

        if self.surface_meshing is not None and self.surface_meshing.refinements is not None:
            for refinement in self.surface_meshing.refinements:
                if isinstance(refinement, UniformRefinement):
                    for entity in refinement.entities.stored_entities:
                        if entity.name in body_ids:
                            conflicting.append(entity.name)

        # Volume meshing: UniformRefinement entities that project to surface
        # (project_to_surface defaults to True for snappy, so None counts as True)

        if self.volume_meshing is not None and self.volume_meshing.refinements is not None:
            for refinement in self.volume_meshing.refinements:
                if isinstance(refinement, UniformRefinement) and (refinement.project_to_surface is not False):
                    for entity in refinement.entities.stored_entities:
                        if entity.name in body_ids:
                            conflicting.append(entity.name)

        if conflicting:
            names_str = ", ".join(f"`{name}`" for name in dict.fromkeys(conflicting))
            raise ValueError(
                f"UniformRefinement entity name(s) {names_str} conflict with body id(s)"
                " in the geometry. Please use different names for the UniformRefinement entities."
            )

        return self

    @contextual_model_validator(mode="after")
    def _check_no_reused_volume_entities(self) -> Self:
        """
        Meshing entities reuse check.
        +------------------------+------------------------+------------------------+------------------------+
        |                        | RotationCylinder       | AxisymmetricRefinement | UniformRefinement      |
        +------------------------+------------------------+------------------------+------------------------+
        | RotationCylinder       |          NO            |           --           |           --           |
        +------------------------+------------------------+------------------------+------------------------+
        | AxisymmetricRefinement |          NO            |           NO           |           --           |
        +------------------------+------------------------+------------------------+------------------------+
        | UniformRefinement      |          YES           |           NO           |           NO           |
        +------------------------+------------------------+------------------------+------------------------+

        +------------------------+------------------------+------------------------+
        |                        |StructuredBoxRefinement | UniformRefinement      |
        +------------------------+------------------------+------------------------+
        |StructuredBoxRefinement |          NO            |           --           |
        +------------------------+------------------------+------------------------+
        | UniformRefinement      |          NO            |           NO           |
        +------------------------+------------------------+------------------------+

        """

        usage = EntityUsageMap()

        for volume_zone in self.zones if self.zones is not None else []:
            if isinstance(volume_zone, (RotationVolume, RotationSphere)):
                _ = [usage.add_entity_usage(item, volume_zone.type) for item in volume_zone.entities.stored_entities]

        for refinement in (
            self.volume_meshing.refinements
            if (self.volume_meshing is not None and self.volume_meshing.refinements is not None)
            else []
        ):
            if isinstance(
                refinement,
                (UniformRefinement, AxisymmetricRefinement, StructuredBoxRefinement),
            ):
                _ = [
                    usage.add_entity_usage(item, refinement.refinement_type)
                    for item in refinement.entities.stored_entities
                ]

        error_msg = ""
        for entity_type, entity_model_map in usage.dict_entity.items():
            for entity_info in entity_model_map.values():
                if len(entity_info["model_list"]) == 1 or sorted(entity_info["model_list"]) in [
                    sorted(["RotationCylinder", "UniformRefinement"]),
                    sorted(["RotationVolume", "UniformRefinement"]),
                    sorted(["RotationSphere", "UniformRefinement"]),
                ]:
                    continue

                model_set = set(entity_info["model_list"])
                if len(model_set) == 1:
                    error_msg += (
                        f"{entity_type} entity `{entity_info['entity_name']}` "
                        + f"is used multiple times in `{model_set.pop()}`."
                    )
                else:
                    model_string = ", ".join(f"`{x}`" for x in sorted(model_set))
                    error_msg += (
                        f"Using {entity_type} entity `{entity_info['entity_name']}` "
                        + f"in {model_string} at the same time is not allowed."
                    )

        if error_msg:
            raise ValueError(error_msg)

        return self

    @property
    def farfield_method(self):
        """Returns the  farfield method used."""
        if self.zones:
            for zone in self.zones:
                if isinstance(zone, AutomatedFarfield):
                    return zone.method
            return "user-defined"
        return None
