"""Meshing related parameters for volume and surface mesher."""

from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd
from typing_extensions import Self

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater import DEFAULT_PLANAR_FACE_TOLERANCE
from flow360.component.simulation.meshing_param import snappy
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    GeometryRefinement,
    PassiveSpacing,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.meshing_specs import (
    MeshingDefaults,
    VolumeMeshingDefaults,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    CustomZones,
    RotationCylinder,
    RotationVolume,
    StructuredBoxRefinement,
    UniformRefinement,
    UserDefinedFarfield,
)
from flow360.component.simulation.primitives import SeedpointZone
from flow360.component.simulation.validation.validation_context import (
    SURFACE_MESH,
    VOLUME_MESH,
    ContextField,
)
from flow360.component.simulation.validation.validation_utils import EntityUsageMap

RefinementTypes = Annotated[
    Union[
        SurfaceEdgeRefinement,
        SurfaceRefinement,
        GeometryRefinement,
        BoundaryLayer,
        PassiveSpacing,
        UniformRefinement,
        StructuredBoxRefinement,
        AxisymmetricRefinement,
    ],
    pd.Field(discriminator="refinement_type"),
]

VolumeZonesTypes = Annotated[
    Union[
        RotationVolume,
        RotationCylinder,
        AutomatedFarfield,
        UserDefinedFarfield,
        SeedpointZone,
        CustomZones,
    ],
    pd.Field(discriminator="type"),
]

SurfaceRefinementTypes = Annotated[
    Union[
        SurfaceEdgeRefinement,
        SurfaceRefinement,
    ],
    pd.Field(discriminator="refinement_type"),
]

VolumeRefinementTypes = Annotated[
    Union[
        UniformRefinement,
        AxisymmetricRefinement,
        BoundaryLayer,
        PassiveSpacing,
        StructuredBoxRefinement,
    ],
    pd.Field(discriminator="refinement_type"),
]


class MeshingParams(Flow360BaseModel):
    """
    Meshing parameters for volume and/or surface mesher. This contains all the meshing related settings.

    Example
    -------

      >>> fl.MeshingParams(
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
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field(
        default=1,
        description="All spacings in refinement regions"
        + "and first layer thickness will be adjusted to generate `r`-times"
        + " finer mesh where r is the refinement_factor value.",
    )

    # pylint: disable=duplicate-code
    gap_treatment_strength: Optional[float] = ContextField(
        default=0,
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

    refinements: List[RefinementTypes] = pd.Field(
        default=[],
        description="Additional fine-tunning for refinements on top of :py:attr:`defaults`",
    )
    # Will add more to the Union
    volume_zones: Optional[List[VolumeZonesTypes]] = pd.Field(
        default=None, description="Creation of new volume zones."
    )

    @pd.field_validator("volume_zones", mode="after")
    @classmethod
    def _check_volume_zones_has_farfied(cls, v):
        if v is None:
            # User did not put anything in volume_zones so may not want to use volume meshing
            return v

        total_farfield = sum(
            isinstance(volume_zone, (AutomatedFarfield, UserDefinedFarfield)) for volume_zone in v
        )
        if total_farfield == 0:
            raise ValueError("Farfield zone is required in `volume_zones`.")

        if total_farfield > 1:
            raise ValueError("Only one farfield zone is allowed in `volume_zones`.")

        return v

    @pd.field_validator("volume_zones", mode="after")
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

    @pd.model_validator(mode="after")
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
            if isinstance(volume_zone, (RotationVolume, RotationCylinder)):
                # pylint: disable=protected-access
                _ = [
                    usage.add_entity_usage(item, volume_zone.type)
                    for item in volume_zone.entities._get_expanded_entities(create_hard_copy=False)
                ]

        for refinement in self.refinements if self.refinements is not None else []:
            if isinstance(
                refinement,
                (UniformRefinement, AxisymmetricRefinement, StructuredBoxRefinement),
            ):
                # pylint: disable=protected-access
                _ = [
                    usage.add_entity_usage(item, refinement.refinement_type)
                    for item in refinement.entities._get_expanded_entities(create_hard_copy=False)
                ]

        error_msg = ""
        for entity_type, entity_model_map in usage.dict_entity.items():
            for entity_info in entity_model_map.values():
                if len(entity_info["model_list"]) == 1 or sorted(entity_info["model_list"]) in [
                    sorted(["RotationCylinder", "UniformRefinement"]),
                    sorted(["RotationVolume", "UniformRefinement"]),
                ]:
                    # RotationCylinder and UniformRefinement are allowed to be used together
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
        if self.volume_zones:
            for zone in self.volume_zones:  # pylint: disable=not-an-iterable
                if isinstance(zone, AutomatedFarfield):
                    return zone.method
            return "user-defined"
        return None


class VolumeMeshingParams(Flow360BaseModel):
    """
    Volume meshing parameters.
    """

    type_name: Literal["VolumeMeshingParams"] = pd.Field("VolumeMeshingParams", frozen=True)
    defaults: VolumeMeshingDefaults = pd.Field()
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field(
        default=1,
        description="All spacings in refinement regions"
        + "and first layer thickness will be adjusted to generate `r`-times"
        + " finer mesh where r is the refinement_factor value.",
    )

    refinements: List[VolumeRefinementTypes] = pd.Field(
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

    gap_treatment_strength: Optional[float] = pd.Field(
        default=0,
        ge=0,
        le=1,
        description="Narrow gap treatment strength used when two surfaces are in close proximity."
        " Use a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment."
        " This parameter has a global impact where the anisotropic transition into the isotropic mesh."
        " However the impact on regions without close proximity is negligible.",
    )


SurfaceMeshingParams = Annotated[
    Union[snappy.SurfaceMeshingParams], pd.Field(discriminator="type_name")
]


class ModularMeshingWorkflow(Flow360BaseModel):
    """
    Structure consolidating surface and volume meshing parameters.
    """

    type_name: Literal["ModularMeshingWorkflow"] = pd.Field("ModularMeshingWorkflow", frozen=True)
    surface_meshing: Optional[SurfaceMeshingParams] = ContextField(
        default=None, context=SURFACE_MESH
    )
    volume_meshing: Optional[VolumeMeshingParams] = ContextField(default=None, context=VOLUME_MESH)
    zones: List[VolumeZonesTypes]

    @pd.field_validator("zones", mode="after")
    @classmethod
    def _check_volume_zones_has_farfied(cls, v):

        total_automated_farfield = sum(
            isinstance(volume_zone, AutomatedFarfield) for volume_zone in v
        )
        total_user_defined_farfield = sum(
            isinstance(volume_zone, UserDefinedFarfield) for volume_zone in v
        )
        total_custom_zones = sum(isinstance(volume_zone, CustomZones) for volume_zone in v)
        total_seedpoint_zone = sum(isinstance(volume_zone, SeedpointZone) for volume_zone in v)

        if (total_custom_zones or total_seedpoint_zone) and total_user_defined_farfield:
            raise ValueError(
                "When using :class:`CustomZones` or SeedpointZone the UserDefinedFarfield will be ignored."
            )

        if total_automated_farfield > 1:
            raise ValueError("Only one AutomatedFarfield zone is allowed in `volume_zones`.")

        if total_user_defined_farfield > 1:
            raise ValueError("Only one UserDefinedFarfield zone is allowed in `volume_zones`.")

        if total_automated_farfield + total_user_defined_farfield > 1:
            raise ValueError("Cannot use AutomatedFarfield and UserDefinedFarfield simultaneously.")

        if (
            total_user_defined_farfield
            + total_automated_farfield
            + total_custom_zones
            + total_seedpoint_zone
        ) == 0:
            raise ValueError("At least one zone defining the farfield is required.")

        if total_automated_farfield and (total_seedpoint_zone or total_custom_zones):
            raise ValueError(
                "SeedpointZone and CustomVolume cannot be used with AutomatedFarfield."
            )

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
                            f"Multiple CustomVolume with the same name `{custom_volume.name}` are not allowed."
                        )
                    to_be_generated_volume_zone_names.add(custom_volume.name)

            if isinstance(volume_zone, SeedpointZone):
                if volume_zone.name in to_be_generated_volume_zone_names:
                    raise ValueError(
                        f"Multiple CustomVolume or SeedpointZone with the same name `{volume_zone.name}` "
                        + "are not allowed."
                    )
                to_be_generated_volume_zone_names.add(volume_zone.name)

        return v

    @pd.model_validator(mode="after")
    def _check_snappy_zones(self) -> Self:
        if isinstance(self.surface_meshing, snappy.SurfaceMeshingParams):
            if self.farfield_method != "auto" and not sum(
                isinstance(volume_zone, SeedpointZone) for volume_zone in self.zones
            ):
                raise ValueError(
                    "snappyHexMeshing requires at least one SeedpointZone when not using AutomatedFarfield."
                )
            for zone in self.zones:  # pylint: disable=not-an-iterable
                if isinstance(zone, CustomZones):
                    raise ValueError(
                        "Volume zones with snappyHexMeshing are defined using SeedpointZone, not CustomZones."
                    )
        else:
            for zone in self.zones:  # pylint: disable=not-an-iterable
                if isinstance(zone, SeedpointZone):
                    raise ValueError("SeedpointZone is applicable only with snappyHexMeshing.")

        return self

    @pd.model_validator(mode="after")
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
            if isinstance(volume_zone, (RotationVolume, RotationCylinder)):
                # pylint: disable=protected-access
                _ = [
                    usage.add_entity_usage(item, volume_zone.type)
                    for item in volume_zone.entities._get_expanded_entities(create_hard_copy=False)
                ]
        # pylint: disable=no-member
        for refinement in (
            self.volume_meshing.refinements
            if (self.volume_meshing is not None and self.volume_meshing.refinements is not None)
            else []
        ):
            if isinstance(
                refinement,
                (UniformRefinement, AxisymmetricRefinement, StructuredBoxRefinement),
            ):
                # pylint: disable=protected-access
                _ = [
                    usage.add_entity_usage(item, refinement.refinement_type)
                    for item in refinement.entities._get_expanded_entities(create_hard_copy=False)
                ]

        error_msg = ""
        for entity_type, entity_model_map in usage.dict_entity.items():
            for entity_info in entity_model_map.values():
                if len(entity_info["model_list"]) == 1 or sorted(entity_info["model_list"]) in [
                    sorted(["RotationCylinder", "UniformRefinement"]),
                    sorted(["RotationVolume", "UniformRefinement"]),
                ]:
                    # RotationCylinder and UniformRefinement are allowed to be used together
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
            for zone in self.zones:  # pylint: disable=not-an-iterable
                if isinstance(zone, AutomatedFarfield):
                    return zone.method
            return "user-defined"
        return None
