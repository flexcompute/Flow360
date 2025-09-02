"""Meshing related parameters for volume and surface mesher."""

from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.updater import DEFAULT_PLANAR_FACE_TOLERANCE
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    GeometryRefinement,
    PassiveSpacing,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.meshing_specs import (
    BetaVolumeMeshingDefaults,
    MeshingDefaults,
    SnappyCastellatedMeshControls,
    SnappyQualityMetrics,
    SnappySmoothControls,
    SnappySnapControls,
    SnappySurfaceMeshingDefaults,
)
from flow360.component.simulation.meshing_param.surface_mesh_refinements import (
    SnappyBodyRefinement,
    SnappyRegionRefinement,
    SnappySurfaceEdgeRefinement,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    RotationCylinder,
    UniformRefinement,
    UserDefinedFarfield,
)
from flow360.component.simulation.primitives import (
    Box,
    CustomVolume,
    Cylinder,
    MeshZone,
)
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.simulation.validation.validation_context import (
    SURFACE_MESH,
    VOLUME_MESH,
    ConditionalField,
    ContextField,
    get_validation_info,
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
        AxisymmetricRefinement,
    ],
    pd.Field(discriminator="refinement_type"),
]

VolumeZonesTypes = Annotated[
    Union[RotationCylinder, AutomatedFarfield, UserDefinedFarfield, CustomVolume],
    pd.Field(discriminator="type"),
]

SurfaceRefinementTypes = Annotated[
    Union[
        SurfaceEdgeRefinement,
        SurfaceRefinement,
    ],
    pd.Field(discriminator="refinement_type"),
]

SnappySurfaceRefinementTypes = Annotated[
    Union[
        SnappyBodyRefinement, SnappySurfaceEdgeRefinement, SnappyRegionRefinement, UniformRefinement
    ],
    pd.Field(discriminator="refinement_type"),
]

VolumeRefinementTypes = Annotated[
    Union[
        UniformRefinement,
        AxisymmetricRefinement,
        BoundaryLayer,
        PassiveSpacing,
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

    type: Literal["MeshingParams"] = pd.Field("MeshingParams", frozen=True)
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field(
        default=1,
        description="All spacings in refinement regions"
        + "and first layer thickness will be adjusted to generate `r`-times"
        + " finer mesh where r is the refinement_factor value.",
    )

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
            if not isinstance(volume_zone, CustomVolume):
                continue
            if volume_zone.name in to_be_generated_volume_zone_names:
                raise ValueError(
                    f"Multiple CustomVolume with the same name `{volume_zone.name}` are not allowed."
                )
            to_be_generated_volume_zone_names.add(volume_zone.name)

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

        """

        usage = EntityUsageMap()

        for volume_zone in self.volume_zones if self.volume_zones is not None else []:
            if isinstance(volume_zone, RotationCylinder):
                # pylint: disable=protected-access
                _ = [
                    usage.add_entity_usage(item, volume_zone.type)
                    for item in volume_zone.entities._get_expanded_entities(create_hard_copy=False)
                ]

        for refinement in self.refinements if self.refinements is not None else []:
            if isinstance(refinement, (UniformRefinement, AxisymmetricRefinement)):
                # pylint: disable=protected-access
                _ = [
                    usage.add_entity_usage(item, refinement.refinement_type)
                    for item in refinement.entities._get_expanded_entities(create_hard_copy=False)
                ]

        error_msg = ""
        for entity_type, entity_model_map in usage.dict_entity.items():
            for entity_info in entity_model_map.values():
                if len(entity_info["model_list"]) == 1 or sorted(
                    entity_info["model_list"]
                ) == sorted(["RotationCylinder", "UniformRefinement"]):
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
    def automated_farfield_method(self):
        """Returns the automated farfield method used."""
        if self.volume_zones:
            for zone in self.volume_zones:  # pylint: disable=not-an-iterable
                if isinstance(zone, AutomatedFarfield):
                    return zone.method
        return None


class SnappySurfaceMeshingParams(Flow360BaseModel):
    type: Literal["SnappySurfaceMeshingParams"] = pd.Field(
        "SnappySurfaceMeshingParams", frozen=True
    )
    defaults: SnappySurfaceMeshingDefaults = pd.Field()
    quality_metrics: SnappyQualityMetrics = pd.Field(SnappyQualityMetrics())
    snap_controls: SnappySnapControls = pd.Field(SnappySnapControls())
    castellated_mesh_controls: SnappyCastellatedMeshControls = pd.Field(
        SnappyCastellatedMeshControls()
    )
    smooth_controls: Optional[SnappySmoothControls] = pd.Field(None)
    bounding_box: Optional[Box] = pd.Field(None)
    zones: Optional[List[MeshZone]] = pd.Field(None)
    cad_is_fluid: bool = pd.Field(False)
    refinements: Optional[List[SnappySurfaceRefinementTypes]] = pd.Field([])

    @pd.model_validator(mode="after")
    def _ensure_mesh_zone_provided(self):
        if self.cad_is_fluid and self.zones is None:
            raise ValueError("Mesh zones must be specified when cad is fluid.")
        return self

    @pd.model_validator(mode="after")
    def _check_body_refinements_w_defaults(self):
        # set body refinements
        for refinement in self.refinements:
            if isinstance(refinement, SnappyBodyRefinement):
                if refinement.min_spacing is None and refinement.max_spacing is None:
                    continue
                if refinement.min_spacing is None and self.defaults.min_spacing.to(
                    "m"
                ) > refinement.max_spacing.to("m"):
                    raise ValueError(
                        "Default minimum spacing is higher that refinement maximum spacing and minimum spacing is not provided."
                    )
                if refinement.max_spacing is None and self.defaults.max_spacing.to(
                    "m"
                ) < refinement.min_spacing.to("m"):
                    raise ValueError(
                        "Default maximum spacing is lower that refinement minimum spacing and maximum spacing is not provided."
                    )
        return self

    @pd.model_validator(mode="after")
    def _check_uniform_refinement_entities(self):
        for refinement in self.refinements:
            if isinstance(refinement, UniformRefinement):
                for entity in refinement.entities.stored_entities:
                    if isinstance(entity, Box) and entity.angle_of_rotation.to("deg") != 0 * u.deg:
                        raise ValueError(
                            "UniformRefinement for snappy accepts only Boxes with axes aligned with the global coordinate system (angle_of_rotation=0)."
                        )
                    if isinstance(entity, Cylinder) and entity.inner_radius.to("m") != 0 * u.m:
                        raise ValueError(
                            "UniformRefinement for snappy accepts only full cylinders (where inner_radius = 0)."
                        )

        return self


class BetaVolumeMeshingParams(Flow360BaseModel):
    type: Literal["BetaVolumeMeshingParams"] = pd.Field("BetaVolumeMeshingParams", frozen=True)
    defaults: BetaVolumeMeshingDefaults = pd.Field(BetaVolumeMeshingDefaults())
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field(
        default=1,
        description="All spacings in refinement regions"
        + "and first layer thickness will be adjusted to generate `r`-times"
        + " finer mesh where r is the refinement_factor value.",
    )

    volume_zones: Optional[List[VolumeZonesTypes]] = pd.Field(
        default=None, description="Creation of new volume zones."
    )

    refinements: List[VolumeRefinementTypes] = pd.Field(
        default=[],
        description="Additional fine-tunning for refinements on top of the global settings",
    )

    planar_face_tolerance: pd.NonNegativeFloat = pd.Field(
        1e-6,
        description="Tolerance used for detecting planar faces in the input surface mesh"
        " that need to be remeshed, such as symmetry planes."
        " This tolerance is non-dimensional, and represents a distance"
        " relative to the largest dimension of the bounding box of the input surface mesh."
        " This is only supported by the beta mesher and can not be overridden per face.",
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

        """

        usage = EntityUsageMap()

        for volume_zone in self.volume_zones if self.volume_zones is not None else []:
            if isinstance(volume_zone, RotationCylinder):
                # pylint: disable=protected-access
                _ = [
                    usage.add_entity_usage(item, volume_zone.type)
                    for item in volume_zone.entities._get_expanded_entities(create_hard_copy=False)
                ]

        for refinement in self.refinements if self.refinements is not None else []:
            if isinstance(refinement, (UniformRefinement, AxisymmetricRefinement)):
                # pylint: disable=protected-access
                _ = [
                    usage.add_entity_usage(item, refinement.refinement_type)
                    for item in refinement.entities._get_expanded_entities(create_hard_copy=False)
                ]

        error_msg = ""
        for entity_type, entity_model_map in usage.dict_entity.items():
            for entity_info in entity_model_map.values():
                if len(entity_info["model_list"]) == 1 or sorted(
                    entity_info["model_list"]
                ) == sorted(["RotationCylinder", "UniformRefinement"]):
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
    def automated_farfield_method(self):
        """Returns the automated farfield method used."""
        if self.volume_zones:
            for zone in self.volume_zones:  # pylint: disable=not-an-iterable
                if isinstance(zone, AutomatedFarfield):
                    return zone.method
        return None


SurfaceMeshingParams = Annotated[Union[SnappySurfaceMeshingParams], pd.Field(discriminator="type")]
VolumeMeshingParams = Annotated[Union[BetaVolumeMeshingParams], pd.Field(discriminator="type")]


class ModularMeshingWorkflow(Flow360BaseModel):
    type: Literal["ModularMeshingWorkflow"] = pd.Field("ModularMeshingWorkflow", frozen=True)
    surface_meshing: Optional[SurfaceMeshingParams] = ContextField(
        default=None, context=SURFACE_MESH
    )
    volume_meshing: Optional[VolumeMeshingParams] = ContextField(default=None, context=VOLUME_MESH)
