"""Meshing related parameters for volume and surface mesher."""

from typing import Annotated, List, Optional, Union

import pydantic as pd
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import (
    BoundaryLayer,
    PassiveSpacing,
    SurfaceRefinement,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    RotationCylinder,
    UniformRefinement,
    UserDefinedFarfield,
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
        BoundaryLayer,
        PassiveSpacing,
        UniformRefinement,
        AxisymmetricRefinement,
    ],
    pd.Field(discriminator="refinement_type"),
]

VolumeZonesTypes = Annotated[
    Union[RotationCylinder, AutomatedFarfield, UserDefinedFarfield], pd.Field(discriminator="type")
]


class MeshingDefaults(Flow360BaseModel):
    """
    Default/global settings for meshing parameters.

    Example
    -------

      >>> fl.MeshingDefaults(
      ...     surface_max_edge_length=1*fl.u.m,
      ...     surface_edge_growth_rate=1.2,
      ...     curvature_resolution_angle=12*fl.u.deg,
      ...     boundary_layer_growth_rate=1.1,
      ...     boundary_layer_first_layer_thickness=1e-5*fl.u.m
      ... )

    ====
    """

    # pylint: disable=no-member
    geometry_accuracy: Optional[LengthType.Positive] = pd.Field(
        None,
        description="The smallest length scale that will be resolved accurately by the surface meshing process. "
        "This parameter is only valid when using geometry AI.",
    )

    ##::   Default surface edge settings
    surface_edge_growth_rate: float = ContextField(
        1.2,
        ge=1,
        description="Growth rate of the anisotropic layers grown from the edges."
        "This can not be overridden per edge.",
        context=SURFACE_MESH,
    )

    ##::    Default boundary layer settings
    boundary_layer_growth_rate: float = ContextField(
        1.2,
        description="Default growth rate for volume prism layers.",
        ge=1,
        context=VOLUME_MESH,
    )
    # pylint: disable=no-member
    boundary_layer_first_layer_thickness: Optional[LengthType.Positive] = ConditionalField(
        None,
        description="Default first layer thickness for volumetric anisotropic layers."
        " This can be overridden with :class:`~flow360.BoundaryLayer`.",
        context=VOLUME_MESH,
    )  # Truly optional if all BL faces already have first_layer_thickness

    number_of_boundary_layers: Optional[pd.NonNegativeInt] = pd.Field(
        None,
        description="Default number of volumetric anisotropic layers."
        " The volume mesher will automatically calculate the required"
        " no. of layers to grow the boundary layer elements to isotropic size if not specified."
        " This is only supported by the beta mesher and can not be overridden per face.",
    )

    planar_face_tolerance: pd.NonNegativeFloat = pd.Field(
        1e-6,
        description="Tolerance used for detecting planar faces in the input surface mesh"
        " that need to be remeshed, such as symmetry planes."
        " This tolerance is non-dimensional, and represents a distance"
        " relative to the largest dimension of the bounding box of the input surface mesh."
        " This is only supported by the beta mesher and can not be overridden per face.",
    )

    ##::    Default surface layer settings
    surface_max_edge_length: Optional[LengthType.Positive] = ConditionalField(
        None,
        description="Default maximum edge length for surface cells."
        " This can be overridden with :class:`~flow360.SurfaceRefinement`.",
        context=SURFACE_MESH,
    )
    curvature_resolution_angle: AngleType.Positive = ContextField(
        12 * u.deg,
        description=(
            "Default maximum angular deviation in degrees. This value will restrict:"
            " 1. The angle between a cell’s normal and its underlying surface normal."
            " 2. The angle between a line segment’s normal and its underlying curve normal."
            " This can not be overridden per face."
        ),
        context=SURFACE_MESH,
    )

    @pd.field_validator("number_of_boundary_layers", mode="after")
    @classmethod
    def invalid_number_of_boundary_layers(cls, value):
        """Ensure number of boundary layers is not specified"""
        validation_info = get_validation_info()

        if validation_info is None:
            return value

        if value is not None and not validation_info.is_beta_mesher:
            raise ValueError("Number of boundary layers is only supported by the beta mesher.")
        return value

    @pd.field_validator("planar_face_tolerance", mode="after")
    @classmethod
    def invalid_planar_face_tolerance(cls, value):
        """Ensure planar face tolerance is not specified"""
        validation_info = get_validation_info()

        if validation_info is None:
            return value

        if (
            value != cls.model_fields["planar_face_tolerance"].default
            and not validation_info.is_beta_mesher
        ):
            raise ValueError("Planar face tolerance is only supported by the beta mesher.")
        return value

    @pd.field_validator("geometry_accuracy", mode="after")
    @classmethod
    def invalid_geometry_accuracy(cls, value):
        """Ensure geometry accuracy is not specified when GAI is not used"""
        validation_info = get_validation_info()

        if validation_info is None:
            return value

        if value is None and validation_info.use_geometry_AI:
            raise ValueError("Geometry accuracy is required when geometry AI is used.")
        return value


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
