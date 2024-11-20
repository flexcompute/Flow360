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
from flow360.component.simulation.primitives import Cylinder
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.component.simulation.validation.validation_context import (
    SURFACE_MESH,
    VOLUME_MESH,
    ConditionalField,
    ContextField,
)

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
    """

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
        description="Default growth rate for volume prism layers."
        " This can not be overridden per face.",
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


class MeshingParams(Flow360BaseModel):
    """
    Meshing parameters for volume and/or surface mesher. This contains all the meshing related settings.
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
    def _check_no_reused_cylinder(self) -> Self:
        """
        Check that the RoatatoinCylinder, AxisymmetricRefinement, and UniformRefinement
        do not share the same cylinder.
        """

        class CylinderUsageMap(dict):
            """A customized dict to store the cylinder name and its usage."""

            def __setitem__(self, key, value):
                if key in self:
                    if self[key] != value:
                        raise ValueError(
                            f"The same cylinder named `{key}` is used in both {self[key]} and {value}."
                        )
                    raise ValueError(
                        f"The cylinder named `{key}` is used multiple times in {value}."
                    )
                super().__setitem__(key, value)

        cylinder_name_to_usage_map = CylinderUsageMap()
        for volume_zone in self.volume_zones if self.volume_zones is not None else []:
            if isinstance(volume_zone, RotationCylinder):
                # pylint: disable=protected-access
                for cylinder in [
                    item
                    for item in volume_zone.entities._get_expanded_entities(create_hard_copy=False)
                    if isinstance(item, Cylinder)
                ]:
                    cylinder_name_to_usage_map[cylinder.name] = RotationCylinder.model_fields[
                        "type"
                    ].default

        for refinement in self.refinements if self.refinements is not None else []:
            if isinstance(refinement, (UniformRefinement, AxisymmetricRefinement)):
                # pylint: disable=protected-access
                for cylinder in [
                    item
                    for item in refinement.entities._get_expanded_entities(create_hard_copy=False)
                    if isinstance(item, Cylinder)
                ]:
                    cylinder_name_to_usage_map[cylinder.name] = refinement.refinement_type
        return self
