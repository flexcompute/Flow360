"""Meshing related parameters for volume and surface mesher."""

from typing import Annotated, List, Optional, Union

import pydantic as pd
from typing_extensions import Self

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.meshing_param.edge_params import SurfaceEdgeRefinement
from flow360.component.simulation.meshing_param.face_params import (
    SurfaceRefinementTypes,
)
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    AxisymmetricRefinement,
    RotationCylinder,
    UniformRefinement,
    VolumeRefinementTypes,
)
from flow360.component.simulation.primitives import Cylinder

RefinementTypes = Annotated[
    Union[SurfaceEdgeRefinement, SurfaceRefinementTypes, VolumeRefinementTypes],
    pd.Field(discriminator="refinement_type"),
]

VolumeZonesTypes = Annotated[
    Union[RotationCylinder, AutomatedFarfield], pd.Field(discriminator="type")
]


class MeshingParams(Flow360BaseModel):
    """
    Meshing parameters for volume and/or surface mesher.

    In `Simulation` this only contains what the user specifies. `Simulation` can derive and add more items according
    to other aspects of simulation. (E.g. BETDisk volume -> ZoneRefinement)

    Meshing related but may and maynot (user specified) need info from `Simulation`:
    1. Add rotational zones.
    2. Add default BETDisk refinement.

    Affects volume meshing:
    - refinement_factor
    - gap_treatment_strength
    - `class` BoundaryLayer
    - `class` UniformRefinement
    - `class` AxisymmetricRefinement
    - `class` RotationCylinder

    Affects surface meshing:
    - surface_layer_growth_rate
    - `class` SurfaceRefinement
    - `class` SurfaceEdgeRefinement
    """

    # Volume **defaults**:
    refinement_factor: Optional[pd.PositiveFloat] = pd.Field(
        default=1,
        description="""If refinementFactor=r is provided all spacings in refinementregions
        and first layer thickness will be adjusted to generate r-times finer mesh.""",
    )
    gap_treatment_strength: Optional[float] = pd.Field(
        default=0,
        ge=0,
        le=1,
        description="""Narrow gap treatment strength used when two surfaces are in close proximity.
        Use a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment.
        This parameter has a global impact where the anisotropic transition into the isotropic mesh.
        However the impact on regions without close proximity is negligible.""",
    )

    surface_layer_growth_rate: float = pd.Field(
        1.2, ge=1, description="Global growth rate of the anisotropic layers grown from the edges."
    )  # Conditionally optional

    refinements: List[RefinementTypes] = pd.Field(
        default=[],
        description="Additional fine-tunning for refinements.",
    )
    # Will add more to the Union
    volume_zones: Optional[List[VolumeZonesTypes]] = pd.Field(
        default=None, description="Creation of new volume zones."
    )

    @pd.field_validator("volume_zones", mode="after")
    @classmethod
    def _finalize_automated_farfield(cls, v) -> Self:
        if v is None:
            # User did not put anything in volume_zones so may not want to use volume meshing
            return v

        has_rotating_zone = any(isinstance(volume_zone, RotationCylinder) for volume_zone in v)
        # pylint: disable=expression-not-assigned, protected-access
        [
            volume_zone._set_up_zone_entity(has_rotating_zone)
            for volume_zone in v
            if isinstance(volume_zone, AutomatedFarfield)
        ]
        return v

    @pd.field_validator("volume_zones", mode="after")
    @classmethod
    def _check_volume_zones_has_farfied(cls, v):
        if v is None:
            # User did not put anything in volume_zones so may not want to use volume meshing
            return v

        has_farfield = any(isinstance(volume_zone, AutomatedFarfield) for volume_zone in v)
        if not has_farfield:
            raise ValueError("AutomatedFarfield is required in volume_zones.")
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
                    raise ValueError(
                        f"The same cylinder named `{key}` is used in both {self[key]} and {value}."
                    )
                super().__setitem__(key, value)

        cylinder_name_to_usage_map = CylinderUsageMap()
        for volume_zone in self.volume_zones if self.volume_zones is not None else []:
            if isinstance(volume_zone, RotationCylinder):
                # pylint: disable=protected-access
                for cylinder in [
                    item
                    for item in volume_zone.entities._get_expanded_entities()
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
                    for item in refinement.entities._get_expanded_entities()
                    if isinstance(item, Cylinder)
                ]:
                    cylinder_name_to_usage_map[cylinder.name] = refinement.refinement_type
        return self
