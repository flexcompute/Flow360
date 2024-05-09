"""
Flow360 meshing parameters
"""

from typing import List, Optional, Union, get_args

import pydantic.v1 as pd
from typing_extensions import Literal

from flow360.flags import Flags

from ..flow360_params.params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _self_named_property_validator,
    flow360_json_encoder,
)
from ..types import Axis, Coordinate, NonNegativeFloat, PositiveFloat, Size


class Refinement(Flow360BaseModel):
    """Base class for refinement zones"""

    center: Coordinate = pd.Field()
    spacing: PositiveFloat


class Volume(Flow360BaseModel):
    """
    Core volume meshing parameters
    """

    if Flags.beta_features():
        num_boundary_layers: Optional[pd.conint(ge=0)] = pd.Field(alias="numBoundaryLayers")
        surface_boundaries: Optional[List[str]] = pd.Field(alias="surfaceBoundaries")


class RotationalModelBase(Flow360BaseModel):
    """:class: RotorDisk"""

    name: Optional[str] = pd.Field()
    inner_radius: Optional[NonNegativeFloat] = pd.Field(alias="innerRadius", default=0)
    outer_radius: PositiveFloat = pd.Field(alias="outerRadius")
    thickness: PositiveFloat = pd.Field()
    center: Coordinate = pd.Field()
    spacing_axial: PositiveFloat = pd.Field(alias="spacingAxial")
    spacing_radial: PositiveFloat = pd.Field(alias="spacingRadial")
    spacing_circumferential: PositiveFloat = pd.Field(alias="spacingCircumferential")


class RotorDisk(RotationalModelBase):
    """:class: RotorDisk"""

    axis_thrust: Axis = pd.Field(alias="axisThrust")


class SlidingInterface(RotationalModelBase):
    """:class: SlidingInterface for meshing"""

    axis_of_rotation: Axis = pd.Field(alias="axisOfRotation")
    enclosed_objects: Optional[List[str]] = pd.Field(alias="enclosedObjects", default=[])


class VolumeMeshingParams(Flow360BaseModel):
    """
    Flow360 Volume Meshing parameters
    """

    volume: Volume = pd.Field()
    refinement_factor: Optional[PositiveFloat] = pd.Field(alias="refinementFactor")
    farfield: Optional[Farfield] = pd.Field()
    refinement: Optional[List[Union[BoxRefinement, CylinderRefinement]]] = pd.Field()
    rotor_disks: Optional[List[RotorDisk]] = pd.Field(alias="rotorDisks")
    sliding_interfaces: Optional[List[SlidingInterface]] = pd.Field(alias="slidingInterfaces")

    if Flags.beta_features():
        version: Optional[Literal["v1", "v2"]] = pd.Field(alias="version", default="v1")

    def flow360_json(self) -> str:
        """Generate a JSON representation of the model, as required by Flow360

        Returns
        -------
        json
            Returns JSON representation of the model.

        Example
        -------
        >>> params.flow360_json() # doctest: +SKIP
        """

        return self.json(encoder=flow360_json_encoder)
