"""
Flow360 meshing parameters
"""

from typing import List, Optional, Union, get_args

import pydantic as pd
from typing_extensions import Literal

from ..flow360_params.params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
    _self_named_property_validator,
    flow360_json_encoder,
)
from ..types import Axis, Coordinate, NonNegativeFloat, PositiveFloat, Size


class Aniso(Flow360BaseModel):
    """Aniso edge"""

    type = pd.Field("aniso", const=True)
    method: Literal["angle", "height", "aspectRatio"] = pd.Field()
    value: PositiveFloat = pd.Field()
    adapt: Optional[bool] = pd.Field()


class ProjectAniso(Flow360BaseModel):
    """ProjectAniso edge"""

    type = pd.Field("projectAnisoSpacing", const=True)
    adapt: Optional[bool] = pd.Field()


class UseAdjacent(Flow360BaseModel):
    """ProjectAniso edge"""

    type = pd.Field("useAdjacent", const=True)
    adapt: Optional[bool] = pd.Field()


EdgeType = Union[Aniso, ProjectAniso, UseAdjacent]


class _GenericEdgeWrapper(Flow360BaseModel):
    v: EdgeType


class Edges(Flow360SortableBaseModel):
    """:class:`Edges` class for setting up Edges meshing constrains

    Parameters
    ----------
    <edge_name> : EdgeType
        Supported edge types: Union[Aniso, ProjectAniso, UseAdjacent]

    Returns
    -------
    :class:`Edges`
        An instance of the component class Edges.

    Example
    -------
    >>>
    """

    @classmethod
    def get_subtypes(cls) -> list:
        return list(get_args(_GenericEdgeWrapper.__fields__["v"].type_))

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def validate_edge(cls, values):
        """Validator for edge list section

        Raises
        ------
        ValidationError
            When edge is incorrect
        """
        return _self_named_property_validator(
            values, _GenericEdgeWrapper, msg="is not any of supported edge types."
        )


class Face(Flow360BaseModel):
    """Face"""

    max_edge_length: PositiveFloat = pd.Field(alias="maxEdgeLength")
    adapt: Optional[bool] = pd.Field()


FaceType = Face


class _GenericFaceWrapper(Flow360BaseModel):
    v: FaceType


class Faces(Flow360SortableBaseModel):
    """:class:`Faces` class for setting up Faces meshing constrains

    Parameters
    ----------
    <face_name> : Face
        Supported face types: Face(max_edge_lengt=)

    Returns
    -------
    :class:`Faces`
        An instance of the component class Faces.

    Example
    -------
    >>>
    """

    @classmethod
    def get_subtypes(cls) -> list:
        return [_GenericFaceWrapper.__fields__["v"].type_]

    # pylint: disable=no-self-argument
    @pd.root_validator(pre=True)
    def validate_face(cls, values):
        """Validator for face list section

        Raises
        ------
        ValidationError
            When face is incorrect
        """

        return _self_named_property_validator(
            values, _GenericFaceWrapper, msg="is not any of supported face types."
        )


class SurfaceMeshingParams(Flow360BaseModel):
    """
    Flow360 Surface Meshing parameters
    """

    max_edge_length: PositiveFloat = pd.Field(alias="maxEdgeLength")
    edges: Optional[Edges] = pd.Field()
    faces: Optional[Faces] = pd.Field()
    curvature_resolution_angle: Optional[PositiveFloat] = pd.Field(
        alias="curvatureResolutionAngle", default=15
    )
    growth_rate: Optional[PositiveFloat] = pd.Field(alias="growthRate", default=1.2)

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


class Refinement(Flow360BaseModel):
    """Base class for refinement zones"""

    center: Coordinate = pd.Field()
    spacing: PositiveFloat


class BoxRefinement(Refinement):
    """
    Box refinement zone
    """

    type = pd.Field("box", const=True)
    size: Size = pd.Field()
    axis_of_rotation: Optional[Axis] = pd.Field(alias="axisOfRotation", default=(0, 0, 1))
    angle_of_rotation: Optional[float] = pd.Field(alias="angleOfRotation", default=0)


class CylinderRefinement(Refinement):
    """
    Box refinement zone
    """

    type = pd.Field("cylinder", const=True)
    radius: PositiveFloat = pd.Field()
    length: PositiveFloat = pd.Field()
    axis: Axis = pd.Field()


class Farfield(Flow360BaseModel):
    """
    Farfield type for meshing
    """

    type: Literal["auto", "quasi-3d"] = pd.Field()


class Volume(Flow360BaseModel):
    """
    Core volume meshing parameters
    """

    first_layer_thickness: PositiveFloat = pd.Field(alias="firstLayerThickness")
    growth_rate: Optional[PositiveFloat] = pd.Field(alias="growthRate", default=1.2)
    gap_treatment_strength: Optional[pd.confloat(ge=0, le=1)] = pd.Field(
        alias="gapTreatmentStrength"
    )


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
