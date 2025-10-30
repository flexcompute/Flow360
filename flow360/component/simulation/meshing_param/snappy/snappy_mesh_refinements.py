"""Reinements for surface meshing"""

from abc import ABCMeta
from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.primitives import SnappyBody, Surface
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.log import log


class SnappyEntityRefinement(Flow360BaseModel, metaclass=ABCMeta):
    """
    Base refinement for snappyHexMesh.
    """

    # pylint: disable=no-member
    min_spacing: Optional[LengthType.Positive] = pd.Field(None)
    max_spacing: Optional[LengthType.Positive] = pd.Field(None)
    proximity_spacing: Optional[LengthType.Positive] = pd.Field(None)

    @pd.model_validator(mode="after")
    def _check_spacing_order(self) -> Self:
        if self.min_spacing and self.max_spacing:
            if self.min_spacing > self.max_spacing:
                raise ValueError("Minimum spacing must be lower than maximum spacing.")
        return self

    @pd.model_validator(mode="after")
    def _check_proximity_spacing(self) -> Self:
        if self.min_spacing and self.proximity_spacing:
            if self.proximity_spacing > self.min_spacing:
                log.warning(
                    f"Proximity spacing ({self.proximity_spacing}) was set higher than the minimal spacing"
                    + f"({self.min_spacing}), setting proximity spacing to minimal spacing."
                )
                self.proximity_spacing = self.min_spacing
        return self


class BodyRefinement(SnappyEntityRefinement):
    """
    Refinement for snappyHexMesh body (searchableSurfaceWithGaps).

    Parameters
    ----------
    gap_resolution: Optional[LengthType.NonNegative], default: None
    """

    # pylint: disable=no-member
    refinement_type: Literal["SnappyBodyRefinement"] = pd.Field("SnappyBodyRefinement", frozen=True)
    gap_resolution: Optional[LengthType.NonNegative] = pd.Field(None)
    entities: EntityList[SnappyBody] = pd.Field(alias="bodies")


class RegionRefinement(SnappyEntityRefinement):
    """
    Refinement for the body region in snappyHexMesh.

    Parameters
    ----------
    min_spacing: LengthType.Positive
    max_spacing: LengthType.Positive
    """

    # pylint: disable=no-member
    min_spacing: LengthType.Positive = pd.Field()
    max_spacing: LengthType.Positive = pd.Field()
    refinement_type: Literal["SnappySurfaceRefinement"] = pd.Field(
        "SnappySurfaceRefinement", frozen=True
    )
    entities: EntityList[Surface] = pd.Field(alias="regions")


class SurfaceEdgeRefinement(Flow360BaseModel):
    """
    Edge refinement for bodies and regions in snappyHexMesh.

    Parameters
    ----------
    spacing: Optional[Union[LengthType.Positive, LengthType.PositiveArray]], default: None
        Spacing on and close to the edges.
        Defaults to default min_spacing.

    distances: Optional[List[LengthType.Positive]], default: None
        Distance from the edge where to apply the spacings.
        Set to None to disable this metric.

    min_elem: Optional[pd.NonNegativeInt], default: None
        Minimum number of elements on the edge to apply the edge refinement.
        Set to None to disable this metric.

    min_len: Optional[LengthType.NonNegative], default: None
        Minimum length of the edge to apply edge refinement.
        Set to None to disable this metric.

    included_angle: AngleType.Positive, default: 150°
        If the angle between two elements is less than this value, the edge is extracted as a feature.

    bodies: Optional[List[SnappyBody]], default: None
    regions: Optional[EntityList[Surface]], default: None

    retain_on_smoothing: Optional[bool], default: True
        Maintain the edge when smoothing is applied.
        Set to None to disable this metric.
    """

    # pylint: disable=no-member
    refinement_type: Literal["SnappySurfaceEdgeRefinement"] = pd.Field(
        "SnappySurfaceEdgeRefinement", frozen=True
    )
    spacing: Optional[Union[LengthType.PositiveArray, LengthType.Positive]] = pd.Field(None)
    distances: Optional[LengthType.PositiveArray] = pd.Field(None)
    min_elem: Optional[pd.NonNegativeInt] = pd.Field(None)
    min_len: Optional[LengthType.NonNegative] = pd.Field(None)
    included_angle: AngleType.Positive = pd.Field(150 * u.deg)
    bodies: Optional[EntityList[SnappyBody]] = pd.Field(None)
    regions: Optional[EntityList[Surface]] = pd.Field(None)
    retain_on_smoothing: Optional[bool] = pd.Field(True)

    @pd.model_validator(mode="after")
    def _check_spacing_format(self) -> Self:
        distances_state = None
        spacing_state = None
        if self.distances is not None:
            distances_state = (True, len(self.distances))
        else:
            distances_state = (False, 0)

        try:
            spacing_state = (True, len(self.spacing))
        except TypeError:
            # spacing is a scalar
            spacing_state = (False, 1)

        if (
            distances_state[0] and spacing_state[0] and (spacing_state[1] != distances_state[1])
        ) or (distances_state[0] is not spacing_state[0]):
            raise ValueError(
                f"When using a distance spacing specification both spacing ({self.spacing}) and distances"
                + f"({self.distances}) fields must be arrays and the same length."
            )
        return self

    @pd.field_validator("spacing", "distances", mode="after")
    @classmethod
    def _check_spacings_increasing(cls, value):
        if value is not None:
            if isinstance(value.tolist(), List) and (sorted(value.tolist()) != value.tolist()):
                raise ValueError("Spacings and distances must be increasing arrays.")
        return value

    @pd.field_validator("spacing", "distances", mode="before")
    @classmethod
    def _convert_list_to_unyt_array(cls, value):
        if isinstance(value, List):
            return u.unyt.unyt_array(value)
        return value

    @pd.model_validator(mode="after")
    def _check_entity_lists(self) -> Self:
        if self.bodies is None and self.regions is None:
            raise ValueError("At least one body or region must be specified.")
        return self


SnappySurfaceRefinementTypes = Annotated[
    Union[BodyRefinement, SurfaceEdgeRefinement, RegionRefinement, UniformRefinement],
    pd.Field(discriminator="refinement_type"),
]
