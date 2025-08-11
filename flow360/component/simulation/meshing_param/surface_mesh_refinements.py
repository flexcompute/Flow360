from abc import ABCMeta
from typing import List, Literal, Optional, Union

import pydantic as pd
from typing_extensions import Self

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import SnappyBody, Surface
from flow360.component.simulation.unit_system import AngleType, LengthType
from flow360.log import log


class SnappyEntityRefinement(Flow360BaseModel, metaclass=ABCMeta):
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
                    f"Proximity spacing ({self.proximity_spacing}) was set higher than the minimal spacing ({self.min_spacing}), setting proximity spacing to minimal spacing."
                )
                self.proximity_spacing = self.min_spacing
        return self


class SnappyBodyRefinement(SnappyEntityRefinement):
    refinement_type: Literal["SnappyBodyRefinement"] = pd.Field("SnappyBodyRefinement", frozen=True)
    gap_resolution: Optional[LengthType.NonNegative] = pd.Field(None)
    entities: List[SnappyBody] = pd.Field(alias="bodies")


class SnappyRegionRefinement(SnappyEntityRefinement):
    min_spacing: LengthType.Positive = pd.Field()
    max_spacing: LengthType.Positive = pd.Field()
    refinement_type: Literal["SnappySurfaceRefinement"] = pd.Field(
        "SnappySurfaceRefinement", frozen=True
    )
    entities: EntityList[Surface] = pd.Field(alias="regions")


class SnappySurfaceEdgeRefinement(Flow360BaseModel):
    refinement_type: Literal["SnappySurfaceEdgeRefinement"] = pd.Field(
        "SnappySurfaceEdgeRefinement", frozen=True
    )
    spacing: Optional[Union[LengthType.Positive, List[LengthType.Positive]]] = pd.Field(None)
    distances: Optional[List[LengthType.Positive]] = pd.Field(None)
    min_elem: Optional[pd.NonNegativeInt] = pd.Field(None)
    min_len: Optional[LengthType.NonNegative] = pd.Field(None)
    included_angle: AngleType.Positive = pd.Field(150 * u.deg)
    bodies: Optional[List[SnappyBody]] = pd.Field(None)
    regions: Optional[EntityList[Surface]] = pd.Field(None)
    retain_on_smoothing: Optional[bool] = pd.Field(True)

    @pd.model_validator(mode="after")
    def _check_spacing_format(self) -> Self:
        if isinstance(self.spacing, List):
            if not self.distances or len(self.distances) != len(self.spacing):
                raise ValueError(
                    f"When using a distance spacing specification both spacing ({self.spacing}) and distances ({self.distances}) fields must be Lists and the same length."
                )
        return self

    @pd.model_validator(mode="after")
    def _check_entity_lists(self) -> Self:
        if self.bodies is None and self.regions is None:
            raise ValueError("At least one body or region must be specified.")
        return self
