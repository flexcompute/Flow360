import pydantic as pd

from abc import ABCMeta
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from typing import List, Optional, Union, Literal
from typing_extensions import Self

from flow360.log import log
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Surface, SnappyBody
from flow360.component.simulation.unit_system import AngleType, LengthType

class SnappyEntityRefinement(Flow360BaseModel, metaclass=ABCMeta):
    min_spacing: Optional[LengthType.Positive] = pd.Field(None)
    max_spacing: Optional[LengthType.Positive] = pd.Field(None)
    proximity_spacing: Optional[LengthType.Positive] = pd.Field(None)

    @pd.model_validator(mode="after")
    def _check_spacing_order(self) -> Self:
        if self.min_spacing > self.max_spacing:
            raise ValueError("Minimum spacing must be lower than maximum spacing.")
        return self
    
    @pd.model_validator(mode="after")
    def _check_proximity_spacing(self) -> Self:
        if self.proximity_spacing > self.min_spacing:
            log.warning(f"Proximity spacing ({self.proximity_spacing}) was set higher than the minimal spacing ({self.min_spacing}), setting proximity spacing to minimal spacing.")
            self.proximity_spacing = self.min_spacing
        return self

class SnappyBodyRefinement(SnappyEntityRefinement):
    refinement_type: Literal["SnappyBodyRefinement"] = pd.Field(
        "SnappyBodyRefinement", frozen=True
    )
    gap_resolution: Optional[LengthType.NonNegative] = pd.Field(None)
    entities: List[SnappyBody] = pd.Field([], alias="bodies")

class SnappyRegionRefinement(SnappyEntityRefinement):
    refinement_type: Literal["SnappySurfaceRefinement"] = pd.Field(
        "SnappySurfaceRefinement", frozen=True
    )
    entities: EntityList[Surface] = pd.Field([], alias="regions")

class SnappySurfaceEdgeRefinement(Flow360BaseModel):
    refinement_type: Literal["SnappySurfaceEdgeRefinement"] = pd.Field(
        "SnappySurfaceEdgeRefinement", frozen=True
    )
    spacing: Optional[Union[LengthType.Positive, List[LengthType.Positive]]] = pd.Field(None)
    distances: Optional[List[LengthType.Positive]] = pd.Field(None)
    min_elem: Optional[pd.NonNegativeInt] = pd.Field(None)
    min_len: Optional[LengthType.NonNegative] = pd.Field(None)
    bodies: List[SnappyBody] = pd.Field([])
    regions: EntityList[Surface] = pd.Field([])

    @pd.model_validator(mode="after")
    def _check_spacing_format(self) -> Self:
        if isinstance(self.spacing, List):
            if not self.distances or len(self.distances) == len(self.spacing):
                raise ValueError(f"When using a distance spacing specification both spacing ({self.spacing}) and distances ({self.distances}) fields must be Lists and the same length.")
            return self