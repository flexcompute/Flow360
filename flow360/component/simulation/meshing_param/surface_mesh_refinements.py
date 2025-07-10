import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from typing import Annotated, List, Optional, Union, Literal

from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Surface, SnappyBody
from flow360.component.simulation.unit_system import AngleType, LengthType
import flow360.component.simulation.units as u

class SnappyBodyRefinement(Flow360BaseModel):
    refinement_type: Literal["SnappyBodyRefinement"] = pd.Field(
        "SnappyBodyRefinement", frozen=True
    )
    gap_resolution: Optional[LengthType.NonNegative] = pd.Field(None)
    entities: List[SnappyBody] = pd.Field(alias="bodies")
    min_spacing: Optional[LengthType.Positive] = pd.Field(None)
    max_spacing: Optional[LengthType.Positive] = pd.Field(None)
    proximity_spacing: Optional[LengthType.Positive] = pd.Field(None)


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

class SnappyRegionRefinement(Flow360BaseModel):
    refinement_type: Literal["SnappySurfaceRefinement"] = pd.Field(
        "SnappySurfaceRefinement", frozen=True
    )
    min_spacing: Optional[LengthType.Positive] = pd.Field(None)
    max_spacing: Optional[LengthType.Positive] = pd.Field(None)
    entities: EntityList[Surface] = pd.Field([], alias="regions")
    proximity_spacing: Optional[LengthType.Positive] = pd.Field(None)