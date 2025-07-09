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
    gap: LengthType = pd.Field(1 * u.mm)
    entities: List[SnappyBody] = pd.Field(alias="bodies")
    min_spacing: LengthType = pd.Field()
    max_spacing: LengthType = pd.Field()


class SnappySurfaceEdgeRefinement(Flow360BaseModel):
    refinement_type: Literal["SnappySurfaceEdgeRefinement"] = pd.Field(
        "SnappySurfaceEdgeRefinement", frozen=True
    )
    spacing: Union[LengthType, List[LengthType]] = pd.Field()
    distances: Optional[List[LengthType]] = pd.Field([])
    min_elem: Optional[pd.NonNegativeInt] = pd.Field(0)
    min_len: Optional[LengthType] = pd.Field(0)
    entities: Union[EntityList[Surface], List[SnappyBody]] = pd.Field([])

class SnappySurfaceRefinement(Flow360BaseModel):
    refinement_type: Literal["SnappySurfaceRefinement"] = pd.Field(
        "SnappySurfaceRefinement", frozen=True
    )
    min_spacing: LengthType = pd.Field()
    max_spacing: LengthType = pd.Field()
    entities: EntityList[Surface] = pd.Field([])
    # TODO: add gap level increment or equivalent